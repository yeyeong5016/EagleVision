import copy
from typing import List, Optional, Tuple

import torch
from mmdet.models.utils import (filter_scores_and_topk, multi_apply,
                                select_single_mlvl, sigmoid_geometric_mean,
                                unmap, images_to_levels, unpack_gt_instances,
                                empty_instances)
from mmdet.utils import (ConfigType, InstanceList, OptConfigType,
                         OptInstanceList, reduce_mean)
from mmengine import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor, nn

from mmrotate.registry import MODELS, TASK_UTILS
from mmdet.models.roi_heads import ConvFCBBoxHead, StandardRoIHead
from internvl.model.internvl_chat import InternVLChatModel, InternVLChatConfig
import numpy as np
from torch.nn import CrossEntropyLoss
import torch.distributed as dist
import random
from internvl.conversation import get_conv_template
from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,
                          LlamaTokenizer, Qwen2ForCausalLM)
from transformers import AutoTokenizer
from mmdet.models.task_modules.samplers import SamplingResult
from mmdet.structures.bbox import bbox2roi, get_box_tensor, scale_boxes
from mmdet.models.layers import multiclass_nms

from internvl.train.constants import (BOX_END_TOKEN, BOX_START_TOKEN,
                                      IMG_CONTEXT_TOKEN, IMG_END_TOKEN,
                                      IMG_START_TOKEN, QUAD_END_TOKEN,
                                      QUAD_START_TOKEN, REF_END_TOKEN,
                                      REF_START_TOKEN)
from torch.nn import functional as F
from peft import LoraConfig, get_peft_model
from mmdet.models.losses import accuracy
from mmdet.structures import DetDataSample

@MODELS.register_module()
class EagleVisionProROIHead(StandardRoIHead):  
    
    def loss(self, x: Tuple[Tensor], rpn_results_list: InstanceList,
             batch_data_samples: List[DetDataSample]) -> dict:
        """Perform forward propagation and loss calculation of the detection
        roi on the features of the upstream network.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: A dictionary of loss components
        """
        assert len(rpn_results_list) == len(batch_data_samples)
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, _ = outputs

        # assign gts and sample proposals
        num_imgs = len(batch_data_samples)
        sampling_results = []
        for i in range(num_imgs):
            # rename rpn_results.bboxes to rpn_results.priors
            rpn_results = rpn_results_list[i]
            rpn_results.priors = rpn_results.pop('bboxes')

            assign_result = self.bbox_assigner.assign(
                rpn_results, batch_gt_instances[i],
                batch_gt_instances_ignore[i])
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                rpn_results,
                batch_gt_instances[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])
            captions_inds = sampling_result.pos_assigned_gt_inds.cpu()
            sampling_result.bbox_captions = batch_gt_instances[i].captions[captions_inds]
            sampling_result.bbox_captions_ignore = batch_gt_instances[i].gt_caption_ignore_flags[captions_inds]
            sampling_results.append(sampling_result)

        losses = dict()
        # bbox head loss
        if self.with_bbox:
            bbox_results = self.bbox_loss(x, sampling_results)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self.mask_loss(x, sampling_results,
                                          bbox_results['bbox_feats'],
                                          batch_gt_instances)
            losses.update(mask_results['loss_mask'])

        return losses
  
    def bbox_loss(self, x: Tuple[Tensor],
                  sampling_results: List[SamplingResult]) -> dict:
        """Perform forward propagation and loss calculation of the bbox head on
        the features of the upstream network.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            sampling_results (list["obj:`SamplingResult`]): Sampling results.

        Returns:
            dict[str, Tensor]: Usually returns a dictionary with keys:

                - `cls_score` (Tensor): Classification scores.
                - `bbox_pred` (Tensor): Box energies / deltas.
                - `bbox_feats` (Tensor): Extract bbox RoI features.
                - `loss_bbox` (dict): A dictionary of bbox loss components.
        """
        rois = bbox2roi([res.priors for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_loss_and_target = self.bbox_head.loss_and_target(
            cls_score=bbox_results['cls_score'],
            bbox_pred=bbox_results['bbox_pred'],
            bbox_feats=bbox_results['bbox_feats'],
            rois=rois,
            sampling_results=sampling_results,
            rcnn_train_cfg=self.train_cfg)

        bbox_results.update(loss_bbox=bbox_loss_and_target['loss_bbox'])
        return bbox_results

    
    def predict_bbox(self,
                     x: Tuple[Tensor],
                     batch_img_metas: List[dict],
                     rpn_results_list: InstanceList,
                     rcnn_test_cfg: ConfigType,
                     rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the bbox head and predict detection
        results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            batch_img_metas (list[dict]): List of image information.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        proposals = [res.bboxes for res in rpn_results_list]
        rois = bbox2roi(proposals)

        if rois.shape[0] == 0:
            return empty_instances(
                batch_img_metas,
                rois.device,
                task_type='bbox',
                box_type=self.bbox_head.predict_box_type,
                num_classes=self.bbox_head.num_classes,
                score_per_cls=rcnn_test_cfg is None)

        bbox_results = self._bbox_forward(x, rois)

        # split batch bbox prediction back to each image
        cls_scores = bbox_results['cls_score']
        bbox_preds = bbox_results['bbox_pred']
        bbox_feats = bbox_results['bbox_feats']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_scores = cls_scores.split(num_proposals_per_img, 0)

        if bbox_feats is not None:
            bbox_feats = bbox_feats.split(num_proposals_per_img, 0)
            
        # some detector with_reg is False, bbox_preds will be None
        if bbox_preds is not None:
            # TODO move this to a sabl_roi_head
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_preds, torch.Tensor):
                bbox_preds = bbox_preds.split(num_proposals_per_img, 0)
            else:
                bbox_preds = self.bbox_head.bbox_pred_split(
                    bbox_preds, num_proposals_per_img)
        else:
            bbox_preds = (None, ) * len(proposals)

        result_list = self.bbox_head.predict_by_feat(
            rois=rois,
            cls_scores=cls_scores,
            bbox_preds=bbox_preds,
            bbox_feats=bbox_feats,
            batch_img_metas=batch_img_metas,
            rcnn_test_cfg=rcnn_test_cfg,
            rescale=rescale)
        return result_list

@MODELS.register_module()
class EagleVisionProHead(ConvFCBBoxHead):
    """EagleVisionHead with separated BN layers and shared conv layers.

    Args:
        pretrained (str): Path to the pretrained LLM. Defaults to None.
        drop_path_rate (float): Drop path rate. Defaults to 0.1.
        vision_select_layer (int): Layer to select features from.
            Defaults to -1.
        dynamic_image_size (bool): Whether to use dynamic image size.
            Defaults to True.
        use_thumbnail (bool): Whether to use thumbnail.
            Defaults to True.
        min_dynamic_patch (int): Minimum number of dynamic patches.
            Defaults to 1.
        max_dynamic_patch (int): Maximum number of dynamic patches.
            Defaults to 6.
        template (str): Template to use for prompt.
            Defaults to "internvl2_5".
        max_seq_length (int): Maximum sequence length.
            Defaults to 4096.
        patch_size (int): Patch embedding size of objects. 
            Defaults to 3.
        caption_num (int): Number of sampling captions. 
            Defaults to 16.
        caption_type (str): Type of vision embedding for caption. 
            Defaults to "global".
        caption_dim (int): Dimension of vision embedding for caption. 
            Defaults to 64.
        loss_caption_weight (float): Weight of caption loss. 
            Defaults to 1.0.
        loss_attr_weight (float): Weight of attribute loss. 
            Defaults to 1.0.
    """
    def __init__(self, 
                 fc_out_channels: int = 1024, 
                 pretrained="OpenGVLab/InternVL2-2B",
                 drop_path_rate = 0.1,
                 vision_select_layer = -1,
                 dynamic_image_size = True,
                 use_thumbnail = True,
                 ps_version = "v2",
                 min_dynamic_patch = 1,
                 max_dynamic_patch = 6,
                 template = "internvl2_5",
                 max_seq_length = 4096,
                 patch_size = 3,
                 caption_num = 16,
                 caption_type = "global",
                 caption_dim = 64,
                 loss_caption_weight=1.0,
                 loss_attr_weight=1.0,
                 *args, 
                 **kwargs) -> None:
        
        super().__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
        
        # load config
        config = InternVLChatConfig.from_pretrained(pretrained)
        config.vision_config.drop_path_rate = drop_path_rate
        if config.llm_config.model_type == 'internlm2':
            config.llm_config.attn_implementation = 'flash_attention_2'  # for InternLM
        else:
            config.llm_config._attn_implementation = 'flash_attention_2'  # for LLaMA
        config.template = template
        config.select_layer = vision_select_layer
        config.dynamic_image_size = dynamic_image_size
        config.use_thumbnail = use_thumbnail
        config.ps_version = ps_version
        config.min_dynamic_patch = min_dynamic_patch
        config.max_dynamic_patch = max_dynamic_patch
        
        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained, add_eos_token=False, trust_remote_code=True, use_fast=False)
        self.tokenizer.tokenizer_path = pretrained
        self.tokenizer.model_max_length = max_seq_length
        self.tokenizer.padding_side = 'left'
        
        
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        
        # load language model
        self.language_model = InternVLChatModel.from_pretrained(pretrained, torch_dtype=torch.bfloat16, config=config).language_model
        self.language_model.img_context_token_id = self.img_context_token_id
        self.language_model.config.use_cache = False
        
        self.return_dict = True
        self.loss_reduction_all_gather = False
        

        # load multimodal up layer
        self.multimodal_up_layer = nn.Linear(self.in_channels, config.hidden_size, bias=True)
        
        self.language_model.eval()
        for param in self.language_model.parameters():
            param.requires_grad = False
        
        # load question
        self.question = '''This image includes a remote sensing object in a bird's-eye view. Please help me to explain the visual content of this object in a fine-grained manner.\n<image>'''
        self.template = config.template
        self.conv_template = get_conv_template(self.template)
        self.system_message = self.conv_template.system_message
        self.patch_size = patch_size
        
        # load generation config
        self.generation_config = dict(
            num_beams=self.language_model.generation_config.num_beams,
            max_new_tokens=1024,
            min_new_tokens=self.language_model.generation_config.min_new_tokens,
            do_sample=False,
            temperature=1.0,
        )
        self.caption_num = caption_num
        self.caption_type = caption_type
        self.caption_dim = caption_dim
        self.loss_caption_weight = loss_caption_weight
        self.loss_attr_weight = loss_attr_weight
        if self.caption_type != "global":
            self.base_emb = nn.Parameter(torch.zeros(self.caption_dim, config.hidden_size), requires_grad=True)
            if self.caption_type == "local":
                nn.init.xavier_normal_(self.base_emb)
            elif self.caption_type == "orthogonal":
                nn.init.orthogonal_(self.base_emb)
            self.inference_q = nn.Linear(config.hidden_size, self.caption_dim)


    def loss_and_target(self,
                        cls_score: Tensor,
                        bbox_pred: Tensor,
                        bbox_feats: Tensor,
                        rois: Tensor,
                        sampling_results: List[SamplingResult],
                        rcnn_train_cfg: ConfigDict,
                        concat: bool = True,
                        reduction_override: Optional[str] = None) -> dict:
        """Calculate the loss based on the features extracted by the bbox head.

        Args:
            cls_score (Tensor): Classification prediction
                results of all class, has shape
                (batch_size * num_proposals_single_image, num_classes)
            bbox_pred (Tensor): Regression prediction results,
                has shape
                (batch_size * num_proposals_single_image, 4), the last
                dimension 4 represents [tl_x, tl_y, br_x, br_y].
            bbox_feats (Tensor): Bbox features with shape
                (batch_size * num_proposals_single_image, hidden_size)
            rois (Tensor): RoIs with the shape
                (batch_size * num_proposals_single_image, 5) where the first
                column indicates batch id of each RoI.
            sampling_results (List[obj:SamplingResult]): Assign results of
                all images in a batch after sampling.
            rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.
            concat (bool): Whether to concatenate the results of all
                the images in a single batch. Defaults to True.
            reduction_override (str, optional): The reduction
                method used to override the original reduction
                method of the loss. Options are "none",
                "mean" and "sum". Defaults to None,

        Returns:
            dict: A dictionary of loss and targets components.
                The targets are only used for cascade rcnn.
        """

        cls_reg_targets = self.get_targets(
            sampling_results, rcnn_train_cfg, concat=concat)
        losses = self.loss(
            cls_score,
            bbox_pred,
            bbox_feats,
            rois,
            *cls_reg_targets,
            reduction_override=reduction_override)

        # cls_reg_targets is only for cascade rcnn
        return dict(loss_bbox=losses, bbox_targets=cls_reg_targets)
    

    def _get_targets_single(self, pos_priors: Tensor, neg_priors: Tensor,
                            pos_gt_bboxes: Tensor, pos_gt_labels: Tensor,
                            pos_gt_captions,
                            pos_gt_captions_ignore,
                            cfg: ConfigDict) -> tuple:
        """Calculate the ground truth for proposals in the single image
        according to the sampling results.

        Args:
            pos_priors (Tensor): Contains all the positive boxes,
                has shape (num_pos, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            neg_priors (Tensor): Contains all the negative boxes,
                has shape (num_neg, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            pos_gt_bboxes (Tensor): Contains gt_boxes for
                all positive samples, has shape (num_pos, 4),
                the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            pos_gt_labels (Tensor): Contains gt_labels for
                all positive samples, has shape (num_pos, ).
            cfg (obj:`ConfigDict`): `train_cfg` of R-CNN.

        Returns:
            Tuple[Tensor]: Ground truth for proposals
            in a single image. Containing the following Tensors:

                - labels(Tensor): Gt_labels for all proposals, has
                  shape (num_proposals,).
                - label_weights(Tensor): Labels_weights for all
                  proposals, has shape (num_proposals,).
                - bbox_targets(Tensor):Regression target for all
                  proposals, has shape (num_proposals, 4), the
                  last dimension 4 represents [tl_x, tl_y, br_x, br_y].
                - bbox_weights(Tensor):Regression weights for all
                  proposals, has shape (num_proposals, 4).
        """
        num_pos = pos_priors.size(0)
        num_neg = neg_priors.size(0)
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_priors.new_full((num_samples, ),
                                     self.num_classes,
                                     dtype=torch.long)
        reg_dim = pos_gt_bboxes.size(-1) if self.reg_decoded_bbox \
            else self.bbox_coder.encode_size
        label_weights = pos_priors.new_zeros(num_samples)
        bbox_targets = pos_priors.new_zeros(num_samples, reg_dim)
        bbox_weights = pos_priors.new_zeros(num_samples, reg_dim)
        bbox_captions = np.empty((num_samples, ), dtype=object)
        bbox_captions_ignore = np.zeros(num_samples, dtype=bool)
        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[:num_pos] = pos_weight
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    pos_priors, pos_gt_bboxes)
            else:
                # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
                # is applied directly on the decoded bounding boxes, both
                # the predicted boxes and regression targets should be with
                # absolute coordinate format.
                pos_bbox_targets = get_box_tensor(pos_gt_bboxes)
            bbox_targets[:num_pos, :] = pos_bbox_targets
            bbox_weights[:num_pos, :] = 1
            bbox_captions[:num_pos] = pos_gt_captions
            bbox_captions_ignore[:num_pos] = pos_gt_captions_ignore
        if num_neg > 0:
            label_weights[-num_neg:] = 1.0

        return labels, label_weights, bbox_targets, bbox_weights, bbox_captions, bbox_captions_ignore
    

    def get_targets(self,
                    sampling_results: List[SamplingResult],
                    rcnn_train_cfg: ConfigDict,
                    concat: bool = True) -> tuple:
        """Calculate the ground truth for all samples in a batch according to
        the sampling_results.

        Almost the same as the implementation in bbox_head, we passed
        additional parameters pos_inds_list and neg_inds_list to
        `_get_targets_single` function.

        Args:
            sampling_results (List[obj:SamplingResult]): Assign results of
                all images in a batch after sampling.
            rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.
            concat (bool): Whether to concatenate the results of all
                the images in a single batch.

        Returns:
            Tuple[Tensor]: Ground truth for proposals in a single image.
            Containing the following list of Tensors:

            - labels (list[Tensor],Tensor): Gt_labels for all
                proposals in a batch, each tensor in list has
                shape (num_proposals,) when `concat=False`, otherwise
                just a single tensor has shape (num_all_proposals,).
            - label_weights (list[Tensor]): Labels_weights for
                all proposals in a batch, each tensor in list has
                shape (num_proposals,) when `concat=False`, otherwise
                just a single tensor has shape (num_all_proposals,).
            - bbox_targets (list[Tensor],Tensor): Regression target
                for all proposals in a batch, each tensor in list
                has shape (num_proposals, 4) when `concat=False`,
                otherwise just a single tensor has shape
                (num_all_proposals, 4), the last dimension 4 represents
                [tl_x, tl_y, br_x, br_y].
            - bbox_weights (list[tensor],Tensor): Regression weights for
                all proposals in a batch, each tensor in list has shape
                (num_proposals, 4) when `concat=False`, otherwise just a
                single tensor has shape (num_all_proposals, 4).
        """
        pos_priors_list = [res.pos_priors for res in sampling_results]
        neg_priors_list = [res.neg_priors for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        pos_gt_captions_list = [res.bbox_captions for res in sampling_results]
        pos_gt_captions_ignore_list = [res.bbox_captions_ignore for res in sampling_results]
        labels, label_weights, bbox_targets, bbox_weights, bbox_captions, bbox_captions_ignore = multi_apply(
            self._get_targets_single,
            pos_priors_list,
            neg_priors_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            pos_gt_captions_list,
            pos_gt_captions_ignore_list,
            cfg=rcnn_train_cfg)

        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
            bbox_captions = np.concatenate(bbox_captions)
            bbox_captions_ignore = np.concatenate(bbox_captions_ignore)
        return labels, label_weights, bbox_targets, bbox_weights, bbox_captions, bbox_captions_ignore

    def loss(self,
             cls_score: Tensor,
             bbox_pred: Tensor,
             bbox_feats: Tensor,
             rois: Tensor,
             labels: Tensor,
             label_weights: Tensor,
             bbox_targets: Tensor,
             bbox_weights: Tensor,
             bbox_captions: Tensor,
             bbox_captions_ignore: Tensor,
             reduction_override: Optional[str] = None) -> dict:
        """Calculate the loss based on the network predictions and targets.

        Args:
            cls_score (Tensor): Classification prediction
                results of all class, has shape
                (batch_size * num_proposals_single_image, num_classes)
            bbox_pred (Tensor): Regression prediction results,
                has shape
                (batch_size * num_proposals_single_image, 4), the last
                dimension 4 represents [tl_x, tl_y, br_x, br_y].
            rois (Tensor): RoIs with the shape
                (batch_size * num_proposals_single_image, 5) where the first
                column indicates batch id of each RoI.
            labels (Tensor): Gt_labels for all proposals in a batch, has
                shape (batch_size * num_proposals_single_image, ).
            label_weights (Tensor): Labels_weights for all proposals in a
                batch, has shape (batch_size * num_proposals_single_image, ).
            bbox_targets (Tensor): Regression target for all proposals in a
                batch, has shape (batch_size * num_proposals_single_image, 4),
                the last dimension 4 represents [tl_x, tl_y, br_x, br_y].
            bbox_weights (Tensor): Regression weights for all proposals in a
                batch, has shape (batch_size * num_proposals_single_image, 4).
            reduction_override (str, optional): The reduction
                method used to override the original reduction
                method of the loss. Options are "none",
                "mean" and "sum". Defaults to None,

        Returns:
            dict: A dictionary of loss.
        """

        losses = dict()

        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                loss_cls_ = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_
                if self.custom_activation:
                    acc_ = self.loss_cls.get_accuracy(cls_score, labels)
                    losses.update(acc_)
                else:
                    losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                    bbox_pred = get_box_tensor(bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), self.num_classes,
                        -1)[pos_inds.type(torch.bool),
                            labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
                
        if bbox_feats is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                pos_captions = bbox_captions[pos_inds.type(torch.bool).cpu()]
                pos_captions_ignore = bbox_captions_ignore[pos_inds.type(torch.bool).cpu()]
                pos_captions = pos_captions[~pos_captions_ignore]
                pos_feats = bbox_feats[pos_inds].flatten(2).permute(0, 2, 1)
                # if isinstance(pos_captions, dict):
                #     pos_captions = [pos_captions]
                if len(pos_captions) > 0:
                    loss_caption, loss_ortho, loss_attr = self.loss_caption(
                        pos_captions,
                        pos_feats)
                else:
                    loss_caption = bbox_pred[:0].sum()
                    loss_ortho = None
                    loss_attr = None
                if loss_ortho is None:
                    loss_ortho = bbox_pred[:0].sum()
                if loss_attr is None:
                    loss_attr = bbox_pred[:0].sum()
            else:
                loss_caption = bbox_pred[pos_inds].sum()
                loss_ortho = bbox_pred[pos_inds].sum()
                loss_attr = bbox_pred[pos_inds].sum()
            if self.loss_caption_weight is not None:
                loss_caption = loss_caption * self.loss_caption_weight
            if self.loss_attr_weight is not None:
                loss_attr = loss_attr * self.loss_attr_weight
            losses["loss_caption"] = loss_caption
            losses["loss_ortho"] = loss_ortho
            losses["loss_attr"] = loss_attr
        return losses
 
    def loss_caption(self, pos_captions, pos_cls_feats):
        # sample caption
        sample_id = [cp_id for cp_id, _ in enumerate(pos_captions)]
        sample_id = random.sample(sample_id, min(len(sample_id), self.caption_num))
        pos_captions = pos_captions[sample_id]
        pos_cls_feats = pos_cls_feats[sample_id]
        
        multimodal_input_feats = self.multimodal_up_layer(pos_cls_feats)
        multimodal_input_feats = multimodal_input_feats.bfloat16()
        
        loss_ortho = None
        if self.caption_type == "local":
            multimodal_input_feats, proj = self.decompose(multimodal_input_feats, self.base_emb.unsqueeze(0))
        elif self.caption_type == "orthogonal":
            multimodal_input_feats, proj = self.decompose(multimodal_input_feats, self.base_emb.unsqueeze(0))
            
            base_emb = F.normalize(self.base_emb, p=2, dim=-1).squeeze(0)
            proto_sim = torch.matmul(base_emb, base_emb.t())
            eye_sim = torch.triu(torch.ones_like(proto_sim), diagonal=1)
            loss_ortho = torch.abs(proto_sim[eye_sim == 1]).mean()
            
        
        input_ids = torch.stack([caption["input_ids"] for caption in pos_captions])
        labels = torch.stack([caption["labels"] for caption in pos_captions])
        attention_mask = torch.stack([caption["attention_mask"] for caption in pos_captions])
        position_ids = torch.stack([caption["position_ids"] for caption in pos_captions])
        attr_indexs = [caption["disentangle_range"][0] for caption in pos_captions]
        attr_tokens = [caption["disentangle_range"][1] for caption in pos_captions]
        
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()
    
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + multimodal_input_feats.reshape(-1, C)
            ignore_flag = False
        except Exception as e:
            multimodal_input_feats = multimodal_input_feats.reshape(-1, C)
            print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                  f'multimodal_input_feats.shape={multimodal_input_feats.shape}')
            n_token = selected.sum()
            input_embeds[selected] = input_embeds[selected] * 0.0 + multimodal_input_feats[:n_token]
            ignore_flag = True

        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=True,
        )
        logits = outputs.logits
        
        loss_l2 = torch.nn.MSELoss(reduction='mean')
        loss_attr = None
        loss_caption = None
        loss_weight = None

        if self.caption_type != 'global':
            loss_attr = []
            for i in range(self.caption_dim):
                q_preds = []
                q_gts = []
                for attr_index, attr_token, q_gt in zip(attr_indexs, attr_tokens, proj):
                    if i in attr_index:
                        attr_ids = attr_token[attr_index.index(i)]
                        attr_ids = torch.tensor(attr_ids).to(logits.device)
                        attr_embeds = self.language_model.get_input_embeddings()(attr_ids).clone()
                        q_pred = self.inference_q(attr_embeds.float().mean(0))

                        q_preds.append(q_pred[i])
                        q_gts.append(q_gt.float().mean(0)[i])

                if len(q_preds) > 0:
                    q_preds = torch.stack(q_preds)
                    q_gts = torch.stack(q_gts)

                    loss_attr.append(loss_l2(q_gts, q_preds))

            if len(loss_attr) > 0:    
                loss_attr = torch.stack(loss_attr).mean()
            else:
                loss_attr = None

        if labels is not None and loss_weight is not None:
            loss_weight = torch.tensor(loss_weight, dtype=torch.float32, device=labels.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_weights = loss_weight[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction='none')
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_weights = shift_weights.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            shift_weights = shift_weights.to(shift_logits.device)
            loss_caption = loss_fct(shift_logits, shift_labels)

            shift_weights_sum = shift_weights.sum()
            if self.loss_reduction_all_gather:
                dist.all_reduce(shift_weights_sum, op=dist.ReduceOp.AVG)

            loss_caption = loss_caption * shift_weights
            loss_caption = loss_caption.sum() / shift_weights_sum
            if ignore_flag:
                loss_caption = loss_caption * 0.0
        elif labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss_caption = loss_fct(shift_logits, shift_labels)
            if ignore_flag:
                loss_caption = loss_caption * 0.0

        
        return loss_caption, loss_ortho, loss_attr
   
    
    def decompose(self, feats, bases):
        """
        Args:
            feats (Tensor): [BxNxC]
            bases (Tensor): [1xKxC]
        Returns:
            Tensor: [BxNxC]
            Tensor: [BxNxC]
        """
        B, N, C = feats.shape
        s1 = F.normalize(bases.bfloat16(), p=2, dim=-1)
        proj = torch.matmul(s1, feats.flatten(0, 1).unsqueeze(-1))
        out_feats = proj.unsqueeze(2) * s1.unsqueeze(-1)
        out_feats = out_feats.reshape(B, N, self.caption_dim, -1).sum(1)
        
        return out_feats, proj.reshape(B, N, -1)
    
    

    def predict_by_feat(self,
                        rois: Tuple[Tensor],
                        cls_scores: Tuple[Tensor],
                        bbox_preds: Tuple[Tensor],
                        bbox_feats: Tuple[Tensor],
                        batch_img_metas: List[dict],
                        rcnn_test_cfg: Optional[ConfigDict] = None,
                        rescale: bool = False) -> InstanceList:
        """Transform a batch of output features extracted from the head into
        bbox results.

        Args:
            rois (tuple[Tensor]): Tuple of boxes to be transformed.
                Each has shape  (num_boxes, 5). last dimension 5 arrange as
                (batch_index, x1, y1, x2, y2).
            cls_scores (tuple[Tensor]): Tuple of box scores, each has shape
                (num_boxes, num_classes + 1).
            bbox_preds (tuple[Tensor]): Tuple of box energies / deltas, each
                has shape (num_boxes, num_classes * 4).
            bbox_feats (tuple[Tensor]): Tuple of box features, each has shape
                (num_boxes, num_classes, caption_dim).
            batch_img_metas (list[dict]): List of image information.
            rcnn_test_cfg (obj:`ConfigDict`, optional): `test_cfg` of R-CNN.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Instance segmentation
            results of each image after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        assert len(cls_scores) == len(bbox_preds)
        result_list = []
        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            results = self._predict_by_feat_single(
                roi=rois[img_id],
                cls_score=cls_scores[img_id],
                bbox_pred=bbox_preds[img_id],
                bbox_feats=bbox_feats[img_id],
                img_meta=img_meta,
                rescale=rescale,
                rcnn_test_cfg=rcnn_test_cfg)
            result_list.append(results)

        return result_list

    def _predict_by_feat_single(
            self,
            roi: Tensor,
            cls_score: Tensor,
            bbox_pred: Tensor,
            bbox_feats: Tensor,
            img_meta: dict,
            bs: int = 32,
            rescale: bool = False,
            rcnn_test_cfg: Optional[ConfigDict] = None) -> InstanceData:
        """Transform a single image's features extracted from the head into
        bbox results.

        Args:
            roi (Tensor): Boxes to be transformed. Has shape (num_boxes, 5).
                last dimension 5 arrange as (batch_index, x1, y1, x2, y2).
            cls_score (Tensor): Box scores, has shape
                (num_boxes, num_classes + 1).
            bbox_pred (Tensor): Box energies / deltas.
                has shape (num_boxes, num_classes * 4). 
            bbox_feats (Tensor): Box features, has shape
                (num_boxes, num_classes, caption_dim).
            img_meta (dict): image information.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of Bbox Head.
                Defaults to None

        Returns:
            :obj:`InstanceData`: Detection results of each image\
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        results = InstanceData()
        if roi.shape[0] == 0:
            return empty_instances([img_meta],
                roi.device,
                task_type='bbox',
                                   instance_results=[results],
                box_type=self.predict_box_type,
                                   use_box_type=False,
                num_classes=self.num_classes,
                score_per_cls=rcnn_test_cfg is None)[0]

        # some loss (Seesaw loss..) may have custom activation
        if self.custom_cls_channels:
            scores = self.loss_cls.get_activation(cls_score)
        else:
            scores = F.softmax(
                cls_score, dim=-1) if cls_score is not None else None

        img_shape = img_meta['img_shape']
        num_rois = roi.size(0)
        # bbox_pred would be None in some detector when with_reg is False,
        # e.g. Grid R-CNN.
        if bbox_pred is not None:
            num_classes = 1 if self.reg_class_agnostic else self.num_classes
            roi = roi.repeat_interleave(num_classes, dim=0)
            bbox_pred = bbox_pred.view(-1, self.bbox_coder.encode_size)
            bboxes = self.bbox_coder.decode(
                roi[..., 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = roi[:, 1:].clone()
            if img_shape is not None and bboxes.size(-1) == 4:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and bboxes.size(0) > 0:
            assert img_meta.get('scale_factor') is not None
            scale_factor = [1 / s for s in img_meta['scale_factor']]
            bboxes = scale_boxes(bboxes, scale_factor)

        # Get the inside tensor when `bboxes` is a box type
        bboxes = get_box_tensor(bboxes)
        box_dim = bboxes.size(-1)
        bboxes = bboxes.view(num_rois, -1)

        if rcnn_test_cfg is None:
            # This means that it is aug test.
            # It needs to return the raw results without nms.
            results.bboxes = bboxes
            results.scores = scores
            results.feats = bbox_feats
        else:
            det_bboxes, det_labels, det_inds = multiclass_nms(
                bboxes,
                scores,
                rcnn_test_cfg.score_thr,
                rcnn_test_cfg.nms,
                rcnn_test_cfg.max_per_img,
                box_dim=box_dim,
                return_inds=True)
            B, C, H, W = bbox_feats.shape
            
            results.bboxes = det_bboxes[:, :-1]
            results.scores = det_bboxes[:, -1]
            results.labels = det_labels
            bbox_feats = bbox_feats[:, None].expand(B, scores.size(1) - 1,  C, H, W).flatten(0, 1)
            results.feats = bbox_feats[det_inds].flatten(2).permute(0, 2, 1)
        
        # caption pred
        if 'caption' in rcnn_test_cfg and rcnn_test_cfg.caption == True:
            post_feats = self.multimodal_up_layer(results.feats).bfloat16()
            if len(post_feats) > 0:
                if self.caption_type != "global":
                    combine_post_feats = []
                    for post_feat in post_feats.split(bs):
                        post_feat, _ = self.decompose(post_feat, self.base_emb.unsqueeze(0))
                        combine_post_feats.append(post_feat)
                        torch.cuda.empty_cache()
                    post_feats = torch.cat(combine_post_feats)
                results.captions = self.chat(self.tokenizer, post_feats, self.generation_config)
            else:
                if self.caption_type != "global":
                    post_feats = post_feats.mean(1).unsqueeze(1).repeat(1, self.caption_dim, 1)
                results.captions = []  
            torch.cuda.empty_cache()
        return results


    def chat(self, tokenizer, cls_feats, generation_config, history=None, return_history=False,
             num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
             verbose=False):

        if history is None and cls_feats is not None and '<image>' not in self.question:
            self.question = '<image>\n' + self.question

        if num_patches_list is None:
            num_patches_list = [cls_feats.shape[0]] if cls_feats is not None else []
        assert cls_feats is None or len(cls_feats) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], self.question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and cls_feats is not None:
            image_bs = cls_feats.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        queries = []
        for _ in range(sum(num_patches_list)):
            if self.caption_type == "global":
                num_patch_token = self.patch_size ** 2
            else:
                num_patch_token = self.caption_dim
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * num_patch_token + IMG_END_TOKEN
            queries.append(query.replace('<image>', image_tokens, 1))

        model_inputs = tokenizer(queries, return_tensors='pt')
        input_ids = model_inputs['input_ids'].cuda()
        attention_mask = model_inputs['attention_mask'].cuda()
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_features=cls_feats,
            **generation_config
        )
        response = []
        for sub_generation_output in generation_output:
            response.extend(tokenizer.batch_decode(sub_generation_output, skip_special_tokens=True))

        return response

    @torch.no_grad()
    def transfer_embeds(self, input_ids, vit_embeds):
        """
        Transfer the image embeddings to the language model.

        Args:
            input_ids: (B, N) input ids
            vit_embeds: (B, N, C) vit embeddings
        """

        input_embeds = self.language_model.get_input_embeddings()(input_ids)
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        assert selected.sum() != 0
        input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device).bfloat16()

        input_embeds = input_embeds.reshape(B, N, C)
        return input_embeds
    
    @torch.no_grad()
    def generate(
            self,
            bs: Optional[int] = 8,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.img_context_token_id is not None
        if pixel_values is not None:
            vit_embeds = self.extract_feature(pixel_values)
            input_embeds = self.transfer_embeds(input_ids, vit_embeds)
        elif visual_features is not None:
            vit_embeds = visual_features
            input_embeds = self.transfer_embeds(input_ids, vit_embeds)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
        
        outputs = []
        for sub_input_embeds, sub_attention_mask in zip(input_embeds.split(bs), attention_mask.split(bs)):
            sub = self.language_model.generate(
                inputs_embeds=sub_input_embeds,
                attention_mask=sub_attention_mask,
                generation_config=generation_config,
                output_hidden_states=output_hidden_states,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
                **generate_kwargs,
            )
            outputs.append(sub)
            torch.cuda.empty_cache()
            
        return outputs


