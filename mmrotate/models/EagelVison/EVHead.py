import copy
from typing import List, Optional, Tuple

import torch
from mmdet.models import inverse_sigmoid
from mmdet.models.task_modules import anchor_inside_flags
from mmdet.models.utils import (filter_scores_and_topk, multi_apply,
                                select_single_mlvl, sigmoid_geometric_mean,
                                unmap, images_to_levels)
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, cat_boxes, distance2bbox
from mmdet.utils import (ConfigType, InstanceList, OptConfigType,
                         OptInstanceList, reduce_mean)
from mmengine import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor, nn

from mmrotate.registry import MODELS, TASK_UTILS
from mmrotate.structures import RotatedBoxes, distance2obb
from mmrotate.models.dense_heads import RotatedRTMDetSepBNHead
from internvl.model.internvl_chat import InternVLChatModel, InternVLChatConfig
import numpy as np
from torch.nn import CrossEntropyLoss
import torch.distributed as dist
import random
from internvl.conversation import get_conv_template
from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,
                          LlamaTokenizer, Qwen2ForCausalLM)
from transformers import AutoTokenizer

from internvl.train.constants import (BOX_END_TOKEN, BOX_START_TOKEN,
                                      IMG_CONTEXT_TOKEN, IMG_END_TOKEN,
                                      IMG_START_TOKEN, QUAD_END_TOKEN,
                                      QUAD_START_TOKEN, REF_END_TOKEN,
                                      REF_START_TOKEN)
from torch.nn import functional as F


@MODELS.register_module()
class EagleVisionHead(RotatedRTMDetSepBNHead):
    """EagleVisionHead with object-level attribute description module.   

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        share_conv (bool): Whether to share conv layers between stages.
            Defaults to True.
        scale_angle (bool): Does not support in RotatedRTMDetSepBNHead,
            Defaults to False.
        norm_cfg (:obj:`ConfigDict` or dict)): Config dict for normalization
            layer. Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (:obj:`ConfigDict` or dict)): Config dict for activation layer.
            Defaults to dict(type='SiLU').
        pred_kernel_size (int): Kernel size of prediction layer. Defaults to 1.
        exp_on_reg (bool): Whether to apply exponential on bbox_pred.
            Defaults to False.
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
                 num_classes: int,
                 in_channels: int,
                 share_conv: bool = True,
                 scale_angle: bool = False,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU'),
                 pred_kernel_size: int = 1,
                 exp_on_reg: bool = False,
                 pretrained: str = "/mnt/data1/jianghx/ckpts/InternVL2-1B",
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
                 **kwargs) -> None:
        super().__init__(
            num_classes,
            in_channels,
            share_conv=share_conv,
            scale_angle=scale_angle,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            pred_kernel_size=pred_kernel_size,
            exp_on_reg=exp_on_reg,
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
        
        # load img_context_token_id
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
        
        # load loss config
        self.loss_caption_weight = loss_caption_weight
        self.loss_attr_weight = loss_attr_weight
        self.caption_num = caption_num
        self.caption_type = caption_type
        self.caption_dim = caption_dim
        if self.caption_type != "global":
            self.base_emb = nn.Parameter(torch.zeros(self.caption_dim, config.hidden_size), requires_grad=True)
            if self.caption_type == "local":
                nn.init.xavier_normal_(self.base_emb)
            elif self.caption_type == "orthogonal":
                nn.init.orthogonal_(self.base_emb)
            self.inference_q = nn.Linear(config.hidden_size, self.caption_dim)
          

    def forward(self, feats: Tuple[Tensor, ...]) -> tuple:
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction
            - cls_scores (list[Tensor]): Classification scores for all scale
              levels, each is a 4D-tensor, the channels number is
              num_base_priors * num_classes.
            - bbox_preds (list[Tensor]): Box energies / deltas for all scale
              levels, each is a 4D-tensor, the channels number is
              num_base_priors * 4.
            - angle_preds (list[Tensor]): Angle prediction for all scale
              levels, each is a 4D-tensor, the channels number is
              num_base_priors * angle_dim.
        """
        cls_scores = []
        bbox_preds = []
        angle_preds = []
        cls_feats = []
        for idx, (x, stride) in enumerate(
                zip(feats, self.prior_generator.strides)):
            cls_feat = x
            reg_feat = x

            for cls_layer in self.cls_convs[idx]:
                cls_feat = cls_layer(cls_feat)
            cls_feats.append(cls_feat)
            cls_score = self.rtm_cls[idx](cls_feat)

            for reg_layer in self.reg_convs[idx]:
                reg_feat = reg_layer(reg_feat)

            if self.with_objectness:
                objectness = self.rtm_obj[idx](reg_feat)
                cls_score = inverse_sigmoid(
                    sigmoid_geometric_mean(cls_score, objectness))
            if self.exp_on_reg:
                reg_dist = self.rtm_reg[idx](reg_feat).exp() * stride[0]
            else:
                reg_dist = self.rtm_reg[idx](reg_feat) * stride[0]

            angle_pred = self.rtm_ang[idx](reg_feat)

            cls_scores.append(cls_score)
            bbox_preds.append(reg_dist)
            angle_preds.append(angle_pred)
        return tuple(cls_scores), tuple(bbox_preds), tuple(angle_preds), tuple(cls_feats)
    
    def loss_caption(self, pos_captions, pos_cls_feats):
        # sample caption
        sample_id = [cp_id for cp_id, _ in enumerate(pos_captions)]
        sample_id = random.sample(sample_id, min(len(sample_id), self.caption_num))
        pos_captions = pos_captions[sample_id]
        pos_cls_feats = pos_cls_feats[sample_id]

        
        multimodal_input_feats = self.multimodal_up_layer(pos_cls_feats).reshape(pos_cls_feats.shape[0], self.patch_size ** 2, -1)
        multimodal_input_feats = multimodal_input_feats.bfloat16()
        
        # obtain caption
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
            output_hidden_states=False,
            return_dict=True,
        )
        logits = outputs.logits
        
        loss_l2 = torch.nn.MSELoss(reduction='mean')
        loss_attr = None
        loss_caption = None
        loss_weight = None

        # compute loss for attribute disentangle
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
    
    def loss_by_feat_single(self, cls_score: Tensor, bbox_pred: Tensor, cls_feats: Tensor,
                            angle_pred: Tensor, labels: Tensor, captions: str, captions_ignore: bool,
                            label_weights: Tensor, bbox_targets: Tensor,
                            assign_metrics: Tensor, stride: List[int]):
        """Compute loss of a single scale level.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Decoded bboxes for each scale
                level with shape (N, num_anchors * 5, H, W) for rbox loss
                or (N, num_anchors * 4, H, W) for hbox loss.
            angle_pred (Tensor): Decoded bboxes for each scale
                level with shape (N, num_anchors * angle_dim, H, W).
            cls_feats (Tensor): features for caption with 
                shape (N, num_anchors * num_classes, C).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors).
            bbox_targets (Tensor): BBox regression targets of each anchor with
                shape (N, num_total_anchors, 4).
            assign_metrics (Tensor): Assign metrics with shape
                (N, num_total_anchors).
            stride (List[int]): Downsample stride of the feature map.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert stride[0] == stride[1], 'h stride is not equal to w stride!'
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(
            -1, self.cls_out_channels).contiguous()
        
        if self.use_hbbox_loss:
            bbox_pred = bbox_pred.reshape(-1, 4)
        else:
            bbox_pred = bbox_pred.reshape(-1, 5)
        bbox_targets = bbox_targets.reshape(-1, 5)

        labels = labels.reshape(-1)
        assign_metrics = assign_metrics.reshape(-1)
        label_weights = label_weights.reshape(-1)
        targets = (labels, assign_metrics)
        
        captions_targets = captions.reshape(-1)
        captions_targets_ignore = captions_ignore.reshape(-1)

        loss_cls = self.loss_cls(
            cls_score, targets, label_weights, avg_factor=1.0)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]

            pos_decode_bbox_pred = pos_bbox_pred
            pos_decode_bbox_targets = pos_bbox_targets
            if self.use_hbbox_loss:
                pos_decode_bbox_targets = bbox_cxcywh_to_xyxy(
                    pos_bbox_targets[:, :4])

            # regression loss
            pos_bbox_weight = assign_metrics[pos_inds]

            loss_angle = angle_pred.sum() * 0
            if self.loss_angle is not None:
                angle_pred = angle_pred.reshape(-1,
                                                self.angle_coder.encode_size)
                pos_angle_pred = angle_pred[pos_inds]
                pos_angle_target = pos_bbox_targets[:, 4:5]
                pos_angle_target = self.angle_coder.encode(pos_angle_target)
                if pos_angle_target.dim() == 2:
                    pos_angle_weight = pos_bbox_weight.unsqueeze(-1)
                else:
                    pos_angle_weight = pos_bbox_weight
                loss_angle = self.loss_angle(
                    pos_angle_pred,
                    pos_angle_target,
                    weight=pos_angle_weight,
                    avg_factor=1.0)

            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                weight=pos_bbox_weight,
                avg_factor=1.0)

            pos_cls_feats = self.assign_multi_feat(cls_feats.permute(0, 2, 3, 1), pos_inds)
            pos_inds_flatten = pos_inds.cpu().numpy().flatten()

            pos_captions = captions_targets[pos_inds_flatten]
            pos_captions = pos_captions[~captions_targets_ignore[pos_inds_flatten]]
            if isinstance(pos_captions, dict):
                pos_captions = [pos_captions]
                
            if len(pos_captions) > 0:
                loss_caption, loss_ortho, loss_attr = self.loss_caption(
                    pos_captions, pos_cls_feats 
                )
            else:
                loss_caption = cls_feats.sum() * 0
                loss_ortho = None
                loss_attr = None
            if loss_ortho == None:
                loss_ortho = bbox_targets.new_tensor(0.)
            if loss_attr == None:
                loss_attr = bbox_targets.new_tensor(0.)
                
            if self.loss_caption_weight is not None:
                loss_caption = loss_caption * self.loss_caption_weight
            if self.loss_attr_weight is not None:
                loss_attr = loss_attr * self.loss_attr_weight

        else:
            loss_bbox = bbox_pred.sum() * 0
            pos_bbox_weight = bbox_targets.new_tensor(0.)
            loss_angle = angle_pred.sum() * 0
            loss_caption = cls_feats.sum() * 0
            loss_ortho = bbox_targets.new_tensor(0.)
            loss_attr = bbox_targets.new_tensor(0.)
        
        return (loss_cls, loss_bbox, loss_angle, loss_caption, loss_ortho, loss_attr, assign_metrics.sum(),
                pos_bbox_weight.sum(), pos_bbox_weight.sum())

    def loss_by_feat(self,
                     cls_scores: List[Tensor],
                     bbox_preds: List[Tensor],
                     angle_preds: List[Tensor],
                     cls_feats: List[Tensor],
                     batch_gt_instances: InstanceList,
                     batch_img_metas: List[dict],
                     batch_gt_instances_ignore: OptInstanceList = None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box predict for each scale
                level with shape (N, num_anchors * 4, H, W) in
                [t, b, l, r] format.
            angle_preds (list[Tensor]): Angle predict for each scale
                level with shape (N, num_anchors * angle_dim, H, W).
            cls_feats (list[Tensor]): features for caption with 
                shape (N, num_anchors * num_classes, C).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_imgs = len(batch_img_metas)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, batch_img_metas, device=device)
        flatten_cls_scores = torch.cat([
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  self.cls_out_channels)
            for cls_score in cls_scores
        ], 1)

        decoded_bboxes = []
        decoded_hbboxes = []
        angle_preds_list = []
        for anchor, bbox_pred, angle_pred in zip(anchor_list[0], bbox_preds,
                                                 angle_preds):
            anchor = anchor.reshape(-1, 4)
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            angle_pred = angle_pred.permute(0, 2, 3, 1).reshape(
                num_imgs, -1, self.angle_coder.encode_size)

            if self.use_hbbox_loss:
                hbbox_pred = distance2bbox(anchor, bbox_pred)
                decoded_hbboxes.append(hbbox_pred)

            decoded_angle = self.angle_coder.decode(angle_pred, keepdim=True)
            bbox_pred = torch.cat([bbox_pred, decoded_angle], dim=-1)

            bbox_pred = distance2obb(
                anchor, bbox_pred, angle_version=self.angle_version)
            decoded_bboxes.append(bbox_pred)
            angle_preds_list.append(angle_pred)

        # flatten_bboxes is rbox, for target assign
        flatten_bboxes = torch.cat(decoded_bboxes, 1)

        cls_reg_targets = self.get_targets(
            flatten_cls_scores,
            flatten_bboxes,
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)
        (anchor_list, labels_list, captions_list, captions_ignore_list, label_weights_list, bbox_targets_list,
         assign_metrics_list, sampling_results_list) = cls_reg_targets
        
        if self.use_hbbox_loss:
            decoded_bboxes = decoded_hbboxes
        # compute loss
        (losses_cls, losses_bbox, losses_angle, losses_caption, loss_ortho, loss_attr, cls_avg_factors,
         bbox_avg_factors, angle_avg_factors) = multi_apply(
             self.loss_by_feat_single, cls_scores, decoded_bboxes, cls_feats,
             angle_preds_list, labels_list, captions_list, captions_ignore_list, label_weights_list,
             bbox_targets_list, assign_metrics_list,
             self.prior_generator.strides)

        cls_avg_factor = reduce_mean(sum(cls_avg_factors)).clamp_(min=1).item()
        losses_cls = list(map(lambda x: x / cls_avg_factor, losses_cls))
        
        losses_caption = list(map(lambda x: x, losses_caption))
        
        loss_ortho = list(map(lambda x: x, loss_ortho))
        
        loss_attr = list(map(lambda x: x, loss_attr))
  
        bbox_avg_factor = reduce_mean(
            sum(bbox_avg_factors)).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / bbox_avg_factor, losses_bbox))
        if self.loss_angle is not None:
            angle_avg_factors = reduce_mean(
                sum(angle_avg_factors)).clamp_(min=1).item()
            losses_angle = list(
                map(lambda x: x / angle_avg_factors, losses_angle))
            return dict(
                loss_cls=losses_cls,
                loss_bbox=losses_bbox,
                loss_angle=losses_angle,
                loss_caption=losses_caption,
                loss_ortho=loss_ortho,
                loss_attr=loss_attr)
        else:
            return dict(loss_cls=losses_cls, loss_bbox=losses_bbox, loss_caption=losses_caption, loss_ortho=loss_ortho, loss_attr=loss_attr)

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
        
    def assign_multi_feat(self,
                          cls_feats,
                          pos_inds):
        """
        function to sample patch embedding
        Args:
            cls_feats (Tensor): [BxNxKxC]
            pos_inds (Tensor): [N]
        Returns:
            Tensor: [NxC]
        """
        B, H, W, C = cls_feats.shape
        
        offset = self.patch_size // 2
        
        # Calculate coordinates for each position
        z = pos_inds // (H * W)  # depth coordinate (z)
        remaining = pos_inds % (H * W)
        y = remaining // W
        x = remaining % W
        
        x[x >= W - 1 - offset] = W - 1 - offset
        y[y >= H - 1 - offset] = H - 1 - offset
        x[x <= offset] = offset
        y[y <= offset] = offset
        
        multi_pos_inds = torch.stack([torch.stack([z, y + i, x + j]) for i in range(-offset, offset + 1) for j in range(-offset, offset + 1)])
        multi_pos_inds = multi_pos_inds.permute(2, 0, 1).reshape(-1, 3)
        # print(multi_pos_inds)
        pos_feats = cls_feats[multi_pos_inds[:, 0], multi_pos_inds[:, 1], multi_pos_inds[:, 2]].reshape(-1, self.patch_size ** 2, C)
        
        return pos_feats
    
    def _get_targets_single(self,
                            cls_scores: Tensor,
                            bbox_preds: Tensor,
                            flat_anchors: Tensor,
                            valid_flags: Tensor,
                            gt_instances: InstanceData,
                            img_meta: dict,
                            gt_instances_ignore: Optional[InstanceData] = None,
                            unmap_outputs=True):
        """Compute regression, classification targets for anchors in a single
        image.

        Args:
            cls_scores (list(Tensor)): Box scores for each image.
            bbox_preds (list(Tensor)): Box energies / deltas for each image.
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors ,4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It usually includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for current image.
            gt_instances_ignore (:obj:`InstanceData`, optional): Instances
                to be ignored during training. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors. Defaults to True.

        Returns:
            tuple: N is the number of total anchors in the image.

            - anchors (Tensor): All anchors in the image with shape (N, 4).
            - labels (Tensor): Labels of all anchors in the image with shape
              (N,).
            - captions (Tensor): Captions of all anchors in the image with shape
              (N,).
            - captions_ignore_flag (Tensor): Captions ignore flag of all anchors in the image with shape
              (N,).
            - label_weights (Tensor): Label weights of all anchor in the
              image with shape (N,).
            - bbox_targets (Tensor): BBox targets of all anchors in the
              image with shape (N, 5).
            - norm_alignment_metrics (Tensor): Normalized alignment metrics
              of all priors in the image with shape (N,).
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg['allowed_border'])
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        pred_instances = InstanceData(
            scores=cls_scores[inside_flags, :],
            bboxes=bbox_preds[inside_flags, :],
            priors=anchors)

        assign_result = self.assigner.assign(pred_instances, gt_instances,
                                             gt_instances_ignore)
                                             
        sampling_result = self.sampler.sample(assign_result, pred_instances,
                                              gt_instances)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = anchors.new_zeros((*anchors.size()[:-1], 5))
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        captions = np.empty((num_valid_anchors, ), dtype=object)
        captions_ignore_flag = np.zeros(num_valid_anchors, dtype=bool)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
        assign_metrics = anchors.new_zeros(
            num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        
        if len(pos_inds) > 0:
            # point-based
            pos_bbox_targets = sampling_result.pos_gt_bboxes
            pos_bbox_targets = pos_bbox_targets.regularize_boxes(
                self.angle_version)
            bbox_targets[pos_inds, :] = pos_bbox_targets

            labels[pos_inds] = sampling_result.pos_gt_labels
            captions[pos_inds.cpu()] = gt_instances['captions'][sampling_result.pos_assigned_gt_inds.cpu()]
            captions_ignore_flag[pos_inds.cpu()] = gt_instances['gt_caption_ignore_flags'][sampling_result.pos_assigned_gt_inds.cpu()]
            if self.train_cfg['pos_weight'] <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg['pos_weight']
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        class_assigned_gt_inds = torch.unique(
            sampling_result.pos_assigned_gt_inds)
        for gt_inds in class_assigned_gt_inds:
            gt_class_inds = pos_inds[sampling_result.pos_assigned_gt_inds ==
                                     gt_inds]
            assign_metrics[gt_class_inds] = assign_result.max_overlaps[
                gt_class_inds]

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(
                labels, num_total_anchors, inside_flags, fill=self.num_classes)
            captions = unmap_np(
                captions, num_total_anchors, inside_flags, fill='')
            
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            assign_metrics = unmap(assign_metrics, num_total_anchors,
                                   inside_flags)
        return (anchors, labels, captions, captions_ignore_flag, label_weights, bbox_targets, assign_metrics,
                sampling_result)

    def get_targets(self,
                    cls_scores: Tensor,
                    bbox_preds: Tensor,
                    anchor_list: List[List[Tensor]],
                    valid_flag_list: List[List[Tensor]],
                    batch_gt_instances: InstanceList,
                    batch_img_metas: List[dict],
                    batch_gt_instances_ignore: OptInstanceList = None,
                    unmap_outputs=True):
        """Compute regression and classification targets for anchors in
        multiple images.

        Args:
            cls_scores (Tensor): Classification predictions of images,
                a 3D-Tensor with shape [num_imgs, num_priors, num_classes].
            bbox_preds (Tensor): Decoded bboxes predictions of one image,
                a 3D-Tensor with shape [num_imgs, num_priors, 4] in [tl_x,
                tl_y, br_x, br_y] format.
            anchor_list (list[list[Tensor]]): Multi level anchors of each
                image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, 4).
            valid_flag_list (list[list[Tensor]]): Multi level valid flags of
                each image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, )
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors. Defaults to True.

        Returns:
            tuple: a tuple containing learning targets.

            - anchors_list (list[list[Tensor]]): Anchors of each level.
            - labels_list (list[Tensor]): Labels of each level.
            - label_weights_list (list[Tensor]): Label weights of each
              level.
            - bbox_targets_list (list[Tensor]): BBox targets of each level.
            - assign_metrics_list (list[Tensor]): alignment metrics of each
              level.
            - captions_list (list[Tensor]): Captions of each level.
            - captions_ignore_list (list[Tensor]): Captions ignore of each level.
        """
        num_imgs = len(batch_img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]

        # concat all level anchors and flags to a single tensor
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            anchor_list[i] = torch.cat(anchor_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        # compute targets for each image
        if batch_gt_instances_ignore is None:
            batch_gt_instances_ignore = [None] * num_imgs
        # anchor_list: list(b * [-1, 4])
        (all_anchors, all_labels, all_captions, all_captions_ignore_flag, all_label_weights, all_bbox_targets,
         all_assign_metrics, sampling_results_list) = multi_apply(
             self._get_targets_single,
             cls_scores.detach(),
             bbox_preds.detach(),
             anchor_list,
             valid_flag_list,
             batch_gt_instances,
             batch_img_metas,
             batch_gt_instances_ignore,
             unmap_outputs=unmap_outputs)
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None

        # split targets to a list w.r.t. multiple levels
        anchors_list = images_to_levels(all_anchors, num_level_anchors)
        labels_list = images_to_levels(all_labels, num_level_anchors)
        captions_list = images_to_levels_np(all_captions, num_level_anchors)
        captions_ignore_list = images_to_levels_np(all_captions_ignore_flag, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        assign_metrics_list = images_to_levels(all_assign_metrics,
                                               num_level_anchors)

        return (anchors_list, labels_list, captions_list, captions_ignore_list, label_weights_list,
                bbox_targets_list, assign_metrics_list, sampling_results_list)
    
    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        angle_preds: List[Tensor],
                        cls_feats: List[Tensor],
                        score_factors: Optional[List[Tensor]] = None,
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = False,
                        with_nms: bool = True) -> InstanceList:
        """Transform a batch of output features extracted from the head into
        bbox results.
        Note: When score_factors is not None, the cls_scores are
        usually multiplied by it then obtain the real score used in NMS,
        such as CenterNess in FCOS, IoU branch in ATSS.
        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            angle_preds (list[Tensor]): Box angle for each scale level
                with shape (N, num_points * angle_dim, H, W)
            cls_feats (list[Tensor]): cls features for all scale levels
            score_factors (list[Tensor], optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, num_priors * 1, H, W). Defaults to None.
            batch_img_metas (list[dict], Optional): Batch image meta info.
                Defaults to None.
            cfg (ConfigDict, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.
        Returns:
            list[:obj:`InstanceData`]: Object detection results of each image
            after the post process. Each item usually contains following keys.
                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 5),
                  the last dimension 5 arrange as (x, y, w, h, t).
                - captions (Tensor): Has a shape (num_instances, )
        """
        assert len(cls_scores) == len(bbox_preds)

        if score_factors is None:
            # e.g. Retina, FreeAnchor, Foveabox, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, AutoAssign, etc.
            with_score_factors = True
            assert len(cls_scores) == len(score_factors)

        num_levels = len(cls_scores)

        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device)

        result_list = []

        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            cls_score_list = select_single_mlvl(
                cls_scores, img_id, detach=True)
            bbox_pred_list = select_single_mlvl(
                bbox_preds, img_id, detach=True)
            angle_pred_list = select_single_mlvl(
                angle_preds, img_id, detach=True)
            cls_feats_list = select_single_mlvl(
                cls_feats, img_id, detach=True)
            if with_score_factors:
                score_factor_list = select_single_mlvl(
                    score_factors, img_id, detach=True)
            else:
                score_factor_list = [None for _ in range(num_levels)]

            results = self._predict_by_feat_single(
                cls_score_list=cls_score_list,
                bbox_pred_list=bbox_pred_list,
                angle_pred_list=angle_pred_list,
                cls_feats_list=cls_feats_list,
                score_factor_list=score_factor_list,
                mlvl_priors=mlvl_priors,
                img_meta=img_meta,
                cfg=cfg,
                rescale=rescale,
                with_nms=with_nms)
            result_list.append(results)
        return result_list

    def _predict_by_feat_single(self,
                                cls_score_list: List[Tensor],
                                bbox_pred_list: List[Tensor],
                                angle_pred_list: List[Tensor],
                                cls_feats_list: List[Tensor],
                                score_factor_list: List[Tensor],
                                mlvl_priors: List[Tensor],
                                img_meta: dict,
                                cfg: ConfigDict,
                                rescale: bool = False,
                                with_nms: bool = True) -> InstanceData:
        """Transform a single image's features extracted from the head into
        bbox results.
        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_priors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has shape
                (num_priors * 4, H, W).
            angle_pred_list (list[Tensor]): Box angle for a single scale
                level with shape (N, num_points * angle_dim, H, W).
            cls_feats_list (list[Tensor]): Box features from all scale
                levels of a single image, each item has shape
                (num_priors * num_classes, H, W).
            score_factor_list (list[Tensor]): Score factor from all scale
                levels of a single image, each item has shape
                (num_priors * 1, H, W).
            mlvl_priors (list[Tensor]): Each element in the list is
                the priors of a single level in feature pyramid. In all
                anchor-based methods, it has shape (num_priors, 4). In
                all anchor-free methods, it has shape (num_priors, 2)
                when `with_stride=True`, otherwise it still has shape
                (num_priors, 4).
            img_meta (dict): Image meta info.
            cfg (mmengine.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.
        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.
                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 5),
                  the last dimension 5 arrange as (x, y, w, h, t).
                - captions (Tensor): Has a shape (num_instances, )
                  
        """
        if score_factor_list[0] is None:
            # e.g. Retina, FreeAnchor, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, etc.
            with_score_factors = True

        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bbox_preds = []
        mlvl_valid_priors = []
        mlvl_scores = []
        mlvl_labels = []
        mlvl_feats = []
        if with_score_factors:
            mlvl_score_factors = []
        else:
            mlvl_score_factors = None
        for level_idx, (
                cls_score, bbox_pred, angle_pred, cls_feats, score_factor, priors) in \
                enumerate(zip(cls_score_list, bbox_pred_list, angle_pred_list, cls_feats_list,
                              score_factor_list, mlvl_priors)):

            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            angle_pred = angle_pred.permute(1, 2, 0).reshape(
                -1, self.angle_coder.encode_size)
            if with_score_factors:
                score_factor = score_factor.permute(1, 2,
                                                    0).reshape(-1).sigmoid()
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                # remind that we set FG labels to [0, num_class-1]
                # since mmdet v2.0
                # BG cat_id: num_class
                scores = cls_score.softmax(-1)[:, :-1]

            # After https://github.com/open-mmlab/mmdetection/pull/6268/,
            # this operation keeps fewer bboxes under the same `nms_pre`.
            # There is no difference in performance for most models. If you
            # find a slight drop in performance, you can set a larger
            # `nms_pre` than before.
            score_thr = cfg.get('score_thr', 0)

            results = filter_scores_and_topk(
                scores, score_thr, nms_pre,
                dict(
                    bbox_pred=bbox_pred, angle_pred=angle_pred, priors=priors))
            scores, labels, keep_idxs, filtered_results = results
            
            feats_pred = cls_feats.permute(1, 2, 0)
            feats_pred = self.assign_multi_feat(feats_pred.unsqueeze(0), keep_idxs)

            bbox_pred = filtered_results['bbox_pred']
            angle_pred = filtered_results['angle_pred']
            priors = filtered_results['priors']

            decoded_angle = self.angle_coder.decode(angle_pred, keepdim=True)
            bbox_pred = torch.cat([bbox_pred, decoded_angle], dim=-1)

            if with_score_factors:
                score_factor = score_factor[keep_idxs]
                

            mlvl_bbox_preds.append(bbox_pred)
            mlvl_valid_priors.append(priors)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)
            mlvl_feats.append(feats_pred)

            if with_score_factors:
                mlvl_score_factors.append(score_factor)

        bbox_pred = torch.cat(mlvl_bbox_preds)
        priors = cat_boxes(mlvl_valid_priors)
        bboxes = self.bbox_coder.decode(priors, bbox_pred, max_shape=img_shape)

        results = InstanceData()
        results.bboxes = RotatedBoxes(bboxes)
        results.scores = torch.cat(mlvl_scores)
        results.labels = torch.cat(mlvl_labels)
        results.feats = torch.cat(mlvl_feats)
        
                
        if with_score_factors:
            results.score_factors = torch.cat(mlvl_score_factors)

        post_results = self._bbox_post_process(
            results=results,
            cfg=cfg,
            rescale=rescale,
            with_nms=with_nms,
            img_meta=img_meta)
        
        # caption pred
        if 'caption' in self.test_cfg and self.test_cfg.caption == True:
            post_feats = self.multimodal_up_layer(post_results.feats).bfloat16()
            if len(post_feats) > 0:
                if self.caption_type != "global":
                    post_feats, _ = self.decompose(post_feats, self.base_emb.unsqueeze(0))
                post_results.captions = self.chat(self.tokenizer, post_feats, self.generation_config)
            else:
                if self.caption_type != "global":
                    post_feats = post_feats.mean(-1).mean(-1).unsqueeze(-1).repeat(1, self.caption_dim)
                post_results.captions = []  
        return post_results

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
            bs: Optional[int] = 64,
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



def unmap_np(data, count, inds, fill=0):
    """Unmap a subset of item (data) back to the original set of items (of size
    count)"""
    if data.ndim == 1:
        ret = np.full(count, fill, dtype=object)
        ret[inds.type(torch.bool).cpu()] = data
    else:
        new_size = (count, ) + data.shape[1:]
        ret = np.full(new_size, '', dtype=object)
        ret[inds.type(torch.bool).cpu(), :] = data
    return ret

def images_to_levels_np(target, num_levels):
    """Convert targets by image to targets by feature level.

    [target_img0, target_img1] -> [target_level0, target_level1, ...]
    """
    target = stack_boxes_np(target, 0)
    level_targets = []
    start = 0
    for n in num_levels:
        end = start + n
        # level_targets.append(target[:, start:end].squeeze(0))
        level_targets.append(target[:, start:end])
        start = end
    return level_targets

def stack_boxes_np(data_list, dim: int = 0):
    """Stack boxes with type of tensor or box type.

    Args:
        data_list (List[Union[Tensor, :obj:`BaseBoxes`]]): A list of tensors
            or box types need to be stacked.
        dim (int): The dimension over which the box are stacked.
            Defaults to 0.

    Returns:
        Union[Tensor, :obj`BaseBoxes`]: Stacked results.
    """
    return np.stack(data_list, axis=dim)