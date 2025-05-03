# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
import os
from typing import Optional, Sequence, Union
import mmcv
import numpy as np
import torch.nn as nn
import copy
from mmdet.apis import inference_detector
from mmdet.evaluation import get_classes

from mmrotate.utils import register_all_modules

from mmengine.config import Config
from mmengine.model.utils import revert_sync_batchnorm
from mmengine.registry import init_default_scope, DATASETS, VISUALIZERS
from mmengine.runner.checkpoint import _load_checkpoint, _load_checkpoint_to_model

from pathlib import Path

from mmdet.apis.inference import Path, Config, init_default_scope, MODELS, revert_sync_batchnorm, warnings


ImagesType = Union[str, np.ndarray, Sequence[str], Sequence[np.ndarray]]

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img_dir', help='Image file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--with-attribute', action='store_true', help='perform attribute understanding')
    parser.add_argument(
        '--patch_sizes',
        type=int,
        nargs='+',
        default=[1024],
        help='The sizes of patches')
    parser.add_argument(
        '--patch_steps',
        type=int,
        nargs='+',
        default=[824],
        help='The steps between two patches')
    parser.add_argument(
        '--img_ratios',
        type=float,
        nargs='+',
        default=[1.0],
        help='Image resizing ratios for multi-scale detecting')
    parser.add_argument(
        '--merge_iou_thr',
        type=float,
        default=0.1,
        help='IoU threshould for merging results')
    parser.add_argument(
        '--merge_nms_type',
        default='nms_rotated',
        choices=['nms', 'nms_rotated', 'nms_quadri'],
        help='NMS type for merging results')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='random',
        choices=['dota', 'sar', 'hrsc', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.4, help='bbox score threshold')
    args = parser.parse_args()
    return args


def load_checkpoint(model,
                    filename,
                    map_location=None,
                    strict=False,
                    logger=None,
                    revise_keys=[(r'^module\.', '')]):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.
        revise_keys (list): A list of customized keywords to modify the
            state_dict in checkpoint. Each item is a (pattern, replacement)
            pair of the regular expression operations. Defaults to strip
            the prefix 'module.' by [(r'^module\\.', '')].

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    checkpoint = _load_checkpoint(filename, map_location, logger)
    checkpoint["state_dict"] = checkpoint["module"]
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f'No state_dict found in checkpoint file {filename}')

    return _load_checkpoint_to_model(model, checkpoint, strict, logger,
                                     revise_keys)
    
def init_detector(
    config: Union[str, Path, Config],
    checkpoint: Optional[str] = None,
    palette: str = 'none',
    device: str = 'cuda:0',
    cfg_options: Optional[dict] = None,
) -> nn.Module:
    """Initialize a detector from config file.

    Args:
        config (str, :obj:`Path`, or :obj:`mmengine.Config`): Config file path,
            :obj:`Path`, or the config object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        palette (str): Color palette used for visualization. If palette
            is stored in checkpoint, use checkpoint's palette first, otherwise
            use externally passed palette. Currently, supports 'coco', 'voc',
            'citys' and 'random'. Defaults to none.
        device (str): The device where the anchors will be put on.
            Defaults to cuda:0.
        cfg_options (dict, optional): Options to override some settings in
            the used config.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, (str, Path)):
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)
    elif 'init_cfg' in config.model.backbone:
        config.model.backbone.init_cfg = None

    scope = config.get('default_scope', 'mmdet')
    if scope is not None:
        init_default_scope(config.get('default_scope', 'mmdet'))

    model = MODELS.build(config.model)
    model = revert_sync_batchnorm(model)
    if checkpoint is None:
        warnings.simplefilter('once')
        warnings.warn('checkpoint is None, use COCO classes by default.')
        model.dataset_meta = {'classes': get_classes('coco')}
    else:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        # Weights converted from elsewhere may not have meta fields.
        checkpoint_meta = checkpoint.get('meta', {})

        # save the dataset_meta in the model for convenience
        if 'dataset_meta' in checkpoint_meta:
            # mmdet 3.x, all keys should be lowercase
            model.dataset_meta = {
                k.lower(): v
                for k, v in checkpoint_meta['dataset_meta'].items()
            }
        elif 'CLASSES' in checkpoint_meta:
            # < mmdet 3.x
            classes = checkpoint_meta['CLASSES']
            model.dataset_meta = {'classes': classes}
        else:
            warnings.simplefilter('once')
            warnings.warn(
                'dataset_meta or class names are not saved in the '
                'checkpoint\'s meta data, use COCO classes by default.')
            model.dataset_meta = {'classes': get_classes('coco')}

    # Priority:  args.palette -> config -> checkpoint
    if palette != 'none':
        model.dataset_meta['palette'] = palette
    else:
        test_dataset_cfg = copy.deepcopy(config.test_dataloader.dataset)
        # lazy init. We only need the metainfo.
        test_dataset_cfg['lazy_init'] = True
        metainfo = DATASETS.build(test_dataset_cfg).metainfo
        cfg_palette = metainfo.get('palette', None)
        if cfg_palette is not None:
            model.dataset_meta['palette'] = cfg_palette
        else:
            if 'palette' not in model.dataset_meta:
                warnings.warn(
                    'palette does not exist, random is used by default. '
                    'You can also set the palette to customize.')
                model.dataset_meta['palette'] = 'random'

    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model
    
def add_(model, with_attribute):
    if 'rcnn' in model.test_cfg:
        model.test_cfg.rcnn.caption = with_attribute
    else:
        model.test_cfg.caption = with_attribute
    return model


def main(args):
    # register all modules in mmrotate into the registries
    register_all_modules()
    
    # build the model from a config file and a checkpoint file
    model = init_detector(
        args.config, args.checkpoint, palette=args.palette, device=args.device)
    model = add_(model, args.with_attribute)
    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_detector
    visualizer.dataset_meta = model.dataset_meta
    # test a image
    result = inference_detector(model, args.img_dir)

    # show the results
    img = mmcv.imread(args.img_dir)
    img = mmcv.imconvert(img, 'bgr', 'rgb')
    os.makedirs("results", exist_ok=True)
    out_dir = os.path.join("results/ans.png")
    visualizer.add_datasample(
        'result',
        img,
        data_sample=result,
        draw_gt=False,
        show=out_dir is None,
        wait_time=0,
        out_file=out_dir,
        pred_score_thr=args.score_thr)


if __name__ == '__main__':
    args = parse_args()
    main(args)
