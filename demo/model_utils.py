import os
import mmcv
import numpy as np
from pathlib import Path
import torch.nn as nn
from typing import Union, Optional
from mmdet.apis import inference_detector
from mmrotate.utils import register_all_modules
from mmengine.config import Config
from mmengine.registry import init_default_scope, DATASETS, VISUALIZERS
from mmengine.runner.checkpoint import _load_checkpoint, _load_checkpoint_to_model
from mmdet.evaluation import get_classes
from mmdet.apis.inference import MODELS, revert_sync_batchnorm
import copy
import warnings

def load_checkpoint(model, filename, map_location=None, strict=False):
    checkpoint = _load_checkpoint(os.path.join(filename, "mp_rank_00_model_states.pt"), map_location)
    checkpoint["state_dict"] = checkpoint["module"]
    return _load_checkpoint_to_model(model, checkpoint, strict)

def init_detector(config, checkpoint, palette='random', device='cuda:0'):
    config = Config.fromfile(config)
    init_default_scope(config.get('default_scope', 'mmdet'))

    model = MODELS.build(config.model)
    model = revert_sync_batchnorm(model)

    if checkpoint:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        if 'dataset_meta' in checkpoint.get('meta', {}):
            model.dataset_meta = {k.lower(): v for k, v in checkpoint['meta']['dataset_meta'].items()}
        else:
            model.dataset_meta = {'classes': get_classes('coco')}
    else:
        model.dataset_meta = {'classes': get_classes('coco')}

    if palette:
        model.dataset_meta['palette'] = palette

    model.cfg = config
    model.to(device)
    model.eval()
    return model

def run_inference(model, image_path, with_attribute=False, score_thr=0.4):
    register_all_modules()
    
    # Add attribute flag to model test config
    if 'rcnn' in model.test_cfg:
        model.test_cfg.rcnn.caption = with_attribute
    else:
        model.test_cfg.caption = with_attribute

    # Run inference
    result = inference_detector(model, image_path)

    # Initialize visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    # Read and convert image
    img = mmcv.imread(image_path)
    img = mmcv.imconvert(img, 'bgr', 'rgb')

    out_path = "demo/result.png"
    
    # Visualize result with threshold
    attr_text = visualizer.add_datasample(
							'result',
							img,
							data_sample=result,
							show=False,
							wait_time=0,
							out_file=out_path,
							pred_score_thr=score_thr)

    # Extract attribute info
    if with_attribute and len(attr_text) == 0:
        attr_text = "⚠️ Attribute info not available from model output."
    
    return out_path, attr_text
