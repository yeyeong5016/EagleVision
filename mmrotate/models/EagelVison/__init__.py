# Copyright (c) OpenMMLab. All rights reserved.
from .EVDetector import EagleVision
from .EVHead import EagleVisionHead
from .EVProHead import EagleVisionProHead, EagleVisionProROIHead

__all__ = ['EagleVision', 'EagleVisionHead', 'EagleVisionProHead', 'EagleVisionProROIHead']
