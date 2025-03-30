# Copyright (c) OpenMMLab. All rights reserved.
import random
import warnings
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from mmengine.registry import FUNCTIONS

# FUNCTIONS is new in MMEngine v0.7.0. Reserve the `COLLATE_FUNCTIONS` to keep
# the compatibility.
COLLATE_FUNCTIONS = FUNCTIONS
IGNORE_INDEX = -100

def worker_init_fn(worker_id: int,
                   num_workers: int,
                   rank: int,
                   seed: int,
                   disable_subprocess_warning: bool = False) -> None:
    """This function will be called on each worker subprocess after seeding and
    before data loading.

    Args:
        worker_id (int): Worker id in [0, num_workers - 1].
        num_workers (int): How many subprocesses to use for data loading.
        rank (int): Rank of process in distributed environment. If in
            non-distributed environment, it is a constant number `0`.
        seed (int): Random seed.
    """
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    if disable_subprocess_warning and worker_id != 0:
        warnings.simplefilter('ignore')

def concat_pad_data(features, max_item_length=None, pad_id=0):
    # first = features[0]
    # batch = []
    if len(features) == 0:
        return features

    batch_lens = [len(feat['input_ids']) for feat in features]
    max_item_length = max_item_length or max(batch_lens)
    for idx in range(len(features)):
        feat = features[idx]
        if not isinstance(feat['input_ids'], torch.Tensor):
            feat['input_ids'] = torch.tensor(feat['input_ids'])
            feat['labels'] = torch.tensor(feat['labels'])
            feat['attention_mask'] = torch.tensor(feat['attention_mask'])
            feat['position_ids'] = torch.tensor(feat['position_ids'])
        
        temp_input_ids = torch.LongTensor([pad_id] * max_item_length)
        temp_input_ids[:feat['input_ids'].shape[0]] = feat['input_ids']
        feat['input_ids'] = temp_input_ids
        temp_labels = torch.LongTensor([IGNORE_INDEX] * max_item_length)
        temp_labels[:feat['labels'].shape[0]] = feat['labels']
        feat['labels'] = temp_labels
        feat['attention_mask'] = feat['input_ids'].ne(pad_id)

        if 'position_ids' in feat:
            temp_position_ids = torch.LongTensor([pad_id] * max_item_length)
            temp_position_ids[:feat['position_ids'].shape[0]] = feat['position_ids']
            feat['position_ids'] = temp_position_ids

        if 'loss_weight' in feat:
            temp_loss_weight = torch.FloatTensor([pad_id] * max_item_length)
            temp_loss_weight[:feat['loss_weight'].shape[0]] = feat['loss_weight']
            feat['loss_weight'] = temp_loss_weight
    return features
    
@FUNCTIONS.register_module()
def ev_collate(data_batch: Sequence) -> Any:
    """Convert list of data sampled from dataset into a batch of data, of which
    type consistent with the type of each data_itement in ``data_batch``.

    The default behavior of dataloader is to merge a list of samples to form
    a mini-batch of Tensor(s). However, in MMEngine, ``pseudo_collate``
    will not stack tensors to batch tensors, and convert int, float, ndarray to
    tensors.

    This code is referenced from:
    `Pytorch default_collate <https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py>`_.

    Args:
        data_batch (Sequence): Batch of data from dataloader.

    Returns:
        Any: Transversed Data in the same format as the data_itement of
        ``data_batch``.
    """  # noqa: E501
    data_item = data_batch[0]
    data_item_type = type(data_item)
    if isinstance(data_item, (str, bytes)):
        return data_batch
    elif isinstance(data_item, tuple) and hasattr(data_item, '_fields'):
        # named tuple
        return data_item_type(*(ev_collate(samples)
                                for samples in zip(*data_batch)))
    elif isinstance(data_item, Sequence):
        # check to make sure that the data_itements in batch have
        # consistent size
        it = iter(data_batch)
        data_item_size = len(next(it))
        if not all(len(data_item) == data_item_size for data_item in it):
            raise RuntimeError(
                'each data_itement in list of batch should be of equal size')
        transposed = list(zip(*data_batch))

        if isinstance(data_item, tuple):
            return [ev_collate(samples)
                    for samples in transposed]  # Compat with Pytorch.
        else:
            try:
                return data_item_type(
                    [ev_collate(samples) for samples in transposed])
            except TypeError:
                # The sequence type may not support `__init__(iterable)`
                # (e.g., `range`).
                return [ev_collate(samples) for samples in transposed]
    elif isinstance(data_item, Mapping):
        return data_item_type({
            key: ev_collate([d[key] for d in data_batch])
            for key in data_item
        })
    elif isinstance(data_item, torch.Tensor):
        return data_batch
    else:
        _data_cat = np.concatenate([data_batch[i].gt_instances.captions for i in range(len(data_batch))])
        # _data_lengths = [len(data_batch[i].gt_instances.captions) for i in range(len(data_batch))]
        # _split_indices = np.cumsum(_data_lengths)[:-1]  # 计算分割的索引位置
        _ = concat_pad_data(_data_cat)
        # _data_batch = np.split(_data_pad, _split_indices)
        
        # for i, j in zip(data_batch, _data_batch):
        #     data_batch[i].gt_instances.captions = j
        
        return data_batch