# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os.path as osp
from typing import List

from mmengine.dataset import BaseDataset

from mmrotate.registry import DATASETS

from internvl.train.dataset import (preprocess,
                                    preprocess_internlm,
                                    preprocess_internvl2_5, 
                                    preprocess_mpt,
                                    preprocess_phi3)


from internvl.train.constants import (BOX_END_TOKEN, BOX_START_TOKEN,
                                      IMG_CONTEXT_TOKEN, IMG_END_TOKEN,
                                      IMG_START_TOKEN, QUAD_END_TOKEN,
                                      QUAD_START_TOKEN, REF_END_TOKEN,
                                      REF_START_TOKEN)
from transformers import AutoTokenizer
from tqdm import tqdm
import os
import json
import torch

@DATASETS.register_module()
class FAIR1MDataset(BaseDataset):
    """FAIR1M dataset for detection.

    Args:
        diff_thr (int): The difficulty threshold of ground truth. Bboxes
            with difficulty higher than it will be ignored. The range of this
            value should be non-negative integer. Defaults to 100.
        img_suffix (str): The suffix of images. Defaults to 'png'.
    """
    METAINFO = {
        'classes':
            ('Passenger Ship', 'Motorboat', 'Fishing Boat','Tugboat', 
             'other-ship', 'Engineering Ship', 'Liquid Cargo Ship','Dry Cargo Ship', 
             'Warship', 'Small Car', 'Bus', 'Cargo Truck',
             'Dump Truck', 'other-vehicle', 'Van', 'Trailer', 
             'Tractor', 'Excavator', 'Truck Tractor', 'Boeing737', 
             'Boeing747', 'Boeing777', 'Boeing787', 'ARJ21', 
             'C919', 'A220', 'A321', 'A330', 
             'A350', 'other-airplane', 'Baseball Field', 'Basketball Court',
             'Football Field', 'Tennis Court', 'Roundabout', 'Intersection', 
             'Bridge'),
        'palette': [(165, 42, 42), (189, 183, 107), (0, 255, 0), (255, 0, 0),
                    (138, 43, 226), (255, 128, 0), (255, 0, 255),
                    (0, 255, 255), (255, 193, 193), (0, 51, 153),
                    (255, 250, 205)]
    }

    def __init__(self,
                 diff_thr: int = 100,
                 img_suffix: str = 'png',
                 pretrained: str = "/mnt/data1/jianghx/ckpts/InternVL2-1B",
                 max_seq_length = 4096,
                 num_image_token = 64,
                 template_name = "Hermes-2",
                 **kwargs) -> None:
        self.diff_thr = diff_thr
        self.img_suffix = img_suffix
        self.template_name = template_name
        self.preprocess_function = self.get_preprocess_function()
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained, add_eos_token=False, trust_remote_code=True, use_fast=False)
        self.tokenizer.tokenizer_path = pretrained
        self.tokenizer.model_max_length = max_seq_length
        self.tokenizer.padding_side = 'left'
        self.num_image_token = num_image_token
        self.tokenizer_name = pretrained.split("/")[-1]
        self.attributes = [
            'ship-visibility', 'ship-purpose', 'ship-motion', 'ship-capacity', 'ship-load-status', 
            'ship-cargo-status', 'ship-mooring-status', 'hull-color', 'hull-size', 'hull-shadow', 'hull-outline', 
            'superstructure-color', 'superstructure-size', 'superstructure-height', 'superstructure-position', 'paint-condition', 
            'bow-design', 'stern-design', 'deck-utilization', 'deck-condition', 'deck-obstacles', 'deck-color', 'deck-structure', 'deck-accessories', 'passenger-facilities', 'container-presence', 'container-count', 'container-color', 'container-layout', 'container-alignment', 'container-densities', 'container-type', 'machinery-presence', 'location', 'weather-condition', 'water-color', 'water-turbulence', 'unique-attributes', 'engine-color', 'engine-location', 'engine-size', 'engine-type', 'engines-number', 'engines-shape', 'engines-visible', 'fuselage-color', 'fuselage-length', 'fuselage-material', 'fuselage-shape', 'nose-cone-color', 'propeller-count', 'tail-color', 'tail-height', 'tail-material', 'tail-shape', 'tail-type', 'wings-angle', 'wings-color', 'wings-material', 'wings-shape', 'wings-span', 'wings-type']
        
        super().__init__(**kwargs)


    def get_preprocess_function(self):
        # Select the appropriate preprocessing function based on the template name
        if self.template_name == 'Hermes-2':
            preprocess_function = preprocess_mpt
        elif self.template_name == 'internlm2-chat':
            preprocess_function = preprocess_internlm
        elif self.template_name == 'phi3-chat':
            preprocess_function = preprocess_phi3
        elif self.template_name == 'internvl2_5':
            preprocess_function = preprocess_internvl2_5
        else:
            preprocess_function = preprocess
        return preprocess_function
    
    def prepocess_name(self, s):
        for i in self.metainfo["classes"]:
            if "</cls_name>{}</cls_name>".format(i) in s:
                name = i
                tmp_s = s.replace("</cls_name>{}</cls_name>".format(i), "")
                bbox_info = tmp_s.split(maxsplit=9)
                break
        return name, bbox_info
    
    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``
        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        cls_map = {c: i
                   for i, c in enumerate(self.metainfo['classes'])
                   }  # in mmdet v2.0 label is 0-based
        data_list = []
        if self.ann_file == '':
            img_files = glob.glob(
                osp.join(self.data_prefix['img_path'], f'*.{self.img_suffix}'))
            for img_path in img_files:
                data_info = {}
                data_info['img_path'] = img_path
                img_name = osp.split(img_path)[1]
                data_info['file_name'] = img_name
                img_id = img_name[:-4]
                data_info['img_id'] = img_id

                instance = dict(bbox=[], bbox_label=[], ignore_flag=0)
                data_info['instances'] = [instance]
                data_list.append(data_info)

            return data_list
        else:
            txt_files = glob.glob(osp.join(self.ann_file, '*.txt'))
            if len(txt_files) == 0:
                raise ValueError('There is no txt file in '
                                 f'{self.ann_file}')
            meta_path = os.path.join(self.ann_file.split("annfiles")[0], "metainfo_{}_{}_token{}.json".format(self.tokenizer_name, 
                                                                                                              self.template_name, 
                                                                                                              self.num_image_token))
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as file:
                    metainfo = json.load(file)
                                
                for v in tqdm(metainfo):
                    if len(v["instances"]) > 0:
                        for i in range(len(v["instances"])):
                            v["instances"][i]["bbox_caption"]["input_ids"] = torch.tensor(v["instances"][i]["bbox_caption"]["input_ids"])
                            v["instances"][i]["bbox_caption"]["labels"] = torch.tensor(v["instances"][i]["bbox_caption"]["labels"])
                            v["instances"][i]["bbox_caption"]["attention_mask"] = torch.tensor(v["instances"][i]["bbox_caption"]["attention_mask"])
                            v["instances"][i]["bbox_caption"]["position_ids"] = torch.tensor(v["instances"][i]["bbox_caption"]["position_ids"])
                            
                    data_list.append(v)
            else:
                num_with_caption = 0
                for txt_file in tqdm(txt_files):
                    data_info = {}
                    img_id = osp.split(txt_file)[1][:-4]
                    data_info['img_id'] = img_id
                    img_name = img_id + f'.{self.img_suffix}'
                    data_info['file_name'] = img_name
                    data_info['img_path'] = osp.join(self.data_prefix['img_path'],
                                                    img_name)

                    instances = []
                    with open(txt_file) as f:
                        s = f.readlines()
                        for si in s:
                            instance = {}
                            name, bbox_info = self.prepocess_name(si)
                            instance['bbox'] = [float(i) for i in bbox_info[:8]]
                            cls_name = name
                            if cls_name not in cls_map:
                                continue
                            instance['bbox_label'] = cls_map[cls_name]
                            difficulty = int(bbox_info[8])
                            if difficulty > self.diff_thr:
                                instance['ignore_flag'] = 1
                            else:
                                instance['ignore_flag'] = 0
                            info_dict = bbox_info[9]
                            if isinstance(info_dict, str):
                                info_dict = json.loads(info_dict)
                            if info_dict:
                                num_with_caption += 1
                                instance["caption_ignore_flag"] = 0
                            else:
                                instance["caption_ignore_flag"] = 1
                            info_dict = {k:v for k, v in info_dict.items() if k != "confidence" and str(v).lower() != "unknown"}
                            bbox_caption_ans = 'This object is a "{}".'.format(cls_name)
                            if info_dict:
                                bbox_caption_ans += ' Its '
                            for i, (k, v) in enumerate(info_dict.items()):
                                bbox_caption_ans += '{} is {}'.format(k, v)
                                if i == len(info_dict.items()) - 1:
                                    if not bbox_caption_ans.endswith("."):
                                        bbox_caption_ans += '.'
                                else:
                                    bbox_caption_ans += ', '
                            bbox_caption_ans = bbox_caption_ans.strip()
                            bbox_question = '''This image includes a remote sensing object in a bird's-eye view. Please help me to explain the visual content of this object in a fine-grained manner.\n<image>'''
                            caption = [[
                                {
                                    'from': 'human', 
                                    'value': bbox_question
                                },
                                {
                                    'from': 'gpt', 
                                    'value': bbox_caption_ans
                                }
                            ]]
                            
                            ret = self.preprocess_function(self.template_name, caption,
                                                    self.tokenizer, [self.num_image_token],
                                                    group_by_length=True,
                                                    use_packed_ds=False,
                                                    ds_name="fair1m")
                            # Calculate position_ids for packed dataset
                            position_ids = ret['attention_mask'].long().cumsum(-1) - 1
                            position_ids.masked_fill_(ret['attention_mask'] == 0, 1)
                            attr_id = []
                            attr_token = []
                            for i, (k, v) in enumerate(info_dict.items()):
                                attr_id.append(self.attributes.index(k))
                                attr_token.append(self.tokenizer.encode(str(v), add_special_tokens=False))

                            # Create the final return dictionary
                            ret = dict(
                                input_ids=ret['input_ids'][0].tolist(),
                                labels=ret['labels'][0].tolist(),
                                attention_mask=ret['attention_mask'][0].tolist(),
                                position_ids=position_ids[0].tolist(),
                                disentangle_range=[attr_id, attr_token]
                            )
                            
                            instance['bbox_caption'] = ret
                            instances.append(instance)
                    data_info['instances'] = instances
                    data_list.append(data_info)
                
                with open(meta_path, 'w') as file:
                    json.dump(data_list, file, indent=4, ensure_ascii=False)  
            return data_list

    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg.

        Returns:
            List[dict]: Filtered results.
        """
        if self.test_mode:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False) \
            if self.filter_cfg is not None else False

        valid_data_infos = []
        for i, data_info in enumerate(self.data_list):
            if filter_empty_gt and len(data_info['instances']) == 0:
                continue
            valid_data_infos.append(data_info)
        
        return valid_data_infos

    def get_cat_ids(self, idx: int) -> List[int]:
        """Get FAIR1M category ids by index.

        Args:
            idx (int): Index of data.
        Returns:
            List[int]: All categories in the image of specified index.
        """

        instances = self.get_data_info(idx)['instances']
        return [instance['bbox_label'] for instance in instances]
    