# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import os.path as osp
import re
import tempfile
import zipfile
from collections import OrderedDict, defaultdict
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
from mmcv.ops import nms_quadri, nms_rotated
from mmengine.evaluator import BaseMetric
from mmengine.fileio import dump
from mmengine.logging import MMLogger

from mmrotate.evaluation import eval_rbbox_map
from mmrotate.registry import METRICS
from mmrotate.structures.bbox import rbox2qbox, qbox2rbox
from xml.dom.minidom import Document
from PIL import Image
from mmcv.ops import box_iou_quadri, box_iou_rotated
import json
import xml.etree.ElementTree as ET
from mmrotate.evaluation.functional.mean_ap import get_cls_results

from multiprocessing import get_context

@METRICS.register_module()
class EVBench(BaseMetric):
    """EagleVision benchmark that includes object detection metrics, online evaluation output, 
    and attribute caption evaluation.

    Note:  In addition to format the output results to JSON like CocoMetric,
    it can also generate the full image's results by merging patches' results.
    The premise is that you must use the tool provided by us to crop the 
    large images, which can be found at: ``tools/data/dota/split``.

    Args:
        iou_thrs (float or List[float]): IoU threshold. Defaults to 0.5.
        scale_ranges (List[tuple], optional): Scale ranges for evaluating
            mAP. If not specified, all bounding boxes would be included in
            evaluation. Defaults to None.
        metric (str | list[str]): Metrics to be evaluated. Only support
            'mAP' now. If is list, the first setting in the list will
             be used to evaluate metric.
        predict_box_type (str): Box type of model results. If the QuadriBoxes
            is used, you need to specify 'qbox'. Defaults to 'rbox'.
        format_only (bool): Format the output results without perform
            evaluation. It is useful when you want to format the result
            to a specific format. Defaults to False.
        outfile_prefix (str, optional): The prefix of json/zip files. It
            includes the file path and the prefix of filename, e.g.,
            "a/b/prefix". If not specified, a temp file will be created.
            Defaults to None.
        merge_patches (bool): Generate the full image's results by merging
            patches' results.
        iou_thr (float): IoU threshold of ``nms_rotated`` used in merge
            patches. Defaults to 0.1.
        eval_mode (str): 'area' or '11points', 'area' means calculating the
            area under precision-recall curve, '11points' means calculating
            the average precision of recalls at [0, 0.1, ..., 1].
            The PASCAL VOC2007 defaults to use '11points', while PASCAL
            VOC2012 defaults to use 'area'. Defaults to '11points'.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
        task (List[str]): List of tasks to be performed. Defaults to ['Task2'].
        img_path (str): Path to the images. Defaults to None.
        xml_path (str): Path to the xml files. Defaults to None.
        caption_gt_path (str): Path to the ground truth captions. Defaults to None.
    """

    default_prefix: Optional[str] = 'fair1m'

    def __init__(self,
                 iou_thrs: Union[float, List[float]] = 0.5,
                 scale_ranges: Optional[List[tuple]] = None,
                 metric: Union[str, List[str]] = 'mAP',
                 predict_box_type: str = 'rbox',
                 format_only: bool = False,
                 outfile_prefix: Optional[str] = None,
                 merge_patches: bool = False,
                 iou_thr: float = 0.1,
                 eval_mode: str = '11points',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 task: List[str] = ['Task2'],
                 img_path: str = None,
                 xml_path: str = None,
                 caption_gt_path: str = None,
                 ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.iou_thrs = [iou_thrs] if isinstance(iou_thrs, float) \
            else iou_thrs
        assert isinstance(self.iou_thrs, list)
        self.scale_ranges = scale_ranges
        # voc evaluation metrics
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP']
        if metric not in allowed_metrics:
            raise KeyError(f"metric should be one of 'mAP', but got {metric}.")
        self.metric = metric
        self.predict_box_type = predict_box_type

        self.format_only = format_only
        if self.format_only:
            assert outfile_prefix is not None, 'outfile_prefix must be not'
            'None when format_only is True, otherwise the result files will'
            'be saved to a temp directory which will be cleaned up at the end.'

        self.outfile_prefix = outfile_prefix
        self.merge_patches = merge_patches
        self.iou_thr = iou_thr

        self.use_07_metric = True if eval_mode == '11points' else False
        self.task = task
        if "Task1" in task:
            assert img_path != None
        if "Task2" in task:
            assert xml_path != None and caption_gt_path != None
        self.img_path = img_path
        self.xml_path = xml_path
        self.caption_gt_path = caption_gt_path
        self.load_gts(xml_path, caption_gt_path)

    def load_gts(self, xml_path, caption_gt_path):
        if xml_path == None or caption_gt_path == None:
            return
        
        # Load caption ground truth
        with open(caption_gt_path, 'r') as f:
            caption_gt = json.load(f)

        # Store object coordinates matrix and captions
        bbox_dict = {}
        caption_dict = {}

        # Build a mapping from img_id to objects for efficient lookup
        img_id_to_objs = {str(item["img_id"]): item["objs"] for item in caption_gt}

        # Iterate through all XML files
        xml_files = [f for f in os.listdir(xml_path) if f.endswith('.xml')]
        for xml_file in xml_files:
            # Extract image ID
            img_id = os.path.splitext(xml_file)[0]

            # Parse XML file
            tree = ET.parse(os.path.join(xml_path, xml_file))
            root = tree.getroot()

            # Get object information for this img_id
            objs = img_id_to_objs[img_id]
            bbox_dict[img_id] = {
                'labels': [],
                'bboxes': [],
                'bboxes_id': []
            }
            caption_dict[img_id] = []

            # Iterate through objects to extract bounding boxes and captions
            for obj in objs:
                obj_id = obj["obj_id"]
                class_name = obj["obj_cls"] 
                points = []

                # Find corresponding bounding box in XML
                object_elem = root.findall(f".//object")[obj_id - 1]  # Corresponding object element
                points_elem = object_elem.find('.//points')
                if points_elem:
                    for point_elem in points_elem.findall('point')[:-1]:
                        x, y = map(float, point_elem.text.split(','))
                        points.append((x, y))
                # Extract coordinates of 4 points
                elif object_elem.find('robndbox'):
                    robndbox = object_elem.find('robndbox')
                    x_left_top = float(robndbox.find('x_left_top').text)
                    y_left_top = float(robndbox.find('y_left_top').text)
                    x_right_top = float(robndbox.find('x_right_top').text)
                    y_right_top = float(robndbox.find('y_right_top').text)
                    x_left_bottom = float(robndbox.find('x_left_bottom').text)
                    y_left_bottom = float(robndbox.find('y_left_bottom').text)
                    x_right_bottom = float(robndbox.find('x_right_bottom').text)
                    y_right_bottom = float(robndbox.find('y_right_bottom').text)
                    points.append((x_left_top, y_left_top))
                    points.append((x_right_top, y_right_top))
                    points.append((x_right_bottom, y_right_bottom))
                    points.append((x_left_bottom, y_left_bottom))
                else:
                    polygon = object_elem.find('polygon')
                    x1 = float(polygon.find('x1').text)
                    y1 = float(polygon.find('y1').text)
                    x2 = float(polygon.find('x2').text) 
                    y2 = float(polygon.find('y2').text) 
                    x3 = float(polygon.find('x3').text) 
                    y3 = float(polygon.find('y3').text) 
                    x4 = float(polygon.find('x4').text) 
                    y4 = float(polygon.find('y4').text) 
                    points.append((x1, y1))
                    points.append((x2, y2))
                    points.append((x3, y3))
                    points.append((x4, y4))
                # Convert coordinates to n*4*2 matrix (4 points, each with x and y)
                bbox_matrix = np.array(points).reshape(1, 4, 2)

                # Save to dictionary
                # label = class_name_to_label[class_name]
                bbox_dict[img_id]['labels'].append(class_name)
                bbox_dict[img_id]['bboxes'].append(bbox_matrix)
                bbox_dict[img_id]['bboxes_id'].append(obj_id)

                # Get caption
                caption = obj["caption"]
                caption_dict[img_id].append(caption)
                
            # bbox_dict[img_id]['labels'] = np.array(bbox_dict[img_id]['labels'])
            qbox = torch.tensor(bbox_dict[img_id]['bboxes']).float()
            rbox = qbox2rbox(qbox.reshape(qbox.shape[0], 8))
            bbox_dict[img_id]['bboxes'] = np.array(rbox)
            bbox_dict[img_id]['bboxes_id'] = np.array(bbox_dict[img_id]['bboxes_id'])
            bbox_dict[img_id]['bboxes_ignore'] = np.array([]).reshape(0, 5)  # Empty bboxes_ignore
            bbox_dict[img_id]['labels_ignore'] = np.array([])  # Empty labels_ignore
            
        self.gt_bbox = bbox_dict
        self.gt_caption = caption_dict
    
    def create_xml(self, id, dets, out_path):
        doc = Document()
        root = doc.createElement('annotation')
        doc.appendChild(root)

        source_list = {'filename': id, 'origin': 'GF2/GF3'}
        node_source = doc.createElement('source')
        for source in source_list:
            node_name = doc.createElement(source)
            node_name.appendChild(doc.createTextNode(source_list[source]))
            node_source.appendChild(node_name)
        root.appendChild(node_source)

        research_list = {'version': '1.0', 'provider': 'Beihang', 'author': 'Yin',
                        'pluginname': 'FAIR1M', 'pluginclass': 'object detection', 'time': '2022-04'}
        node_research = doc.createElement('research')
        for research in research_list:
            node_name = doc.createElement(research)
            node_name.appendChild(doc.createTextNode(research_list[research]))
            node_research.appendChild(node_name)
        root.appendChild(node_research)

        img = Image.open(os.path.join(self.img_path, id + '.tif'))
        size_list = {'width': str(img.size[0]), 'height': str(img.size[1]), 'depth': '3'}
        node_size = doc.createElement('size')
        for size in size_list:
            node_name = doc.createElement(size)
            node_name.appendChild(doc.createTextNode(size_list[size]))
            node_size.appendChild(node_name)
        root.appendChild(node_size)

        node_objects = doc.createElement('objects')
        for det_class_id, classs_det in enumerate(dets):
            label_name = self.dataset_meta['classes'][det_class_id]
            for det in classs_det:
                node_object = doc.createElement('object')
                object_fore_list = {'coordinate': 'pixel', 'type': 'rectangle', 'description': 'None'}
                for object_fore in object_fore_list:
                    node_name = doc.createElement(object_fore)
                    node_name.appendChild(doc.createTextNode(object_fore_list[object_fore]))
                    node_object.appendChild(node_name)

                node_possible_result = doc.createElement('possibleresult')
                node_name = doc.createElement('name')
                node_name.appendChild(doc.createTextNode(label_name))
                node_possible_result.appendChild(node_name)
                
                node_probability = doc.createElement('probability')
                node_probability.appendChild(doc.createTextNode(str(det[-1])))
                node_possible_result.appendChild(node_probability)
                
                node_object.appendChild(node_possible_result)

                node_points = doc.createElement('points')
                det = det[:8].reshape(4, 2)
                for j in range(4):
                    node_point = doc.createElement('point')
                    text = '{},{}'.format(det[j][0], det[j][1])
                    node_point.appendChild(doc.createTextNode(text))
                    node_points.appendChild(node_point)
                node_point = doc.createElement('point')
                text = '{},{}'.format(det[0][0], det[0][1])
                node_point.appendChild(doc.createTextNode(text))
                node_points.appendChild(node_point)
                node_object.appendChild(node_points)

                node_objects.appendChild(node_object)
        root.appendChild(node_objects)

        # Start writing xml document
        filename = os.path.join(out_path, id + '.xml')
        fp = open(filename, 'w', encoding='utf-8')
        doc.writexml(fp, indent='', addindent='\t', newl='\n', encoding="utf-8")
        fp.close()
    

    def get_cls_results(self, 
                        det_results, 
                        det_captions,
                        annotations, 
                        anno_captions,
                        class_id, 
                        box_type):
        """Get det results and gt information of a certain class.

        Args:
            det_results (list[list]): Same as `eval_map()`.
            det_captions (list[list]): Same as `eval_map()`.
            annotations (list[dict]): Same as `eval_map()`.
            class_id (int): ID of a specific class.
            box_type (str): Box type. If the QuadriBoxes is used, you need to
                specify 'qbox'. Defaults to 'rbox'.

        Returns:
            tuple[list[np.ndarray]]: detected bboxes, gt bboxes, ignored gt bboxes
        """
        cls_dets = [img_res[class_id] for img_res in det_results]
        cls_captions = [img_res[class_id] for img_res in det_captions]

        cls_gts = []
        cls_ids = []
        cls_gts_ignore = []
        cls_gt_captions = []
        for ann, anno_caption in zip(annotations, anno_captions):
            if len(ann['bboxes']) != 0:
                gt_inds = ann['labels'] == class_id
                cls_gts.append(ann['bboxes'][gt_inds, :])
                ignore_inds = ann['labels_ignore'] == class_id
                cls_gts_ignore.append(ann['bboxes_ignore'][ignore_inds, :])
                
                cls_gt_captions.append(np.array(anno_caption)[gt_inds].tolist())
                cls_ids.append(ann['bboxes_id'][gt_inds])
            else:
                if box_type == 'rbox':
                    cls_gts.append(torch.zeros((0, 5), dtype=torch.float64))
                    cls_gts_ignore.append(torch.zeros((0, 5), dtype=torch.float64))
                    cls_gt_captions.append([])
                    cls_ids.append([])
                elif box_type == 'qbox':
                    cls_gts.append(torch.zeros((0, 8), dtype=torch.float64))
                    cls_gts_ignore.append(torch.zeros((0, 8), dtype=torch.float64))
                else:
                    raise NotImplementedError

        return cls_dets, cls_captions, cls_gts, cls_gt_captions, cls_ids, cls_gts_ignore
    
    def solve_task(self, 
                   id_list,
                   dets_list,
                   dets_list_match,
                   dets_list_captions,
                   outfile_prefix: str, 
                   nproc: int = 4, 
                   iou_thr: int = 0.5):
        """
        Args:
            id_list: list of image ids
            dets_list: list of detected bboxes
            dets_list_match: list of detected bboxes after matching
            dets_list_captions: list of object captions
            outfile_prefix: prefix of output file
            nproc: number of processes
            iou_thr: iou threshold

        Returns:
            tuple[list[np.ndarray]]: detected bboxes, gt bboxes, ignored gt bboxes
        """
        
        os.makedirs(outfile_prefix, exist_ok=True)   
        if "Task1" in self.task:
            if osp.exists(osp.join(outfile_prefix, 'Task1')):
                raise ValueError(f'The outfile_prefix should be a non-exist path, '
                                f'but {outfile_prefix} is existing. '
                                f'Please delete it firstly.')
            os.makedirs(osp.join(outfile_prefix, 'Task1'))

            files = [
                osp.join(outfile_prefix, 'Task1', img_id + '.xml')
                for img_id in id_list
            ]
            for img_id, dets in zip(id_list, dets_list):
                self.create_xml(img_id, dets, osp.join(outfile_prefix, 'Task1'))

            zip_path = osp.join(outfile_prefix, 'Task1.zip')
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as t:
                for f in files:
                    t.write(f, osp.join('test', osp.split(f)[-1]))

        if "Task2" in self.task:
            
            num_imgs = len(id_list)
            class_name_to_label = {class_name: class_id for class_id, class_name in enumerate(self.dataset_meta['classes'])}
            annotations = [self.gt_bbox[i] for i in id_list]
            anno_captions = [self.gt_caption[i] for i in id_list]
            for ann in annotations:
                ann["labels"] = np.array([class_name_to_label[i] for i in ann["labels"]])
            pool = get_context('spawn').Pool(nproc)
            cp_results = [{
                "img_id": i,
                "objs": []
                } for i in id_list]
            for i in range(len(self.dataset_meta["classes"])):
                # get gt and det bboxes of this class
                cls_dets, cls_captions, cls_gts, cls_gt_captions, cls_ids, cls_gts_ignore = self.get_cls_results(
                    dets_list_match, dets_list_captions, annotations, anno_captions, i, self.predict_box_type)
                matched = pool.starmap(
                    self.match_captions,
                    zip(cls_dets, cls_captions, cls_gts, cls_gt_captions, cls_ids, cls_gts_ignore,
                        [iou_thr for _ in range(num_imgs)],
                        [self.predict_box_type for _ in range(num_imgs)],
                        [i for i in id_list],
                        [self.dataset_meta["classes"][i] for _ in range(num_imgs)]))
                for j in range(len(id_list)):
                    cp_results[j]["objs"] += matched[j]["objs"]
            
            with open(osp.join(outfile_prefix, 'Task2.json'), 'w') as f:
                json.dump(cp_results, f, indent=4)
                
    
    
    def patch_results(self, results: Sequence[dict],
                      outfile_prefix: str, 
                      nproc: int = 4, iou_thr: int = 0.5) -> str:
        """Merge patches' predictions into full image's results and generate a
        zip file for FAIR1M online evaluation.

        You can submit it at:
        https://www.gaofen-challenge.com/benchmark

        Args:
            results (Sequence[dict]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the zip files. If the
                prefix is "somepath/xxx", the zip files will be named
                "somepath/xxx/xxx.zip".
        """
        
        collector = defaultdict(list)
        captioner = defaultdict(list)

        for idx, result in enumerate(results):
            oriname = result.get('img_id', idx)
            labels = result['labels']
            bboxes = result['bboxes']
            scores = result['scores']
            if 'captions' in result:
                captions = result['captions']
                captioner[oriname].extend(captions)
            ori_bboxes = bboxes.copy()
            label_dets = np.concatenate(
                [labels[:, np.newaxis], ori_bboxes, scores[:, np.newaxis]],
                axis=1)
            collector[oriname].append(label_dets)

        id_list, dets_list, dets_list_match, dets_list_captions = [], [], [], []
        for oriname, label_dets_list in collector.items():
            big_img_results = []
            big_img_results_match = []
            big_img_results_captions = []
            
            label_dets = np.concatenate(label_dets_list, axis=0)
            labels, dets = label_dets[:, 0], label_dets[:, 1:]
            
            if oriname in captioner:
                captions = captioner[oriname]
            for i in range(len(self.dataset_meta['classes'])):
                if len(dets[labels == i]) == 0:
                    big_img_results.append(dets[labels == i])
                    big_img_results_match.append(dets[labels == i])
                    if oriname in captioner:
                        big_img_results_captions.append(np.array(captions)[labels == i].tolist())
                else:
                    try:
                        cls_dets = torch.from_numpy(dets[labels == i]).cuda()
                    except:  # noqa: E722
                        cls_dets = torch.from_numpy(dets[labels == i])
                    big_img_results_match.append(cls_dets.cpu().numpy())
                    big_img_results.append(torch.cat([rbox2qbox(cls_dets[:, :5]), cls_dets[:, -2:]], dim=-1).cpu().numpy())
                    
                    if oriname in captioner:
                        cls_captions = np.array(captions)[labels == i].tolist()
                        if isinstance(cls_captions, str):
                            cls_captions = [cls_captions]
                        big_img_results_captions.append(cls_captions)
                        
            id_list.append(oriname)
            dets_list.append(big_img_results)
            dets_list_match.append(big_img_results_match)
            dets_list_captions.append(big_img_results_captions)
            
        self.solve_task(id_list, dets_list, dets_list_match, dets_list_captions, outfile_prefix, nproc, iou_thr)
    
    def merge_results(self, results: Sequence[dict],
                      outfile_prefix: str, 
                      nproc: int = 4, iou_thr: int = 0.5) -> str:
        """Merge patches' predictions into full image's results and generate a
        zip file for FAIR1M online evaluation.

        You can submit it at:
        https://www.gaofen-challenge.com/benchmark

        Args:
            results (Sequence[dict]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the zip files. If the
                prefix is "somepath/xxx", the zip files will be named
                "somepath/xxx/xxx.zip".
        """
        
        collector = defaultdict(list)
        captioner = defaultdict(list)

        for idx, result in enumerate(results):
            img_id = result.get('img_id', idx)
            splitname = img_id.split('__')
            oriname = splitname[0]
            pattern1 = re.compile(r'__\d+___\d+')
            x_y = re.findall(pattern1, img_id)
            x_y_2 = re.findall(r'\d+', x_y[0])
            x, y = int(x_y_2[0]), int(x_y_2[1])
            labels = result['labels']
            bboxes = result['bboxes']
            scores = result['scores']
            if 'captions' in result:
                captions = result['captions']
                captioner[oriname].extend(captions)
            ori_bboxes = bboxes.copy()
            if self.predict_box_type == 'rbox':
                ori_bboxes[..., :2] = ori_bboxes[..., :2] + np.array(
                    [x, y], dtype=np.float32)
            elif self.predict_box_type == 'qbox':
                ori_bboxes[..., :] = ori_bboxes[..., :] + np.array(
                    [x, y, x, y, x, y, x, y], dtype=np.float32)
            else:
                raise NotImplementedError
            label_dets = np.concatenate(
                [labels[:, np.newaxis], ori_bboxes, scores[:, np.newaxis]],
                axis=1)
            collector[oriname].append(label_dets)

        id_list, dets_list, dets_list_match, dets_list_captions = [], [], [], []
        for oriname, label_dets_list in collector.items():
            big_img_results = []
            big_img_results_match = []
            big_img_results_captions = []
            
            label_dets = np.concatenate(label_dets_list, axis=0)
            labels, dets = label_dets[:, 0], label_dets[:, 1:]
            
            if oriname in captioner:
                captions = captioner[oriname]
            for i in range(len(self.dataset_meta['classes'])):
                if len(dets[labels == i]) == 0:
                    big_img_results.append(dets[labels == i])
                    big_img_results_match.append(dets[labels == i])
                    if oriname in captioner:
                        big_img_results_captions.append(np.array(captions)[labels == i].tolist())
                else:
                    try:
                        cls_dets = torch.from_numpy(dets[labels == i]).cuda()
                    except:  # noqa: E722
                        cls_dets = torch.from_numpy(dets[labels == i])
                    if self.predict_box_type == 'rbox':
                        nms_dets, keep_ids = nms_rotated(cls_dets[:, :5],
                                                  cls_dets[:,
                                                           -1], self.iou_thr)
                    elif self.predict_box_type == 'qbox':
                        nms_dets, keep_ids = nms_quadri(cls_dets[:, :8],
                                                 cls_dets[:, -1], self.iou_thr)
                    big_img_results_match.append(nms_dets.cpu().numpy())
                    big_img_results.append(torch.cat([rbox2qbox(nms_dets[:, :5]), nms_dets[:, -2:]], dim=-1).cpu().numpy())
                    
                    if oriname in captioner:
                        nms_captions = np.array(captions)[labels == i][keep_ids.cpu()].tolist()
                        if isinstance(nms_captions, str):
                            nms_captions = [nms_captions]
                        big_img_results_captions.append(nms_captions)
                        
            id_list.append(oriname)
            dets_list.append(big_img_results)
            dets_list_match.append(big_img_results_match)
            dets_list_captions.append(big_img_results_captions)
            
        self.solve_task(id_list, dets_list, dets_list_match, dets_list_captions, outfile_prefix, nproc, iou_thr)

    def results2json(self, results: Sequence[dict],
                     outfile_prefix: str) -> dict:
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (Sequence[dict]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict: Possible keys are "bbox", "segm", "proposal", and
            values are corresponding filenames.
        """
        bbox_json_results = []
        for idx, result in enumerate(results):
            image_id = result.get('img_id', idx)
            labels = result['labels']
            bboxes = result['bboxes']
            scores = result['scores']
            # bbox results
            for i, label in enumerate(labels):
                data = dict()
                data['image_id'] = image_id
                data['bbox'] = bboxes[i].tolist()
                data['score'] = float(scores[i])
                data['category_id'] = int(label)
                bbox_json_results.append(data)

        result_files = dict()
        result_files['bbox'] = f'{outfile_prefix}.bbox.json'
        dump(bbox_json_results, result_files['bbox'])

        return result_files

    def process(self, data_batch: Sequence[dict],
                data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        for data_sample in data_samples:
            gt = copy.deepcopy(data_sample)
            gt_instances = gt['gt_instances']
            gt_ignore_instances = gt['ignored_instances']
            if gt_instances == {}:
                ann = dict()
            else:
                ann = dict(
                    labels=gt_instances['labels'].cpu().numpy(),
                    bboxes=gt_instances['bboxes'].cpu().numpy(),
                    bboxes_ignore=gt_ignore_instances['bboxes'].cpu().numpy(),
                    labels_ignore=gt_ignore_instances['labels'].cpu().numpy())
            result = dict()
            pred = data_sample['pred_instances']
            result['img_id'] = data_sample['img_id']
            result['bboxes'] = pred['bboxes'].cpu().numpy()
            result['scores'] = pred['scores'].cpu().numpy()
            result['labels'] = pred['labels'].cpu().numpy()
            if "captions" in pred:
                result['captions'] = pred['captions']

            result['pred_bbox_scores'] = []
            for label in range(len(self.dataset_meta['classes'])):
                index = np.where(result['labels'] == label)[0]
                pred_bbox_scores = np.hstack([
                    result['bboxes'][index], result['scores'][index].reshape(
                        (-1, 1))
                ])
                result['pred_bbox_scores'].append(pred_bbox_scores)

            self.results.append((ann, result))

    def compute_metrics(self, results: list) -> dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.
        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        gts, preds = zip(*results)

        tmp_dir = None
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            outfile_prefix = self.outfile_prefix
            
        eval_results = OrderedDict()
        if self.merge_patches:
            # convert predictions to txt format and dump to zip file
            self.merge_results(preds, outfile_prefix)
            # logger.info(f'The submission file save at {zip_path}')
            return eval_results
        else:
            _ = self.results2json(preds, outfile_prefix)
            if "Task2" in self.task:
                self.patch_results(preds, outfile_prefix)
            if self.format_only:
                logger.info('results are saved in '
                            f'{osp.dirname(outfile_prefix)}')
                return eval_results

        if self.metric == 'mAP':
            assert isinstance(self.iou_thrs, list)
            dataset_name = self.dataset_meta['classes']
            dets = [pred['pred_bbox_scores'] for pred in preds]

            mean_aps = []
            for iou_thr in self.iou_thrs:
                logger.info(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                mean_ap, _ = eval_rbbox_map(
                    dets,
                    gts,
                    scale_ranges=self.scale_ranges,
                    iou_thr=iou_thr,
                    use_07_metric=self.use_07_metric,
                    box_type=self.predict_box_type,
                    dataset=dataset_name,
                    logger=logger)
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
            eval_results.move_to_end('mAP', last=False)
        else:
            raise NotImplementedError
        return eval_results


    def match_captions(self,
                       det_bboxes,
                       det_captions,
                       gt_bboxes,
                       gt_captions,
                       gt_bboxes_id,
                       gt_bboxes_ignore=None,
                       iou_thr=0.5,
                       box_type='rbox',
                       img_id=None,
                       obj_cls=None,
                       area_ranges=None):
        """Check if detected bboxes are true positive or false positive.

        Args:
            det_bboxes (ndarray): Detected bboxes of this image, of shape (m, 6).
            det_captions (ndarray): Captions of detected bboxes of this image, of shape (m, 1).
            gt_bboxes (ndarray): GT bboxes of this image, of shape (n, 5).
            gt_captions (ndarray): Captions of GT bboxes of this image, of shape (n, 1).
            gt_bboxes_ignore (ndarray): Ignored gt bboxes of this image,
                of shape (k, 5). Defaults to None
            iou_thr (float): IoU threshold to be considered as matched.
                Defaults to 0.5.
            box_type (str): Box type. If the QuadriBoxes is used, you need to
                specify 'qbox'. Defaults to 'rbox'.
            img_id (int): Image id.
            obj_cls (int): Object class id.
            area_ranges (list[tuple], optional): Range of bbox areas to be
                evaluated, in the format [(min1, max1), (min2, max2), ...].
                Defaults to None.

        Returns:
            tuple[np.ndarray]: (tp, fp) whose elements are 0 and 1. The shape of
            each array is (num_scales, m).
        """
        # an indicator of ignored gts
        det_bboxes = np.array(det_bboxes)
        gt_ignore_inds = np.concatenate(
            (np.zeros(gt_bboxes.shape[0],
                    dtype=bool), np.ones(gt_bboxes_ignore.shape[0], dtype=bool)))
        # stack gt_bboxes and gt_bboxes_ignore for convenience
        gt_bboxes = np.vstack((gt_bboxes, gt_bboxes_ignore))

        num_gts = gt_bboxes.shape[0]
        if area_ranges is None:
            area_ranges = [(None, None)]
        # tp and fp are of shape (num_scales, num_gts), each row is tp or fp of
        # a certain scale

        # if there is no gt bboxes in this image, then all det bboxes
        # within area range are false positives
        match_captions = {
            "img_id": img_id,
            "objs": [{
                "obj_id": int(obj_id),
                "obj_cls": obj_cls,
                "gt_caption": caption
                } for caption, obj_id in zip(gt_captions, gt_bboxes_id)]
            }
        
        if gt_bboxes.shape[0] == 0:
            return match_captions

        if box_type == 'rbox':
            ious = box_iou_rotated(
                torch.from_numpy(det_bboxes).float(),
                torch.from_numpy(gt_bboxes).float()).numpy()
        elif box_type == 'qbox':
            ious = box_iou_quadri(
                torch.from_numpy(det_bboxes).float(),
                torch.from_numpy(gt_bboxes).float()).numpy()
        else:
            raise NotImplementedError
        # for each det, the max iou with all gts
        ious_max = ious.max(axis=1)
        # for each det, which gt overlaps most with it
        ious_argmax = ious.argmax(axis=1)
        # sort all dets in descending order by scores
        sort_inds = np.argsort(-det_bboxes[:, -1])
        
        for k, (min_area, max_area) in enumerate(area_ranges):
            gt_covered = np.zeros(num_gts, dtype=bool)
            # if no area range is specified, gt_area_ignore is all False
            if min_area is None:
                gt_area_ignore = np.zeros_like(gt_ignore_inds, dtype=bool)
            else:
                raise NotImplementedError
            for i in sort_inds:
                if ious_max[i] >= iou_thr:
                    matched_gt = ious_argmax[i]
                    if not (gt_ignore_inds[matched_gt]
                            or gt_area_ignore[matched_gt]):
                        if not gt_covered[matched_gt]:
                            gt_covered[matched_gt] = True
                            pred_caption = self.solve_captions(det_captions[i])
                            if 'object' in pred_caption:
                                del pred_caption['object']
                            match_captions['objs'][matched_gt]['pred_caption'] = pred_caption

        return match_captions
    
    def solve_captions(self, text):
        
        pattern = r'(\w+(?:-\w+)*)\s+is\s+([^,\.]+)'  # Matches patterns like "a is b" where b can contain hyphens
        matches = re.findall(pattern, text)

        # Build dictionary from matches
        result = {a: b for a, b in matches}
        return result
