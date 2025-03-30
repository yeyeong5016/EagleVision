_base_ = [
    './_base_/default_runtime.py', './_base_/schedule_1x.py',
    './_base_/fair1m_Task1.py'
]

angle_version = 'le90'
pretrained = "/mnt/data1/jianghx/ckpts/InternVL2-8B"
num_image_token = 64
template = "internlm2-chat"
backend_args = None

model = dict(
    type='EagleVisionPro',
    data_preprocessor=dict(
        type='EVDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        boxtype2tensor=False),
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='OrientedRPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='mmdet.AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64],
            use_box_type=True),
        bbox_coder=dict(
            type='MidpointOffsetCoder',
            angle_version=angle_version,
            target_means=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0, 0.5, 0.5]),
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(
            type='mmdet.SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0)),
    roi_head=dict(
        type='EagleVisionProROIHead',
        bbox_roi_extractor=dict(
            type='RotatedSingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlignRotated',
                out_size=5,
                sample_num=2,
                clockwise=True),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='EagleVisionProHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=5,
            num_classes=37,
            reg_predictor_cfg=dict(type='mmdet.Linear'),
            cls_predictor_cfg=dict(type='mmdet.Linear'),
            bbox_coder=dict(
                type='DeltaXYWHTRBBoxCoder',
                angle_version=angle_version,
                norm_factor=None,
                edge_swap=True,
                proj_xy=True,
                target_means=(.0, .0, .0, .0, .0),
                target_stds=(0.1, 0.1, 0.2, 0.2, 0.1)),
            reg_class_agnostic=True,
            loss_cls=dict(
                type='mmdet.CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='mmdet.SmoothL1Loss', beta=1.0, loss_weight=1.0),
            pretrained=pretrained,
            template=template,
            patch_size=5,
            caption_num=4,
            caption_type="orthogonal",
            loss_caption_weight=0.1,
            loss_attr_weight=1.0,
            caption_dim=num_image_token
            )),


    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='mmdet.MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1,
                iou_calculator=dict(type='RBbox2HBboxOverlaps2D')),
            sampler=dict(
                type='mmdet.RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.8),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='mmdet.MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                iou_calculator=dict(type='RBboxOverlaps2D'),
                ignore_iof_thr=-1),
            sampler=dict(
                type='mmdet.RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.8),
            min_bbox_size=0),
        rcnn=dict(
            nms_pre=2000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms_rotated', iou_threshold=0.1),
            max_per_img=2000)))

strategy = dict(
    type='DeepSpeedStrategy',
    zero_optimization=dict(
        stage=2,
        allgather_partitions=True,
        reduce_scatter=True,
        allgather_bucket_size=5000000,
        reduce_bucket_size=5000000,
        overlap_comm=True,
        contiguous_gradients=True,
        round_robin_gradients=True,
        offload_optimizer=dict(
            device="cpu",
            pin_memory=True
        )
        )
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 3,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

optim_wrapper = dict(
    type='DeepSpeedOptimWrapper',
    optimizer=dict(type='DeepSpeedCPUAdam', lr=6e-5))

train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='EVLoadAnnotations', with_bbox=True, box_type='qbox', with_caption=True),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(
        type='RandomRotate',
        prob=0.5,
        angle_range=180,
        rect_obj_labels=None),
    dict(
        type='mmdet.Pad', size=(1024, 1024),
        pad_val=dict(img=(114, 114, 114))),
    dict(type='EVPackDetInputs')
]

val_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    # avoid bboxes being resized
    dict(type='EVLoadAnnotations', with_bbox=True, box_type='qbox', with_caption=True),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.Pad', size=(1024, 1024),
        pad_val=dict(img=(114, 114, 114))),
    dict(
        type='EVPackDetInputs')
]

# batch_size = (4 GPUs) x (2 samples per GPU) = 8
train_dataloader = dict(batch_size=2, 
                        num_workers=2,
                        dataset=dict(
                            num_image_token=num_image_token,
                            template_name=template,
                            pretrained=pretrained,
                            pipeline=train_pipeline),
                        collate_fn=dict(type='ev_collate'))


val_dataloader = dict(batch_size=1,
                        num_workers=2,
                        dataset=dict(
                            num_image_token=num_image_token,
                            template_name=template,
                            pretrained=pretrained,
                            pipeline=val_pipeline),
                        collate_fn=dict(type='ev_collate'))


test_evaluator_task1 = dict(
    type='EVBench',
    format_only=True,
    merge_patches=True,
    img_path='/mnt/data1/jianghx/data/FAIR1M/test/images',
    task='Task1',
    outfile_prefix='./work_dirs/EagleVision_7B-fair1m/test')

test_evaluator_task2 = dict(
    type='EVBench',
    format_only=True,
    merge_patches=True,
    xml_path='/mnt/data1/jianghx/data/FAIR1M/val/labelXml',
    caption_gt_path='/mnt/data1/jianghx/data/FAIR1M/EVAttrs-FAIR1M-val.json',
    task='Task2',
    outfile_prefix='./work_dirs/EagleVision_7B-fair1m/val')