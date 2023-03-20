_base_ = [
    '../../_base_/models/yolox_x_8x8.py', '../../_base_/default_runtime.py',
    '../../_base_/datasets/aicity_challenge.py'
]

img_scale = (640, 640)
strides = [8, 16, 32]

model = dict(
    type='QDTrackSSTG',
    data_preprocessor=dict(
        _delete_=True, type='TrackDataPreprocessor', pad_size_divisor=32),
    freeze_detector=True,
    detector=dict(
        _scope_='mmdet',
        bbox_head=dict(num_classes=1),
        train_cfg=dict(
            score_thr=0, nms=dict(type='nms', iou_threshold=0.7,
                                  max_num=1000)),
        test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.7)),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'work_dirs/yolox_x_aicity/epoch_10.pth'  # noqa: E501
        )),
    track_head=dict(
        type='QuasiDenseTrackHead',
        roi_extractor=dict(
            _scope_='mmdet',
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=320,
            featmap_strides=strides),
        embed_head=dict(
            type='QuasiDenseEmbedHead',
            num_convs=4,
            num_fcs=1,
            in_channels=320,
            embed_channels=256,
            norm_cfg=dict(type='GN', num_groups=32),
            loss_track=dict(type='MultiPosCrossEntropyLoss', loss_weight=0.25),
            loss_track_aux=dict(
                type='L2Loss',
                neg_pos_ub=3,
                pos_margin=0,
                neg_margin=0.1,
                hard_mining=True,
                loss_weight=1.0)),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0),
        train_cfg=dict(
            assigner=dict(
                _scope_='mmdet',
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                _scope_='mmdet',
                type='CombinedSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=3,
                add_gt_as_proposals=True,
                pos_sampler=dict(type='InstanceBalancedPosSampler'),
                neg_sampler=dict(type='RandomSampler')))),
    tracker=dict(
        type='QuasiDenseTracker',
        init_score_thr=0.9,
        obj_score_thr=0.5,
        match_score_thr=0.5,
        memo_tracklet_frames=120,
        memo_backdrop_frames=1,
        memo_momentum=0.8,
        nms_conf_thr=0.5,
        nms_backdrop_iou_thr=0.3,
        nms_class_iou_thr=0.7,
        with_cats=False,
        match_metric='bisoftmax'),
    init_cfg=dict(
        type='Pretrained', checkpoint='work_dirs/qdtrack_yolox_x/epoch_5.pth'))

# data pipeline
train_pipeline = [
    dict(
        type='TransformBroadcaster',
        share_random_params=True,
        transforms=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadTrackAnnotations', with_instance_id=True),
            dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
            dict(type='mmdet.PhotoMetricDistortion')
        ]),
    dict(
        type='TransformBroadcaster',
        share_random_params=False,
        transforms=[
            dict(
                type='mmdet.RandomCrop',
                crop_size=img_scale,
                bbox_clip_border=False)
        ]),
    dict(
        type='TransformBroadcaster',
        share_random_params=True,
        transforms=[
            dict(type='mmdet.RandomFlip', prob=0.5),
        ]),
    dict(type='PackTrackInputs', ref_prefix='ref')
]

# dataset settings
train_dataloader = dict(
    dataset=dict(
        ann_file='annotations/train_cocoformat_subset_0.1_consec.json',
        pipeline=train_pipeline))

val_dataloader = dict(
    dataset=dict(
        ann_file='annotations/validation_cocoformat_subset_0.2_consec.json'))
test_dataloader = val_dataloader

# optimizer
lr = 0.001
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=lr, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2))

# some hyper parameters
# training settings
total_epochs = 5
resume_from = None
interval = 1

# learning policy
param_scheduler = [
    dict(
        type='mmdet.MultiStepLR',
        begin=0,
        end=total_epochs,
        by_epoch=True,
        milestones=[3])
]

# runtime settings
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=total_epochs, val_interval=interval)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(checkpoint=dict(interval=1))

# vis_backends = [
#     dict(
#         type="WandbVisBackend",
#         init_kwargs=dict(
#             entity="iai-mtmc",
#             project="qdtrack",
#         ),
#     ),
# ]
# visualizer = dict(
#     type="TrackLocalVisualizer", vis_backends=vis_backends, name="visualizer"
# )
