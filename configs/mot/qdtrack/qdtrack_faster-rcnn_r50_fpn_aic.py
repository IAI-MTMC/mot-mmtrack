_base_ = [
    './qdtrack_faster-rcnn_r50_fpn_4e_base.py',
    '../../_base_/datasets/aicity_challenge.py',
]

custom_imports = dict(
    imports=['mmtrack.models.trackers.quasi_dense_tracker_plus'])

img_scale = (1088, 1088)

model = dict(
    data_preprocessor=dict(bgr_to_rgb=False),
    track_head=dict(
        type='QuasiDenseTrackHead',
        roi_extractor=dict(
            _scope_='mmdet',
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256),
        embed_head=dict(
            type='QuasiDenseEmbedHead',
            num_convs=4,
            num_fcs=2,
            embed_channels=256,
            norm_cfg=dict(type='GN', num_groups=32),
            loss_track=dict(type='MultiPosCrossEntropyLoss', loss_weight=0.5),
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
        type='QuasiDenseTrackerPlus',
        init_score_thr=0.9,
        obj_score_thr=0.5,
        match_score_thr=0.5,
        memo_tracklet_frames=90,
        memo_backdrop_frames=2,
        memo_momentum=0.8,
        det_momentum_thr=0.8,
        nms_conf_thr=0.5,
        nms_backdrop_iou_thr=0.3,
        nms_class_iou_thr=0.7,
        with_cats=True,
        match_metric='bisoftmax'))

# data pipeline
train_pipeline = [
    dict(
        type='TransformBroadcaster',
        share_random_params=True,
        transforms=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadTrackAnnotations', with_instance_id=True),
            dict(
                type='mmdet.RandomResize',
                scale=img_scale,
                ratio_range=(0.8, 1.2),
                keep_ratio=True,
                clip_object_border=False),
            dict(type='mmdet.PhotoMetricDistortion'),
            dict(type='mmdet.RandomFlip', prob=0.5),
        ]),
    dict(type='PackTrackInputs', ref_prefix='ref'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadTrackAnnotations', with_instance_id=True),
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
    dict(type='PackTrackInputs', pack_single_img=True),
]

train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        pipeline=train_pipeline,
        ann_file='annotations/train_cocoformat_subset_0.5_consec.json',))
val_dataloader = dict(
    dataset=dict(
        pipeline=test_pipeline,
        ann_file='annotations/validation_cocoformat_subset_0.1_consec.json',
        metainfo=dict(CLASSES=('person', ), classes=('person', ))))
test_dataloader = val_dataloader

# evaluator
val_evaluator = [
    dict(type='CocoVideoMetric', metric=['bbox'], classwise=True),
    dict(type='MOTChallengeMetrics', metric=['HOTA', 'CLEAR', 'Identity'])
]
test_evaluator = val_evaluator

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2))
# learning policy
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 1000,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        type='mmdet.MultiStepLR',
        begin=0,
        end=5,
        by_epoch=True,
        milestones=[2, 4])
]

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=5, val_interval=2)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

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
