# dataset settings
dataset_type = 'mmdet.CocoDataset'
data_root = '../datasets/AIC23_Track1_MTMC_Tracking/'

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomResize',
        scale=(640, 640),
        ratio_range=(0.8, 1.2),
        keep_ratio=True,
        clip_object_border=False),
    dict(type='PhotoMetricDistortion'),
    dict(type='RandomCrop', crop_size=(640, 640), bbox_clip_border=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='mmdet.AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        _scope_='mmdet',
        ann_file='annotations/train_cocoformat_subset_8000.json',
        data_prefix=dict(img='train/'),
        metainfo=dict(classes=('person', )),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        _scope_='mmdet',
        ann_file='annotations/validation_cocoformat_subset_6000.json',
        data_prefix=dict(img='validation/'),
        metainfo=dict(classes=('person', )),
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='mmdet.CocoMetric',
    ann_file=data_root + 'annotations/validation_cocoformat_subset_6000.json',
    metric='bbox',
    format_only=False)
test_evaluator = val_evaluator
