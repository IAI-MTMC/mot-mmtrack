_base_ = ["../_base_/default_runtime.py", "../_base_/datasets/aicity_challenge_det.py"]

img_scale = (640, 640)
batch_size = 2

model = dict(
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        pad_size_divisor=32,
        batch_augments=[
            dict(
                type="BatchSyncRandomResize",
                random_size_range=(320, 640),
                size_divisor=32,
                interval=10,
            )
        ],
    ),
    _scope_="mmdet",
    type="YOLOX",
    backbone=dict(type="CSPDarknet", deepen_factor=1.33, widen_factor=1.25),
    neck=dict(
        type="YOLOXPAFPN",
        in_channels=[320, 640, 1280],
        out_channels=320,
        num_csp_blocks=4,
    ),
    bbox_head=dict(type="YOLOXHead", num_classes=1, in_channels=320, feat_channels=320),
    train_cfg=dict(assigner=dict(type="SimOTAAssigner", center_radius=2.5)),
    test_cfg=dict(score_thr=0.01, nms=dict(type="nms", iou_threshold=0.7)),
    init_cfg=dict(
        type="Pretrained",
        checkpoint=  # noqa: E251
        # 'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_x_8x8_300e_coco/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth'  # noqa: E501
        "checkpoints/yolox_x_crowdhuman_mot17-private-half.pth",
    ),
)

train_pipeline = [
    dict(type="Mosaic", img_scale=img_scale, pad_val=114.0, bbox_clip_border=False),
    dict(
        type="RandomAffine",
        scaling_ratio_range=(0.1, 2),
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        bbox_clip_border=False,
    ),
    dict(
        type="MixUp",
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        pad_val=114.0,
        bbox_clip_border=False,
    ),
    dict(type="YOLOXHSVRandomAug"),
    dict(type="RandomFlip", prob=0.5),
    dict(type="Resize", scale=img_scale, keep_ratio=True, clip_object_border=False),
    dict(type="Pad", size_divisor=32, pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type="FilterAnnotations", min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type="PackDetInputs"),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=img_scale, keep_ratio=True),
    dict(type="Pad", size_divisor=32, pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]

train_dataloader = dict(
    _delete_=True,
    batch_size=batch_size,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="MultiImageMixDataset",
        _scope_="mmdet",
        dataset=dict(
            type="CocoDataset",
            data_root=_base_.data_root, # type: ignore
            ann_file="annotations/train_cocoformat_subset_8000.json",
            data_prefix=dict(img="train/"),
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            metainfo=dict(CLASSES=("person",)),
            pipeline=[
                dict(type="LoadImageFromFile"),
                dict(type="LoadAnnotations"),
            ],
        ),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

# optimizer
lr = 0.001 * batch_size
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="SGD", lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True),
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0),
)

# some hyper parameters
# training settings
total_epochs = 15
num_last_epochs = 5
resume_from = None
interval = 5

train_cfg = dict(
    type="EpochBasedTrainLoop", max_epochs=total_epochs, val_interval=interval
)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
# learning policy
param_scheduler = [
    dict(
        # use quadratic formula to warm up 1 epochs
        # and lr is updated by iteration
        type="mmdet.QuadraticWarmupLR",
        by_epoch=True,
        begin=0,
        end=1,
        convert_to_iter_based=True,
    ),
    dict(
        # use cosine lr from 1 to 70 epoch
        type="mmdet.CosineAnnealingLR",
        eta_min=lr * 0.05,
        begin=1,
        T_max=total_epochs - num_last_epochs,
        end=total_epochs - num_last_epochs,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(
        # use fixed lr during last 10 epochs
        type="mmdet.ConstantLR",
        by_epoch=True,
        factor=1,
        begin=total_epochs - num_last_epochs,
        end=total_epochs,
    ),
]

custom_hooks = [
    dict(type="YOLOXModeSwitchHook", num_last_epochs=num_last_epochs, priority=48),
    dict(type="mmdet.SyncNormHook", priority=48),
    dict(
        type="mmdet.EMAHook",
        ema_type="mmdet.ExpMomentumEMA",
        momentum=0.0001,
        update_buffers=True,
        priority=49,
    ),
]
default_hooks = dict(checkpoint=dict(interval=interval))
