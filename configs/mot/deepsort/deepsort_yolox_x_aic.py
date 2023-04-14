_base_ = [
    '../../_base_/models/yolox_x_8x8.py',
    '../../_base_/datasets/aicity_challenge.py',
    '../../_base_/default_runtime.py'
]

img_scale = (1088, 1088)

model = dict(
    type='DeepSORT',
    detector=dict(
        bbox_head=dict(num_classes=1),
        test_cfg=dict(score_thr=0.1, nms=dict(type='nms', iou_threshold=0.7)),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='checkpoints/yolox_x_crowdhuman_mot17-private-half.pth')
    ),
    motion=dict(type='KalmanFilter', center_only=False),
    reid=dict(
        type='BaseReID',
        data_preprocessor=None,
        backbone=dict(
            type='mmcls.ResNet',
            depth=50,
            num_stages=4,
            out_indices=(3, ),
            style='pytorch'),
        neck=dict(type='GlobalAveragePooling', kernel_size=(8, 4), stride=1),
        head=dict(
            type='LinearReIDHead',
            num_fcs=1,
            in_channels=2048,
            fc_channels=1024,
            out_channels=128,
            num_classes=380,
            loss_cls=dict(type='mmcls.CrossEntropyLoss', loss_weight=1.0),
            loss_triplet=dict(type='TripletLoss', margin=0.3, loss_weight=1.0),
            norm_cfg=dict(type='BN1d'),
            act_cfg=dict(type='ReLU')),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'https://download.openmmlab.com/mmtracking/mot/reid/tracktor_reid_r50_iter25245-a452f51f.pth'  # noqa: E501
        )),
    tracker=dict(
        type='SORTTracker',
        obj_score_thr=0.5,
        reid=dict(
            num_samples=10,
            img_scale=(256, 128),
            img_norm_cfg=dict(
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),  # Change channel order (1, 2, 3) -> (3, 2, 1)
            match_score_thr=2.0),
        match_iou_thr=0.5,
        momentums=None,
        num_tentatives=4,
        num_frames_retain=120))

train_cfg = None
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadTrackAnnotations', with_instance_id=True),
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
    dict(type='PackTrackInputs', pack_single_img=True),
]

train_dataloader = None
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

train_cfg = None
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
