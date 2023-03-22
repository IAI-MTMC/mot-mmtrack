_base_ = [
    '../../_base_/models/yolox_x_8x8.py',
    '../../_base_/datasets/aicity_challenge.py',
    '../../_base_/default_runtime.py'
]

custom_imports = dict(
    imports=['mmtrack.models.reid.my_reid'], allow_failed_imports=False)

img_scale = (800, 1440)

model = dict(
    type='DeepSORT',
    detector=dict(
        bbox_head=dict(num_classes=1),
        test_cfg=dict(score_thr=0.1, nms=dict(type='nms', iou_threshold=0.7)),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='checkpoints/yolox_x_aic.pth')
    ),
    motion=dict(type='KalmanFilter', center_only=False),
    reid=dict(
        type='MyReID',
        model_name='osnet_x1_0',
        model_path='checkpoints/reid_osnet_aic.pth.tar-5',
        device='cuda'),
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
        num_tentatives=2,
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
