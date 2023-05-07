_base_ = [
    '../../_base_/models/yolov8_m_syncbn_fast.py',
    '../../_base_/datasets/vtx.py', '../../_base_/default_runtime.py'
]

custom_imports = dict(
    imports=[
        'mmtrack.models.mot.my_deep_sort',
        'mmtrack.models.trackers.my_sort_tracker'
    ],
    allow_failed_imports=False)

model = dict(
    type='MyDeepSORT',
    detector=dict(
        bbox_head=dict(head_module=dict(num_classes=1)),
        test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.7)),
        init_cfg=dict(
            type='Pretrained', checkpoint='checkpoints/detect/epoch_10.pth')),
    motion=dict(type='KalmanFilter', center_only=False),
    pose=dict(
        type='TopdownPoseEstimator',
        _scope_='mmpose',
        data_preprocessor=None,
        backbone=dict(
            type='ResNet',
            depth=50,
            init_cfg=dict(
                type='Pretrained', checkpoint='torchvision://resnet50'),
        ),
        head=dict(
            type='HeatmapHead',
            in_channels=2048,
            out_channels=17,
            loss=dict(type='KeypointMSELoss', use_target_weight=True),
            decoder=dict(
                type='MSRAHeatmap',
                input_size=(192, 256),
                heatmap_size=(48, 64),
                sigma=2)),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_256x192-ec54d7f3_20200709.pth'
        ),
        test_cfg=dict(
            flip_test=False,
            flip_mode='heatmap',
            shift_heatmap=True,
        )),
    tracker=dict(
        type='MySORTTracker',
        obj_score_thr=0.5,
        reid=dict(
            img_scale=(256, 128),
            img_norm_cfg=dict(
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),  # Change channel order (1, 2, 3) -> (3, 2, 1)
            pose=0.2,
            num_samples_pose=1,
            match_score_thr_pose=1.0),
        match_iou_thr=0.5,
        momentums=None,
        num_tentatives=2,
        num_frames_retain=100))

test_dataloader = dict(
    dataset=dict(ann_file='annotations/sparse_val_cocoformat.json'))
