_base_ = [
    '../../_base_/models/yolov8_m_syncbn_fast.py',
    '../../_base_/datasets/vtx.py', '../../_base_/default_runtime.py'
]

custom_imports = dict(
    imports=[
        'mmtrack.models.reid.my_reid',
        'mmtrack.models.trackers.my_sort_tracker'
    ],
    allow_failed_imports=False)

model = dict(
    type='DeepSORT',
    detector=dict(
        bbox_head=dict(head_module=dict(num_classes=1)),
        test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.7)),
        init_cfg=dict(
            type='Pretrained', checkpoint='checkpoints/detect/epoch_10.pth')),
    motion=dict(type='KalmanFilter', center_only=False),
    reid=dict(
        type='MyReID',
        model_name='osnet_x1_0',
        model_path='checkpoints/reid/model.pth.tar-100',
        device='cuda',
        feature_dim=512),
    tracker=dict(
        type='MySORTTracker',
        obj_score_thr=0.5,
        reid=dict(
            pose=False,
            reid=True,
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
        num_frames_retain=100))

test_dataloader = dict(dataset=dict(ann_file='sparse_val_cocoformat.json'))
