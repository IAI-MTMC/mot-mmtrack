_base_ = [
    '../../_base_/models/yolov8_m_syncbn_fast.py',
    '../../_base_/datasets/vtx.py',
    '../../_base_/default_runtime.py'
]

custom_imports = dict(
    imports=['mmtrack.models.reid.my_reid'], allow_failed_imports=False)

model = dict(
    type='DeepSORT',
    detector=dict(
        bbox_head=dict(
            head_module=dict(num_classes=1)),
        test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.7)),
        init_cfg=dict(
            type='Pretrained', checkpoint='work_dirs/yolov8_m_syncbn_fast_1xb16-10e_vtx/epoch_10.pth')),
    motion=dict(type='KalmanFilter', center_only=False),
    reid=dict(
        type='MyReID',
        model_name='osnet_x1_0',
        model_path='checkpoints/osnet-vtx.pth.tar-50',
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
        num_frames_retain=30))
