_base_ = [
    '../../_base_/models/yolov8_m_syncbn_fast.py',
    '../../_base_/datasets/vtx.py', '../../_base_/default_runtime.py'
]

custom_imports = dict(
    imports=[
        'mmtrack.models.reid.my_reid',
        'mmtrack.models.trackers.deep_byte_tracker'
    ],
    allow_failed_imports=False)

model = dict(
    type='DeepSORT',
    detector=dict(
        bbox_head=dict(head_module=dict(num_classes=1)),
        test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.7)),
        init_cfg=dict(
            type='Pretrained', checkpoint='checkpoints/yolov8_m_syncbn_fast_1xb64_10e_vtx.pth')),
    motion=dict(type='KalmanFilter', center_only=False),
    reid=dict(
        type='MyReID',
        model_name='osnet_x1_0',
        model_path='checkpoints/reid/model.pth.tar-100',
        feature_dim=512),
    tracker=dict(
        type='DeepByteTracker',
        obj_score_thrs=dict(high=0.6, low=0.1),
        init_track_thr=0.7,
        weight_iou_with_det_scores=True,
        match_iou_thrs=dict(high=0.1, low=0.5, tentative=0.3),
        weight_assoc_embed=0.75,
        embed_cost_diff_limit=0.5,
        embed_momentum_factor=0.5,
        update_embed_thr=0.5,
        reid=dict(
            reid=True,
            num_samples=10,
            img_scale=(256, 128),
            img_norm_cfg=dict(
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True)),
        num_frames_retain=30))

test_dataloader = dict(dataset=dict(ann_file='sparse_val_cocoformat.json'))
