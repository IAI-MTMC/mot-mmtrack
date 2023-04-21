_base_ = [
    '../../_base_/models/yolov8_m_syncbn_fast.py',
    '../../_base_/datasets/vtx.py', '../../_base_/default_runtime.py'
]

model = dict(
    type='ByteTrack',
    detector=dict(
        bbox_head=dict(head_module=dict(num_classes=1)),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='work_dirs/yolov8_m_syncbn_fast_1xb16-10e_vtx/epoch_10.pth')),
    motion=dict(type='KalmanFilter'),
    tracker=dict(
        type='ByteTracker',
        obj_score_thrs=dict(high=0.6, low=0.1),
        init_track_thr=0.7,
        weight_iou_with_det_scores=True,
        match_iou_thrs=dict(high=0.1, low=0.5, tentative=0.3),
        num_frames_retain=30))

test_dataloader = dict(
    dataset=dict(
        ann_file='annotations/crowd_val_cocoformat.json'))
