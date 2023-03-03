_base_ = [
    '../../_base_/models/yolox_x_8x8.py',
    '../../_base_/datasets/aicity_challenge.py', 
    '../../_base_/default_runtime.py'
]

img_scale = (640, 640)

model = dict(
    type='ByteTrack',
    detector=dict(
        _scope_='mmdet',
        bbox_head=dict(num_classes=1),
        test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.7)),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/home/hoang/Workspace/mot-mmtrack/checkpoints/epoch_10.pth'  # noqa: E501
        )),
    motion=dict(type='KalmanFilter'),
    tracker=dict(
        type='ByteTracker',
        obj_score_thrs=dict(high=0.6, low=0.1),
        init_track_thr=0.7,
        weight_iou_with_det_scores=True,
        match_iou_thrs=dict(high=0.1, low=0.5, tentative=0.3),
        num_frames_retain=180))

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadTrackAnnotations', with_instance_id=True),
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='mmdet.Pad',
        size_divisor=32,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='PackTrackInputs', pack_single_img=True)
]

train_dataloader = None
val_dataloader = dict(
    dataset=dict(
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        ann_file='annotations/validation_cocoformat_subset_0.2_consec.json',
        pipeline=test_pipeline))
test_dataloader = val_dataloader

# evaluator
val_evaluator = dict(postprocess_tracklet_cfg=[
    dict(type='InterpolateTracklets', min_num_frames=5, max_num_frames=20)
])
test_evaluator = val_evaluator

train_cfg = None
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
