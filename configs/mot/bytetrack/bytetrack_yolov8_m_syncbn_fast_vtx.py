_base_ = [
    '../../_base_/models/yolov8_m_syncbn_fast.py',
    '../../_base_/default_runtime.py'
]

img_scale = (640, 480)

model = dict(
    type='ByteTrack',
    detector=dict(
        bbox_head=dict(head_module=dict(num_classes=1)),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='checkpoints/yolov8_m_syncbn_fast_1xb64_10e_vtx.pth')),
    motion=dict(type='KalmanFilter'),
    tracker=dict(
        type='ByteTracker',
        obj_score_thrs=dict(high=0.6, low=0.1),
        init_track_thr=0.7,
        weight_iou_with_det_scores=True,
        match_iou_thrs=dict(high=0.1, low=0.5, tentative=0.3),
        num_frames_retain=30))

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

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='VideoSampler'),
    dataset=dict(
        # type=dataset_type,
        # data_root=data_root,
        ann_file='annotations/half-val_cocoformat.json',
        data_prefix=dict(img_path='train'),
        ref_img_sampler=None,
        load_as_video=True,
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader