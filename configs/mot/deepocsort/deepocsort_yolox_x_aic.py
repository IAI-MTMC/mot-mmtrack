_base_ = [
    '../../_base_/models/yolox_x_8x8.py',
    '../../_base_/datasets/aicity_challenge.py',
    '../../_base_/default_runtime.py'
]

custom_imports = dict(
    imports=[
        'mmtrack.models.trackers.deep_ocsort_tracker',
        'mmtrack.models.reid.my_reid'
    ],
    allow_failed_imports=False)

img_scale = (800, 1440)

model = dict(
    type='DeepSORT',
    detector=dict(
        _scope_='mmdet',
        bbox_head=dict(num_classes=1),
        test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.7)),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'checkpoints/yolox_x_crowdhuman_mot17-private-half.pth'  # noqa: E501
        )),
    motion=dict(type='KalmanFilter'),
    reid=dict(
        type='MyReID',
        model_name='osnet_x1_0',
        model_path='checkpoints/reid_osnet_aic.pth.tar-5',
        device='cuda'),
    tracker=dict(
        type='DeepOCSORTTracker',
        obj_score_thr=0.1,
        init_track_thr=0.7,
        det_momentum_thr=0.6,
        embed_momentum_factor=0.9,
        reid=dict(
            img_scale=(256, 128),
            img_norm_cfg=dict(
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True)),  # Change channel order (1, 2, 3) -> (3, 2, 1)
        weight_iou_with_det_scores=False,
        match_iou_thr=0.5,
        num_tentatives=3,
        vel_consist_weight=0.2,
        vel_delta_t=3,
        num_frames_retain=100))

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadTrackAnnotations', with_instance_id=True),
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
    dict(type='PackTrackInputs', pack_single_img=True)
]

train_dataloader = None
val_dataloader = dict(
    dataset=dict(
        ann_file='annotations/validation_cocoformat_subset_0.2_consec.json',
        pipeline=test_pipeline))
test_dataloader = val_dataloader

train_cfg = None
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# # evaluator
# val_evaluator = dict(postprocess_tracklet_cfg=[
#     dict(type='InterpolateTracklets', min_num_frames=5, max_num_frames=20)
# ])
# test_evaluator = val_evaluator
