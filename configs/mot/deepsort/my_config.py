# _base_ = [
#     "../../_base_/datasets/mot_challenge.py",
# ]

custom_imports = dict(
    imports=[
        "mmtrack.models.detectors.yolov7_det",
        "mmtrack.models.detectors.components",
        "mmtrack.models.reid.my_reid",
    ],
    allow_failed_imports=False
)

model = dict(
    type="DeepSORT",
    detector=dict(
        type="YOLOv7",
        backbone=dict(
            type="YOLOv7Net",
            cfg_path="configs/others/yolov7/yolov7-tiny.yaml"
        ),
        bbox_head=dict(
            type="YOLOv7Head",
            num_classes=80,
            conf_thresh=0.5,
            iou_thres=0.45,
        )
    ),
    motion=dict(type="KalmanFilter", center_only=False),
    reid=dict(
        type="MyReID",
        model_name="resnet50",
        model_path=None,
        device="cuda"
    ),
    # reid=dict(
    #     type="BaseReID",
    #     backbone=dict(
    #         type="ResNet", depth=50, num_stages=4, out_indices=(3,), style="pytorch"
    #     ),
    #     neck=dict(type="GlobalAveragePooling", kernel_size=(8, 4), stride=1),
    #     head=dict(
    #         type="LinearReIDHead",
    #         num_fcs=1,
    #         in_channels=2048,
    #         fc_channels=1024,
    #         out_channels=128,
    #         num_classes=380,
    #         loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
    #         loss_pairwise=dict(type="TripletLoss", margin=0.3, loss_weight=1.0),
    #         norm_cfg=dict(type="BN1d"),
    #         act_cfg=dict(type="ReLU"),
    #     ),
    #     init_cfg=dict(
    #         type="Pretrained",
    #         checkpoint="https://download.openmmlab.com/mmtracking/mot/reid/tracktor_reid_r50_iter25245-a452f51f.pth",  # noqa: E251  # noqa: E501
    #     ),
    # ),
    tracker=dict(
        type="SortTracker",
        obj_score_thr=0.5,
        reid=dict(
            num_samples=10, img_scale=(256, 128), img_norm_cfg=None, match_score_thr=2.0
        ),
        match_iou_thr=0.5,
        momentums=None,
        num_tentatives=2,
        num_frames_retain=100,
    ),
)

# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_norm_cfg = dict(
    mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0], to_rgb=True)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1080, 1080),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='VideoCollect', keys=['img'])
        ])
]
dataset_type = 'MOTChallengeDataset'
data_root = 'data/MOT17/'
data = dict(
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/half-val_cocoformat.json',
        img_prefix=data_root + 'train',
        ref_img_sampler=None,
        pipeline=test_pipeline)
)