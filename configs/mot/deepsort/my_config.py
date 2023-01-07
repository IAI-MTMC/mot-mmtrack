_base_ = [
    "../../_base_/datasets/mot_challenge.py",
    "../../_base_/default_runtime.py",
]

custom_imports = dict(
    imports=[
        "mmtrack.models.detectors.yolov7_det",
        "mmtrack.models.detectors.components",
    ],
    allow_failed_imports=False
)

model = dict(
    type="DeepSORT",
    detector=dict(
        type="YOLOv7",
        backbone=dict(
            type="YOLOv7Net",
            cfg_path="configs/others/yolov7/yolov7-tiny.yaml",
            pretrained_path="https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt"
        ),
        bbox_head=dict(
            type="YOLOv7Head",
            num_classes=80
        )
    ),
    motion=dict(type="KalmanFilter", center_only=False),
    reid=dict(
        type="BaseReID",
        backbone=dict(
            type="ResNet", depth=50, num_stages=4, out_indices=(3,), style="pytorch"
        ),
        neck=dict(type="GlobalAveragePooling", kernel_size=(8, 4), stride=1),
        head=dict(
            type="LinearReIDHead",
            num_fcs=1,
            in_channels=2048,
            fc_channels=1024,
            out_channels=128,
            num_classes=380,
            loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
            loss_pairwise=dict(type="TripletLoss", margin=0.3, loss_weight=1.0),
            norm_cfg=dict(type="BN1d"),
            act_cfg=dict(type="ReLU"),
        ),
        init_cfg=dict(
            type="Pretrained",
            checkpoint="https://download.openmmlab.com/mmtracking/mot/reid/tracktor_reid_r50_iter25245-a452f51f.pth",  # noqa: E251  # noqa: E501
        ),
    ),
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
# learning policy
lr_config = dict(
    policy="step", warmup="linear", warmup_iters=100, warmup_ratio=1.0 / 100, step=[3]
)
# runtime settings
total_epochs = 4
evaluation = dict(metric=["bbox", "track"], interval=1)
search_metrics = ["MOTA", "IDF1", "FN", "FP", "IDs", "MT", "ML"]
