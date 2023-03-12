# model settings
img_scale = (640, 640)

model = dict(
    data_preprocessor=dict(
        type='TrackDataPreprocessor',
        pad_size_divisor=32,
        batch_augments=[
            dict(
                type='mmdet.BatchSyncRandomResize',
                random_size_range=(320, 640),
                size_divisor=32,
                interval=10)
        ]),
    detector=dict(
        # _scope_='mmdet',
        type='YOLOX',
        backbone=dict(
            type='CSPDarknet', 
            deepen_factor=0.67, 
            widen_factor=0.75),
        neck=dict(
            type='YOLOXPAFPN',
            in_channels=[192, 284, 786],
            out_channels=192,
            num_csp_blocks=2),
        bbox_head=dict(
            type='YOLOXHead',
            num_classes=80,
            in_channels=192,
            feat_channels=192),
        train_cfg=dict(
            assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
        test_cfg=dict(
            score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65))))
