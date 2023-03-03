# model settings
img_scale = (640, 640)

model = dict(
    data_preprocessor=dict(
        type='TrackDataPreprocessor',
        pad_size_divisor=32,
        batch_augments=[
            dict(
                type='mmdet.BatchSyncRandomResize',
                random_size_range=(480, 800),
                size_divisor=32,
                interval=10)
        ]),
    detector=dict(
        # _scope_='mmdet',
        type='YOLOX',
        backbone=dict(
            type='CSPDarknet', 
            deepen_factor=1.0, 
            widen_factor=1.0),
        neck=dict(
            type='YOLOXPAFPN',
            in_channels=[256, 512, 1024],
            out_channels=256,
            num_csp_blocks=3),
        bbox_head=dict(
            type='YOLOXHead',
            num_classes=80,
            in_channels=256,
            feat_channels=256),
        train_cfg=dict(
            assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
        test_cfg=dict(
            score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65))))
