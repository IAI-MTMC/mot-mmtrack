# dataset settings
dataset_type = 'MOTChallengeDataset'
data_root = '../datasets/VTX/COMBINE_DATA_V3'

img_scale = (640, 480)

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
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/sparse_val_cocoformat.json',
        metainfo=dict(CLASSES=('person', )),
        data_prefix=dict(img_path='extracted_images'),
        filter_cfg=dict(filter_empty_gt=True),
        ref_img_sampler=None,
        load_as_video=True,
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# evaluator
val_evaluator = dict(
    type='MOTChallengeMetrics',
    metric=['HOTA', 'CLEAR', 'Identity'],
    postprocess_tracklet_cfg=[
        dict(type='InterpolateTracklets', min_num_frames=5, max_num_frames=20)
    ])
test_evaluator = val_evaluator
