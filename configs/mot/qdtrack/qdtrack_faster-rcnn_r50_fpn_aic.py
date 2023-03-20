_base_ = [
    './qdtrack_faster-rcnn_r50_fpn_4e_base.py',
    '../../_base_/datasets/aicity_challenge.py',
]

img_scale = (640, 640)
# data pipeline
train_pipeline = [
    dict(
        type='TransformBroadcaster',
        share_random_params=True,
        transforms=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadTrackAnnotations', with_instance_id=True),
            dict(
                type='mmdet.RandomResize',
                scale=img_scale,
                ratio_range=(0.8, 1.2),
                keep_ratio=True,
                clip_object_border=False),
            dict(type='mmdet.PhotoMetricDistortion'),
        ]),
    dict(
        type='TransformBroadcaster',
        share_random_params=True,
        transforms=[
            dict(type='mmdet.RandomFlip', prob=0.5),
        ]),
    dict(type='PackTrackInputs', ref_prefix='ref', num_key_frames=1),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadTrackAnnotations', with_instance_id=True),
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
    dict(type='PackTrackInputs', pack_single_img=True),
]

train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        pipeline=train_pipeline,
        ann_file='annotations/train_cocoformat_subset_0.5_consec.json',))
val_dataloader = dict(
    dataset=dict(
        pipeline=test_pipeline,
        ann_file='annotations/val_cocoformat_subset_0.2_consec.json'))
test_dataloader = val_dataloader

# evaluator
val_evaluator = [
    dict(type='CocoVideoMetric', metric=['bbox'], classwise=True),
    dict(type='MOTChallengeMetrics', metric=['HOTA', 'CLEAR', 'Identity'])
]

test_evaluator = val_evaluator
