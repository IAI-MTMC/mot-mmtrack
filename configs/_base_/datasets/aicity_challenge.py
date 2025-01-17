# dataset settings
dataset_type = "MOTChallengeDataset"
data_root = "../datasets/AIC23_Track1_MTMC_Tracking"

# data pipeline
train_pipeline = [
    dict(
        type="TransformBroadcaster",
        share_random_params=True,
        transforms=[
            dict(type="LoadImageFromFile"),
            dict(type="LoadTrackAnnotations", with_instance_id=True),
            dict(
                type="mmdet.RandomResize",
                scale=(640, 640),
                ratio_range=(0.8, 1.2),
                keep_ratio=True,
                clip_object_border=False,
            ),
            dict(type="mmdet.PhotoMetricDistortion"),
        ],
    ),
    dict(
        type="TransformBroadcaster",
        share_random_params=False,
        transforms=[
            dict(
                type="mmdet.RandomCrop",
                crop_size=(640, 640),
                bbox_clip_border=False)
        ],
    ),
    dict(
        type="TransformBroadcaster",
        share_random_params=True,
        transforms=[
            dict(type="mmdet.RandomFlip", prob=0.5),
        ],
    ),
    dict(type="PackTrackInputs", ref_prefix="ref", num_key_frames=1),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadTrackAnnotations", with_instance_id=True),
    dict(type="mmdet.Resize", scale=(640, 640), keep_ratio=True),
    dict(type="PackTrackInputs", pack_single_img=True),
]

# dataloader
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=dict(type="mmdet.AspectRatioBatchSampler"),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="annotations/train_cocoformat.json",
        data_prefix=dict(img_path="train/"),
        metainfo=dict(CLASSES=("person", )),
        ref_img_sampler=dict(
            num_ref_imgs=1,
            frame_range=10,
            filter_key_img=True,
            method="uniform"),
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="VideoSampler"),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="annotations/validation_cocoformat.json",
        data_prefix=dict(img_path="validation/"),
        metainfo=dict(CLASSES=("person", )),
        ref_img_sampler=None,
        load_as_video=True,
        test_mode=True,
        pipeline=test_pipeline,
    ),
)
test_dataloader = val_dataloader

# evaluator
val_evaluator = dict(
    type="MOTChallengeMetrics",
    benchmark="MOT15",
    metric=["HOTA", "CLEAR", "Identity"])
test_evaluator = val_evaluator
