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

dataset_info = dict(
    dataset_name='vtx',
    keypoint_info={
        'nose': dict(id=0, color=[51, 153, 255], swap=''),
        'left_eye': dict(id=1, color=[51, 153, 255], swap='right_eye'),
        'right_eye': dict(id=2, color=[51, 153, 255], swap='left_eye'),
        'left_ear': dict(id=3, color=[51, 153, 255], swap='right_ear'),
        'right_ear': dict(id=4, color=[51, 153, 255], swap='left_ear'),
        'left_shoulder': dict(id=5, color=[0, 255, 0], swap='right_shoulder'),
        'right_shoulder': dict(id=6, color=[255, 128, 0], swap='left_shoulder'),
        'left_elbow': dict(id=7, color=[0, 255, 0], swap='right_elbow'),
        'right_elbow': dict(id=8, color=[255, 128, 0], swap='left_elbow'),
        'left_wrist': dict(id=9, color=[0, 255, 0], swap='right_wrist'),
        'right_wrist': dict(id=10, color=[255, 128, 0], swap='left_wrist'),
        'left_hip': dict(id=11, color=[0, 255, 0], swap='right_hip'),
        'right_hip': dict(id=12, color=[255, 128, 0], swap='left_hip'),
        'left_knee': dict(id=13, color=[0, 255, 0], swap='right_knee'),
        'right_knee': dict(id=14, color=[255, 128, 0], swap='left_knee'),
        'left_ankle': dict(id=15, color=[0, 255, 0], swap='right_ankle'),
        'right_ankle': dict(id=16, color=[255, 128, 0], swap='left_ankle')
    },
    skeleton_info=[
        dict(link=('left_ankle', 'left_knee'), id=0, color=[0, 255, 0]),
        dict(link=('left_knee', 'left_hip'), id=1, color=[0, 255, 0]),
        dict(link=('right_ankle', 'right_knee'), id=2, color=[255, 128, 0]),
        dict(link=('right_knee', 'right_hip'), id=3, color=[255, 128, 0]),
        dict(link=('left_hip', 'right_hip'), id=4, color=[51, 153, 255]),
        dict(link=('left_shoulder', 'left_hip'), id=5, color=[51, 153, 255]),
        dict(link=('right_shoulder', 'right_hip'), id=6, color=[51, 153, 255]),
        dict(link=('left_shoulder', 'right_shoulder'),id=7,color=[51, 153, 255]),
        dict(link=('left_shoulder', 'left_elbow'), id=8, color=[0, 255, 0]),
        dict(link=('right_shoulder', 'right_elbow'), id=9, color=[255, 128, 0]),
        dict(link=('left_elbow', 'left_wrist'), id=10, color=[0, 255, 0]),
        dict(link=('right_elbow', 'right_wrist'), id=11, color=[255, 128, 0]),
        dict(link=('left_eye', 'right_eye'), id=12, color=[51, 153, 255]),
        dict(link=('nose', 'left_eye'), id=13, color=[51, 153, 255]),
        dict(link=('nose', 'right_eye'), id=14, color=[51, 153, 255]),
        dict(link=('left_eye', 'left_ear'), id=15, color=[51, 153, 255]),
        dict(link=('right_eye', 'right_ear'), id=16, color=[51, 153, 255]),
        dict(link=('left_ear', 'left_shoulder'), id=17, color=[51, 153, 255]),
        dict(link=('right_ear', 'right_shoulder'), id=18, color=[51, 153, 255]),
    ],
)
