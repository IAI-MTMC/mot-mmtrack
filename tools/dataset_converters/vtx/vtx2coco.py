# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from argparse import ArgumentParser
from collections import defaultdict
from warnings import warn

import mmcv
import mmengine
from tqdm import tqdm

CLASSES = [dict(id=1, name='person')]


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'data_dir', type=str, help='Path to the data directory.')
    parser.add_argument(
        '--details', type=str, help='Path to the predefined subset json file.')

    return parser.parse_args()


def parse_gts(gts):
    outputs = defaultdict(list)
    for gt in gts:
        gt = gt.strip().split(',')
        frame_id, ins_id = map(int, gt[:2])
        bbox = list(map(float, gt[2:6]))
        conf = float(gt[6])
        category_id = int(gt[7])
        visibility = float(gt[8])

        anns = dict(
            category_id=category_id,
            bbox=bbox,
            area=bbox[2] * bbox[3],
            iscrowd=False,
            visibility=visibility,
            mot_instance_id=ins_id,
            mot_conf=conf)
        outputs[frame_id].append(anns)
    return outputs


def main(args):
    if args.details:
        details = mmengine.load(args.details)
        assert isinstance(details, dict), 'Details must be a dict.'
        sets = details['subsets']
        splits = ('train', 'val', 'test')
    else:
        details = None
        sets = ['vtx']
        splits = ['all']

    vid_id, img_id, ann_id, ins_id = 1, 1, 1, 0

    data_dir = args.data_dir
    out_dir = osp.join(data_dir, 'annotations')
    imgs_root_dir = osp.join(data_dir, 'extracted_images')

    outputs = dict()
    for subset in sets:
        for split in splits:
            if split == 'all' or split == 'train':
                subset_name = split
            else:
                subset_name = f'{subset}_{split}'

            outputs[subset_name] = defaultdict(list)
            outputs[subset_name]['categories'] = CLASSES

    for scene_dir in os.scandir(data_dir):
        if not scene_dir.is_dir():
            warn(f'Unexpected file in data directory: {scene_dir.name}')
            continue
        if scene_dir.name in ('annotations', 'extracted_images',
                              '.ipynb_checkpoints'):
            continue

        # Retrieve annotation directory for this scene
        last_version = -1
        for dir in os.listdir(scene_dir.path):
            if dir.startswith('MOT_gt_processed'):
                # It will be something like MOT_gt_processed, MOT_gt_processed_v2, ...
                if dir == 'MOT_gt_processed':
                    version = 1
                else:
                    version = int(dir.rsplit('_', maxsplit=1)[-1][1:])
                if version > last_version:
                    last_version = version
        assert last_version != -1, f'No annotation directory found for scene {scene_dir.name}'
        annotation_dir_name = 'MOT_gt_processed' if last_version == 1 else f'MOT_gt_processed_v{last_version}'
        annotation_dir = osp.join(scene_dir.path, annotation_dir_name)

        for duration_dir in os.scandir(annotation_dir):
            if not duration_dir.is_dir():
                warn(
                    f'Unexpected file in annotation directory: {duration_dir.name}'
                )
                continue
            # Get subset outputs
            if details:
                if scene_dir.name not in details['scenes']:
                    continue
                subset_outputs = None
                for split in splits:
                    if duration_dir.name in details['scenes'][
                            scene_dir.name][split]:
                        if split == 'train':
                            subset_name = split
                        else:
                            subset_name = f'{details["scenes"][scene_dir.name]["subset"]}_{split}'
                        subset_outputs = outputs[subset_name]
                        break
                assert subset_outputs is not None, f'No subset found for scene {scene_dir.name} and duration {duration_dir.name}'
            else:
                subset_outputs = outputs[sets[0]]
            # Instance ID is independent for each duration
            ins_maps = dict()

            for camera_gt_dir in os.scandir(duration_dir.path):
                if not camera_gt_dir.is_dir():
                    warn(
                        f'Unexpected file in camera directory: {camera_gt_dir.name}'
                    )
                    continue

                # video_name will be something like 'MOT17-02-SDP/img1'
                video_name = osp.join(scene_dir.name, duration_dir.name,
                                      camera_gt_dir.name)
                gt_file = osp.join(camera_gt_dir.path, 'gt', 'gt.txt')
                imgs_dir = osp.join(imgs_root_dir, video_name)
                img_names = sorted([
                    img_name for img_name in os.listdir(imgs_dir)
                    if img_name.endswith('.jpg')
                ])

                height, width = mmcv.imread(osp.join(imgs_dir,
                                                     '000001.jpg')).shape[:2]
                video = dict(
                    id=vid_id, name=video_name, width=width, height=height)

                # parse annotations
                gts = mmengine.list_from_file(gt_file)
                # skip video if no instance is found
                if len(gts) == 0:
                    continue
                img2gts = parse_gts(gts)

                for frame_id, name in enumerate(img_names):
                    img_name = osp.join(video_name, name)
                    mot_frame_id = int(osp.splitext(name)[0])

                    image = dict(
                        id=img_id,
                        file_name=img_name,
                        height=height,
                        width=width,
                        video_id=vid_id,
                        frame_id=frame_id,
                        mot_frame_id=mot_frame_id)

                    gts = img2gts[mot_frame_id]
                    for gt in gts:
                        gt.update(id=ann_id, image_id=img_id)
                        mot_ins_id = gt['mot_instance_id']

                        if mot_ins_id not in ins_maps:
                            ins_maps[mot_ins_id] = ins_id
                            ins_id += 1
                        gt['instance_id'] = ins_maps[mot_ins_id]

                        subset_outputs['annotations'].append(gt)
                        ann_id += 1
                    subset_outputs['images'].append(image)
                    img_id += 1
                subset_outputs['videos'].append(video)
                vid_id += 1

    for subset in list(outputs.keys()):
        if len(outputs[subset]['videos']) == 0:
            outputs.pop(subset)
            continue
        out_file = osp.join(out_dir, f'{subset}_cocoformat.json')
        mmengine.dump(outputs[subset], out_file)
        print('=> {} subset is done'.format(subset))


if __name__ == '__main__':
    args = parse_args()
    main(args)
