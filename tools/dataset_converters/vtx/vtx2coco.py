# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from argparse import ArgumentParser
from collections import defaultdict
from warnings import warn

import mmcv
import mmengine

CLASSES = [dict(id=1, name='person')]


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('data_dir', help='Path to the data directory')
    parser.add_argument(
        '--split-train',
        action='store_true',
        help='Split the train set into half-train and half-validate.')

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
    if args.split_train:
        sets = ['half-train', 'half-val']
    else:
        sets = ['vtx']

    vid_id, img_id, ann_id = 1, 1, 1

    for subset in sets:
        ins_id = 0

        data_dir = args.data_dir
        imgs_root_dir = osp.join(data_dir, 'extracted_images')
        out_file = osp.join(data_dir, 'annotations',
                            f'{subset}_cocoformat.json')

        outputs = defaultdict(list)
        outputs['categories'] = CLASSES

        for scene_dir in os.scandir(data_dir):
            if not scene_dir.is_dir():
                warn(f'Unexpected file in data directory: {scene_dir.name}')
                continue
            if scene_dir.name in ('annotations', 'extracted_images'):
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
            annotation_dir = osp.join(scene_dir.path,
                                      f'MOT_gt_processed_v{last_version}')

            for duration_dir in os.scandir(annotation_dir):
                if not duration_dir.is_dir():
                    warn(
                        f'Unexpected file in annotation directory: {duration_dir.name}'
                    )
                    continue
                ins_maps = dict() # Instance ID is independent for each duration

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
                    img_names = sorted([img_name for img_name in os.listdir(imgs_dir) if img_name.endswith('.jpg')])

                    height, width = mmcv.imread(
                        osp.join(imgs_dir, '000001.jpg')).shape[:2]
                    video = dict(
                        id=vid_id, name=video_name, width=width, height=height)

                    # parse annotations
                    gts = mmengine.list_from_file(gt_file)
                    img2gts = parse_gts(gts)

                    if 'half' in subset:
                        split_frame = len(img_names) // 2 + 1
                        if 'train' in subset:
                            img_names = img_names[:split_frame]
                        elif 'val' in subset:
                            img_names = img_names[split_frame:]
                        else:
                            raise ValueError(
                                f'Subset must be named with `train` or `val`. Got {subset}'
                            )

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

                            outputs['annotations'].append(gt)
                            ann_id += 1
                        outputs['images'].append(image)
                        img_id += 1
                    outputs['videos'].append(video)
                    vid_id += 1
        outputs['num_instances'] = ins_id
        print(f'Subset `{subset}` has {ins_id} instances.')
        mmengine.dump(outputs, out_file)
        print(f'Done! Saved as {out_file}')


if __name__ == '__main__':
    args = parse_args()
    main(args)
