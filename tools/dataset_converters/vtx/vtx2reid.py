import argparse
import os
import os.path as osp
from warnings import warn
import random
from collections import defaultdict

import mmcv
import mmengine
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert VTX dataset into ReID dataset.')
    parser.add_argument('data_dir', help='path of VTX data')
    parser.add_argument('--output', help='path to save ReID dataset')
    parser.add_argument('--details', help='path to the predefined subset json file')
    parser.add_argument('--vis-threshold', type=float, default=0.3, help='threshold of visibility for each person')
    parser.add_argument('--min-per-person', type=int, default=8, help='minimum number of images for each person')
    parser.add_argument('--max-per-person', type=int, default=10000, help='maxmum number of images for each person')

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
        splits = ('train', 'val', 'test')
    else:
        details = None
        splits = ['all']

    if not osp.isdir(args.output):
        os.makedirs(args.output)
    elif os.listdir(args.output):
        raise OSError(f'Directory must empty: \'{args.output}\'')
    
    data_dir = args.data_dir
    imgs_root_dir = osp.join(data_dir, 'extracted_images')
    out_dir = args.output

    for scene_dir in tqdm(os.scandir(data_dir)):
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
            
            if details:
                if scene_dir.name not in details['scenes']:
                    continue

                reid_folder = None
                for split in splits:
                    if duration_dir.name in details['scenes'][
                            scene_dir.name][split]:
                        reid_folder = osp.join(out_dir, split)
                if reid_folder is None:
                    warn(f'No split found for duration {scene_dir.name}/{duration_dir.name}')
                    continue
            else:
                reid_folder = osp.join(out_dir, splits[0])
            
            os.makedirs(reid_folder, exist_ok=True)

            for camera_gt_dir in os.scandir(duration_dir.path):
                if not camera_gt_dir.is_dir():
                    warn(
                        f'Unexpected file in camera directory: {camera_gt_dir.name}'
                    )
                    continue

                video_name = osp.join(scene_dir.name, duration_dir.name,
                                    camera_gt_dir.name)
                gt_file = osp.join(camera_gt_dir.path, 'gt', 'gt.txt')
                imgs_dir = osp.join(imgs_root_dir, video_name)
                img_names = sorted([
                    img_name for img_name in os.listdir(imgs_dir)
                    if img_name.endswith('.jpg')
                ])

                # parse annotations
                gts = mmengine.list_from_file(gt_file)
                # skip video if no instance is found
                if len(gts) == 0:
                    continue
                img2gts = parse_gts(gts)

                for img_name in img_names:
                    img_path = osp.join(imgs_dir, img_name)
                    raw_img = mmcv.imread(img_path)
                    mot_frame_id = int(osp.splitext(img_name)[0])
                    gts = img2gts[mot_frame_id]

                    for gt in gts:
                        if gt['visibility'] < args.vis_threshold:
                            continue
                        
                        mot_ins_id = gt['mot_instance_id']
                        reid_img_folder = osp.join(reid_folder, 'imgs', f'{video_name.replace("/", "_")}_{mot_ins_id:06d}')
                        os.makedirs(reid_img_folder, exist_ok=True)
                        idx = len(os.listdir(reid_img_folder))
                        reid_img_name = f'{idx:06d}.jpg'
                        
                        ltwh = gt['bbox']
                        xyxy = np.asarray(
                            [ltwh[0], ltwh[1], ltwh[0] + ltwh[2], ltwh[1] + ltwh[3]])
                        reid_img = mmcv.imcrop(raw_img, xyxy)
                        mmcv.imwrite(reid_img, osp.join(reid_img_folder, reid_img_name)) 

            reid_img_folder_names = os.listdir(osp.join(reid_folder, 'imgs'))
            reid_dataset_list = []
            label = 0
            for reid_img_folder_name in reid_img_folder_names:
                reid_img_names = os.listdir(osp.join(reid_folder, 'imgs', reid_img_folder_name))

                # ignore ids whose number of image is less than min_per_person
                if (len(reid_img_names) < args.min_per_person):
                    continue
                # downsampling when there are too many images owned by one id
                if (len(reid_img_names) > args.max_per_person):
                    reid_img_names = random.sample(reid_img_names, args.max_per_person)
                
                for reid_img_name in reid_img_names:
                    reid_dataset_list.append(f'{osp.join(reid_folder, "imgs", reid_img_folder_name)} {label}\n')
                label += 1
            with open(osp.join(reid_folder, 'meta.txt'), 'w') as f:
                f.writelines(reid_dataset_list)

if __name__ == '__main__':
    args = parse_args()
    main(args)
