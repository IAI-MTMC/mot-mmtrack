# Copyright (c) OpenMMLab. All rights reserved.
import multiprocessing as mp
import os
import shutil
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from warnings import warn

import mmcv
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('data_dir', help='Path to the data directory')
    parser.add_argument(
        '--num-workers',
        type=int,
        default=mp.cpu_count(),
        help='Number of workers to use. Default: all available cpus')

    return parser.parse_args()


def extract_images(vid_path: str, imgs_dir: str):
    print(f'Extracting images from video: {vid_path}')
    assert os.path.exists(vid_path), f'Missing video: {vid_path}'

    reader = mmcv.VideoReader(vid_path)
    if os.path.exists(imgs_dir):
        extracted_imgs = os.listdir(imgs_dir)
        if len(extracted_imgs) == reader.frame_cnt:
            warn(f'Images already extracted: {imgs_dir}. Skipping...')
            return
        else:
            warn(
                f'Mismatch in number of extracted images. Number of extracted images: {len(extracted_imgs)}. Expected {reader.frame_cnt}'
            )
            shutil.rmtree(imgs_dir)
    os.makedirs(imgs_dir)

    for frame_id, frame in enumerate(reader):
        frame_id += 1  # 1-based indexing
        mmcv.imwrite(frame, os.path.join(imgs_dir, f'{frame_id:06d}.jpg'))


def main(args):
    imgs_root_dir = os.path.join(args.data_dir, 'extracted_images')

    # Retrieve video paths
    vid_paths, imgs_dir = [], []
    for scene_dir in os.scandir(args.data_dir):
        if not scene_dir.is_dir():
            warn(f'Skipping file: {scene_dir.name}')
            continue
        if scene_dir.name in ('extracted_images', 'annotations',
                              '.ipynb_checkpoints'):
            continue

        vids_dir = os.path.join(scene_dir.path, 'videos')
        for duration_dir in os.scandir(vids_dir):
            if not duration_dir.is_dir():
                warn(f'Skipping file: {duration_dir.name}')
                continue

            for vid_file in os.scandir(duration_dir.path):
                cam_id = vid_file.name.split('_')[0]

                if cam_id.isnumeric():
                    vid_paths.append(vid_file.path)
                    imgs_dir.append(
                        os.path.join(imgs_root_dir, scene_dir.name,
                                     duration_dir.name, cam_id))
                else:
                    warn(f'Skipping video: {vid_file.name}')

    # Extract images
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        for _ in tqdm(
                executor.map(extract_images, vid_paths, imgs_dir),
                total=len(vid_paths)):
            pass


if __name__ == '__main__':
    main(parse_args())
