# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import shutil
from argparse import ArgumentParser
from typing import List

import torch
from mmcv import VideoReader
from mmengine.config import Config
from mmengine.dataset import Compose, default_collate
from tqdm import tqdm

from mmtrack.apis import batch_inference_mot, init_model
from mmtrack.models.mot import BaseMultiObjectTracker
from mmtrack.structures import TrackDataSample
from mmtrack.utils import register_all_modules


def parse_args():
    parser = ArgumentParser(description='Test a model')

    parser.add_argument('config', help='test config file path')
    parser.add_argument('--input-dir', help='input directory contains videos')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument(
        '--batch-size', type=int, default=1, help='batch size for inference')
    parser.add_argument(
        '--device', default='cuda:0', help='device used for inference')
    parser.add_argument('--output-dir', help='output directory')
    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


@torch.no_grad()
def test_once(model: BaseMultiObjectTracker, vid_path: str, pipeline: Compose):
    video = VideoReader(vid_path)
    # device = nextmodel.parameters()
    results: List[TrackDataSample] = []
    with tqdm(video) as pbar:
        for frame_id, frame in enumerate(pbar):
            data = dict(
                img=frame, frame_id=frame_id, ori_shape=frame.shape[:2])
            data = pipeline(data)

            data = default_collate([data])
            result = model.test_step(data)
            results.extend(result)
    return results


@torch.no_grad()
def batch_test(model: BaseMultiObjectTracker,
               vid_path: str,
               batch_size: int = 4):

    video = VideoReader(vid_path)
    results: List[TrackDataSample] = []

    frame_id = 0
    with tqdm(total=len(video)) as pbar:
        while frame_id < len(video):
            frame_ids = []
            frames = []

            for _ in range(batch_size):
                if frame_id >= len(video):
                    break

                frame_ids.append(frame_id)
                frames.append(video[frame_id])
                frame_id += 1

            result = batch_inference_mot(model, frames, frame_ids)
            results.extend(result)

            pbar.update(len(result))

    return results


def save_pred(results: List[TrackDataSample], filepath: str):
    """Save tracking results to a text file.

    Args:
        results (list[dict]): Tracking results.
        filepath (str): Path to save the tracking results.
    """
    with open(filepath, 'w') as f:
        for track_sample in results:
            track_ids = track_sample.pred_track_instances.instances_id
            track_bboxes = track_sample.pred_track_instances.bboxes
            track_scores = track_sample.pred_track_instances.scores
            # Convert bbox to xywh format
            track_bboxes[:, 2] -= track_bboxes[:, 0]
            track_bboxes[:, 3] -= track_bboxes[:, 1]

            for track_id, track_bbox, track_score in zip(
                    track_ids, track_bboxes, track_scores):
                f.write(
                    f"{track_sample.metainfo['frame_id']},{track_id},{track_bbox[0]:.3f},{track_bbox[1]:.3f},{track_bbox[2]:.3f},{track_bbox[3]:.3f},{track_score:.3f},-1,-1,-1\n"
                )


def main(args):
    # register all modules in mmtrack into the registries
    # do not init the default scope here because it will be init in the runner
    register_all_modules()

    # load config
    cfg = Config.fromfile(args.config)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.output_dir is None:
        args.output_dir = osp.join('./test_outputs',
                                   osp.splitext(osp.basename(args.config))[0])

    # clean the output directory
    if osp.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)

    model = init_model(cfg, args.checkpoint, device=args.device)

    for vid_name in os.scandir(args.input_dir):
        if vid_name.is_file() and vid_name.name.endswith('.mp4'):
            print(f'Testing {vid_name.name} ...')
            # results = test_once(model, osp.join(args.input_dir, vid_name.name), test_pipeline)
            results = batch_test(
                model,
                osp.join(args.input_dir, vid_name.name),
                batch_size=args.batch_size)

            print(
                f'Saving results to {osp.join(args.output_dir, vid_name.name)} ...'
            )
            save_pred(
                results,
                osp.join(args.output_dir,
                         osp.splitext(vid_name.name)[0] + '.txt'))
        else:
            print(f'Skipping {vid_name.name} ...')

    print('Done!')


if __name__ == '__main__':
    args = parse_args()
    main(args)
