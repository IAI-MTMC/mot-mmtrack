# Copyright (c) OpenMMLab. All rights reserved.
import os
import shutil
from argparse import ArgumentParser
from typing import List
from warnings import warn

import motmetrics as mm
import numpy as np
import pandas as pd
import torch
from mmcv import VideoReader
from mmengine.config import Config
from tqdm import tqdm

from mmtrack.apis import batch_inference_mot, init_model
from mmtrack.models import BaseMultiObjectTracker
from mmtrack.structures import TrackDataSample
from mmtrack.structures.bbox import bbox_xyxy_to_x1y1wh
from mmtrack.utils import imshow_mot_errors, register_all_modules
from mmtrack.utils.evaluation import MOTEvaluator


def parse_args():
    parser = ArgumentParser(
        description='visualize errors for multiple object tracking')
    parser.add_argument('config', help='path of the config file')
    parser.add_argument('--checkpoint', help='path of the checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='device used for inference')
    parser.add_argument(
        '--scenes-dir', help='path of the directory contains scenes')
    parser.add_argument(
        '--out-dir',
        help='path of the directory that stores the visualization results')
    parser.add_argument('--batch-size', type=int, default=1)

    args = parser.parse_args()
    return args


def compare_res_gts(results: List[TrackDataSample], gts: str):
    """Evaluate the results of the video.

    Args:
        results (List[TrackDataSample]): The results of the video.
        gts (str): The path of the ground truth file.
    """
    # convert the results to the format of motmetrics
    res = {
        'FrameId': [],
        'Id': [],
        'X': [],
        'Y': [],
        'Width': [],
        'Height': [],
        'Confidence': [],
        'ClassId': [],
        'Visibility': []
    }

    for track_sample in results:
        frame_id = track_sample.metainfo['frame_id']
        track_ids = track_sample.pred_track_instances.instances_id
        track_bboxes = bbox_xyxy_to_x1y1wh(
            track_sample.pred_track_instances.bboxes)
        track_scores = track_sample.pred_track_instances.scores

        for track_id, track_bbox, track_score in zip(track_ids, track_bboxes,
                                                     track_scores):
            res['FrameId'].append(frame_id)
            res['Id'].append(track_id.item())
            res['X'].append(track_bbox[0].item())
            res['Y'].append(track_bbox[1].item())
            res['Width'].append(track_bbox[2].item())
            res['Height'].append(track_bbox[3].item())
            res['Confidence'].append(track_score.item())
            res['ClassId'].append(1)
            res['Visibility'].append(1.0)

    res_df = pd.DataFrame(res)
    res_df = res_df.round({
        'X': 2,
        'Y': 2,
        'Width': 2,
        'Height': 2,
        'Confidence': 3
    })
    res_df.set_index(['FrameId', 'Id'], inplace=True)
    # convert the gts to the format of motmetrics
    gt_df = pd.read_csv(
        gts,
        names=['FrameId', 'Id', 'X', 'Y', 'Width', 'Height', 'ClassId'],
        usecols=range(7))
    # Insert the confidence column before the ClassId column
    gt_df.insert(len(gt_df.columns) - 1, 'Confidence', 1.0)
    gt_df['Visibility'] = 1.0
    gt_df.set_index(['FrameId', 'Id'], inplace=True)

    acc = mm.utils.compare_to_groundtruth(gt_df, res_df)

    return acc, res_df, gt_df


@torch.no_grad()
def batch_test(model: BaseMultiObjectTracker,
               vid_path: str,
               batch_size: int = 1):

    video = VideoReader(vid_path)
    results: List[TrackDataSample] = []

    model.tracker.reset()
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


def _save_trackeval_gt_pred(gt: pd.DataFrame, pred: pd.DataFrame, seq: str,
                            pred_dir: str, gt_dir: str):
    """Save the gts and preds to the format of TrackEval.

    Args:
        gt (pd.DataFrame): Ground truth annotations.
        pred (pd.DataFrame): Predicted annotations.
        seq (str): The name of the sequence.
        pred_dir (str): The directory that stores the predicted annotations.
        gt_dir (str): The directory that stores the ground truth annotations.
    """
    os.makedirs(os.path.join(gt_dir, seq))
    gt_path = os.path.join(gt_dir, seq, 'gt.txt')
    pred_path = os.path.join(pred_dir, seq + '.txt')

    gt = gt.copy()
    pred = pred.copy()

    pred.drop(columns=['ClassId', 'Visibility'], inplace=True)
    pred.drop(
        [0, 1],
        inplace=True)  # Remove the first and second frames to match with gt
    pred['A'] = -1
    pred['B'] = -1
    pred['C'] = -1

    # Save to text files
    gt.to_csv(gt_path, sep=',', header=False)
    pred.to_csv(pred_path, sep=',', header=False)

    # Create the seqmap file
    seqmap_dir = os.path.join(gt_dir, 'seqmaps')
    seqmap_file = os.path.join(seqmap_dir, 'MOT17-train.txt')
    if os.path.exists(seqmap_file):
        with open(seqmap_file, 'a') as f:
            f.write(seq + '\n')
    else:
        os.makedirs(seqmap_dir)
        with open(seqmap_file, 'w') as f:
            f.write('name\n')
            f.write(seq + '\n')

    # Create seqinfo.ini
    with open(os.path.join(gt_dir, seq, 'seqinfo.ini'), 'w') as f:
        f.write('[Sequence]\n')
        f.write('name=' + seq + '\n')
        f.write('imDir=img\n')
        f.write('frameRate=30\n')
        f.write('seqLength=' +
                str(gt.index.get_level_values('FrameId').max() + 1) + '\n')
        f.write('imWidth=1920\n')
        f.write('imHeight=1080\n')
        f.write('imExt=.jpg\n')


def _make_trackeval_dirs(root_dir: str = './cache',
                         tracker_name: str = 'default-tracker'):
    """Make the directories that stores the gts and preds."""
    if os.path.exists(root_dir):
        shutil.rmtree(root_dir)

    gt_dir = os.path.join(root_dir, 'gt')
    trackers_dir = os.path.join(root_dir, 'trackers')
    tracker_dir = os.path.join(trackers_dir, tracker_name)

    os.makedirs(gt_dir)
    os.makedirs(tracker_dir)

    return gt_dir, trackers_dir, tracker_dir


def main(args):
    register_all_modules()

    # load config
    cfg = Config.fromfile(args.config)
    tracker_name = os.path.splitext(os.path.basename(args.config))[0]
    output_dir = os.path.join(args.out_dir, tracker_name)
    gt_dir, trackers_dir, tracker_dir = _make_trackeval_dirs(
        tracker_name=tracker_name)

    pred_cache_dir = 'predcache'
    os.makedirs(pred_cache_dir, exist_ok=True)

    if os.path.exists(output_dir):
        warn(f'{output_dir} already exists. It will be removed.')
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    model = init_model(cfg, args.checkpoint, device=args.device)
    assert isinstance(model, BaseMultiObjectTracker)

    for scene in os.scandir(args.scenes_dir):
        if not scene.is_dir():
            continue
        for camera in os.scandir(scene.path):
            if not camera.is_dir():
                continue

            print(f'Processing {scene.name}/{camera.name} ...')
            # load the video
            vid_path = os.path.join(camera.path, 'video.mp4')
            gt_path = os.path.join(camera.path, 'label.txt')

            cached_res = os.path.join(pred_cache_dir, camera.name + '.pkl')
            if os.path.exists(cached_res):
                import torch
                results = torch.load(cached_res)
            else:
                results = batch_test(
                    model, vid_path, batch_size=args.batch_size)
                import torch
                torch.save(results, cached_res)
            acc, res, gt = compare_res_gts(results, gt_path)
            _save_trackeval_gt_pred(gt, res, camera.name, tracker_dir, gt_dir)

            # frames_id_list = sorted(
            # list(set(acc.mot_events.index.get_level_values(0))))
            # vid_reader = VideoReader(vid_path)
            # for frame_id in frames_id_list:
            #     # events in the current frame
            #     events = acc.mot_events.xs(frame_id)
            #     cur_res = res.loc[frame_id] if frame_id in res.index else None
            #     cur_gt = gt.loc[frame_id] if frame_id in gt.index else None
            #     # path of image
            #     # img = filenames_dict[video_name][frame_id]
            #     fps = events[events.Type == 'FP']
            #     fns = events[events.Type == 'MISS']
            #     idsws = events[events.Type == 'SWITCH']

            #     bboxes, ids, error_types = [], [], []
            #     for fp_index in fps.index:
            #         hid = events.loc[fp_index].HId
            #         bboxes.append([
            #             cur_res.loc[hid].X, cur_res.loc[hid].Y,
            #             cur_res.loc[hid].X + cur_res.loc[hid].Width,
            #             cur_res.loc[hid].Y + cur_res.loc[hid].Height,
            #             cur_res.loc[hid].Confidence
            #         ])
            #         ids.append(hid)
            #         # error_type = 0 denotes false positive error
            #         error_types.append(0)
            #     for fn_index in fns.index:
            #         oid = events.loc[fn_index].OId
            #         bboxes.append([
            #             cur_gt.loc[oid].X, cur_gt.loc[oid].Y,
            #             cur_gt.loc[oid].X + cur_gt.loc[oid].Width,
            #             cur_gt.loc[oid].Y + cur_gt.loc[oid].Height,
            #             cur_gt.loc[oid].Confidence
            #         ])
            #         ids.append(-1)
            #         # error_type = 1 denotes false negative error
            #         error_types.append(1)
            #     for idsw_index in idsws.index:
            #         hid = events.loc[idsw_index].HId
            #         bboxes.append([
            #             cur_res.loc[hid].X, cur_res.loc[hid].Y,
            #             cur_res.loc[hid].X + cur_res.loc[hid].Width,
            #             cur_res.loc[hid].Y + cur_res.loc[hid].Height,
            #             cur_res.loc[hid].Confidence
            #         ])
            #         ids.append(hid)
            #         # error_type = 2 denotes id switch
            #         error_types.append(2)
            #     if len(bboxes) == 0:
            #         bboxes = np.zeros((0, 5), dtype=np.float32)
            #     else:
            #         bboxes = np.asarray(bboxes, dtype=np.float32)
            #     ids = np.asarray(ids, dtype=np.int32)
            #     error_types = np.asarray(error_types, dtype=np.int32)

            #     imshow_mot_errors(
            #         vid_reader[frame_id],
            #         bboxes,
            #         ids,
            #         error_types,
            #         show=False,
            #         out_file=os.path.join(output_dir,
            #                         f'{camera.name}/{frame_id:06d}.jpg')
            #         if args.out_dir else None)

        evaluator = MOTEvaluator(tracker_dir, gt_dir)
        evaluator.compute_metrics()


if __name__ == '__main__':
    args = parse_args()
    main(args)
