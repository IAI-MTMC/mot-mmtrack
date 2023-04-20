# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from argparse import ArgumentParser
from collections import defaultdict
from tempfile import TemporaryDirectory
from typing import Any, List, Optional, Sequence, Union

from mmcv import VideoReader, imread
from mmengine import dump, load
from mmengine.config import Config, DictAction
from torch import nn
from tqdm import tqdm

from mmtrack.apis import batch_inference_mot
from mmtrack.registry import MODELS
from mmtrack.structures import TrackDataSample
from mmtrack.utils import register_all_modules


def parse_args():
    parser = ArgumentParser(description='Test a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--device', default='cuda:0', help='device')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument(
        '--output-dir',
        default='./work_dirs',
        help='output directory to save results')

    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.',
    )
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher',
    )
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def get_frames(data_root: str,
               video_name: str,
               data_prefix: Optional[str] = None):
    if data_prefix is not None:
        video_name = osp.join(data_prefix, video_name)
    video_path = osp.join(data_root, video_name)
    frames = os.listdir(video_path)
    frames.sort()

    frames = [osp.join(video_path, frame_name) for frame_name in frames]

    return frames


def load_annotations(data_root: str, ann_file: str):
    annotations = load(osp.join(data_root, ann_file))

    return annotations


def run_vid_test(
    model: nn.Module,
    frames: Union[Sequence[str], Sequence[Any]],
    data_pipeline: List[dict],
    batch_size: int = 1,
):
    model.eval()

    outputs: List[TrackDataSample] = []
    data_iter = iter(frames)
    frame_id_cnt = 0
    done = False
    while not done:
        batched_frames = []
        frame_ids = []

        for _ in range(batch_size):
            try:
                frame = next(data_iter)
                if isinstance(frame, str):
                    frame = imread(frame)
                batched_frames.append(frame)
                frame_ids.append(frame_id_cnt)
                frame_id_cnt += 1
            except StopIteration:
                done = True
                break
        batched_res = batch_inference_mot(model, batched_frames, frame_ids,
                                          data_pipeline)

        outputs.extend(batched_res)

    return outputs


def run_evaluate(track_results, tracker_name='default_tracker'):
    # prepare gt and pred folder
    with TemporaryDirectory() as tmpdir:
        vid2frames = defaultdict(list)
        image2gts = defaultdict(list)
        frame2preds = defaultdict(list)

        for image in track_results['images']:
            vid2frames[image['video_id']].append(image)
        for gt in track_results['annotations']:
            image2gts[gt['image_id']].append(gt)
        for pred in track_results['predictions']:
            frame2preds[pred['frame_id']].append(pred)

        gt_dir = osp.join(tmpdir, 'gt')
        pred_dir = osp.join(tmpdir, 'trackers')
        seqmap_file = osp.join(gt_dir, 'seqmaps.txt')
        with open(seqmap_file, 'wt') as f:
            f.write('name\n')
            for video in track_results['videos']:
                f.write(f'{video["name"]}\n')

        for video in track_results['videos']:
            video_id = video['id']
            video_name = video['name']
            save_gt_dir = osp.join(gt_dir, video_name, 'gt')
            save_pred_dir = osp.join(pred_dir, tracker_name, video_name,
                                     'data')
            os.makedirs(save_gt_dir, exist_ok=True)
            os.makedirs(save_pred_dir, exist_ok=True)

            for image in vid2frames[video_id]:
                image_id = image['id']
                frame_id = image['frame_id']
                gt_file = osp.join(save_gt_dir, 'gt.txt')
                pred_file = osp.join(save_pred_dir, f'{video_name}.txt')
                gts = image2gts[image_id]
                preds = frame2preds[frame_id]

                with open(gt_file, 'wt') as f:
                    for gt in gts:
                        line = '%d,%d,%d,%d,%d,%d,%d,%d,%.5f\n' % (
                            frame_id,
                            gt['instance_id'],
                            gt['bbox'][0],
                            gt['bbox'][1],
                            gt['bbox'][2],
                            gt['bbox'][3],
                            gt['mot_conf'],
                            gt['category_id'],
                            gt['visibility'],
                        )
                        f.write(line)

                with open(pred_file, 'wt') as f:
                    for pred in preds:
                        line = '%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,-1,-1,-1\n' % (
                            frame_id,
                            pred['instance_id'],
                            pred['bbox'][0],
                            pred['bbox'][1],
                            pred['bbox'][2],
                            pred['bbox'][3],
                            pred['mot_conf'],
                        )
                        f.write(line)

            # save .ini file
            ini_file = osp.join(gt_dir, video_name, 'seqinfo.ini')
            with open(ini_file, 'wt') as f:
                f.write('[Sequence]\n')
                f.write(f'name={video_name}')
                f.write(f'imDir={video_name}')
                f.write(f'seqLength={len(vid2frames[video_id])}')
                f.write(f'imWidth={video["width"]}')
                f.write(f'imHeight={video["height"]}')
                f.write(f'imExt=.jpg')

        from .TrackEval import trackeval

        eval_config = trackeval.Evaluator.get_default_eval_config()
        eval_config['DISPLAY_LESS_PROGRESS'] = False
        dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config(
        )
        dataset_config['GT_FOLDER'] = gt_dir
        dataset_config['TRACKERS_FOLDER'] = pred_dir
        dataset_config['SEQMAP_FILE'] = seqmap_file
        metrics_config = {
            'METRICS': ['HOTA', 'CLEAR', 'Identity'],
            'THRESHOLD': 0.5
        }

        evaluator = trackeval.Evaluator(eval_config)
        dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
        metrics_list = []
        for metric in [
                trackeval.metrics.HOTA, trackeval.metrics.CLEAR,
                trackeval.metrics.Identity, trackeval.metrics.VACE
        ]:
            if metric.get_name() in metrics_config['METRICS']:
                metrics_list.append(metric(metrics_config))
        if len(metrics_list) == 0:
            raise Exception('No metrics selected for evaluation')

        evaluator.evaluate(dataset_list, metrics_list)


def main(args):
    register_all_modules()

    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg.load_from = args.checkpoint

    # build the model and load checkpoint
    model = MODELS.build(cfg.model)
    model.init_weights()
    model.to(args.device)

    # prepare data
    data_cfg = cfg.test_dataloader.dataset
    annotations = load_annotations(data_cfg.data_root, data_cfg.ann_file)

    tracker_name = osp.splitext(osp.basename(args.config))[0]
    out_file = osp.join(args.output_dir, tracker_name + '.json')

    # Generate results of tracker
    if not osp.exists(out_file):
        pred_results = []
        with tqdm(annotations['videos']) as pbar:
            for vid_info in pbar:
                pbar.set_description(f"Processing {vid_info['name']}")

                frames = get_frames(data_cfg.data_root, vid_info['name'],
                                    data_cfg.data_prefix.img_path)
                vid_track_results = run_vid_test(model, frames,
                                                 data_cfg.pipeline[2:],
                                                 args.batch_size)

                for track_result in vid_track_results:
                    track_ids = track_result.pred_track_instances.instances_id  # type: ignore
                    track_bboxes = track_result.pred_track_instances.bboxes  # type: ignore
                    track_scores = track_result.pred_track_instances.scores  # type: ignore

                    # Convert bbox to xywh format
                    track_bboxes[:, 2] -= track_bboxes[:, 0]
                    track_bboxes[:, 3] -= track_bboxes[:, 1]

                    for track_id, track_bbox, track_score in zip(
                            track_ids, track_bboxes, track_scores):
                        pred_results.append({
                            'video_id':
                            vid_info['id'],
                            'frame_id':
                            track_result.metainfo['frame_id'],
                            'instance_id':
                            track_id.item(),
                            'bbox':
                            track_bbox.tolist(),
                            'mot_conf':
                            track_score.item(),
                        })
        annotations['predictions'] = pred_results
        dump(annotations, out_file)

    # Start evaluating...
    track_results = load(out_file)
    run_evaluate(track_results, tracker_name)


if __name__ == '__main__':
    args = parse_args()
    main(args)
