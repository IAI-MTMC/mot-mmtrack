# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from argparse import ArgumentParser
from typing import Any, List, Optional

from mlops.database import init_mot_exporter
from mmcv import VideoReader, imread
from mmengine import dump, load
from mmengine.config import Config, DictAction
from tqdm import tqdm

from mmtrack.apis import batch_inference_mot
from mmtrack.registry import MODELS
from mmtrack.utils import register_all_modules

MOT_EXPORTER = init_mot_exporter()


def parse_args():
    default_workdirs = osp.join(
        osp.dirname(osp.abspath(__file__)), 'work_dirs')

    parser = ArgumentParser(description='Test a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--device', default='cuda:0', help='device')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument(
        '--max-storage',
        type=int,
        default=100,
        help='Max documents to store before push to database')
    parser.add_argument(
        '--error-analysis', action='store_true', help='run error analysis')
    parser.add_argument(
        '--output-dir',
        default=default_workdirs,
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


def get_video_path(data_root: str, video_name: str):
    scene, duration, camera_id = video_name.split('_')
    # Hard code for VTX extracted images structure
    videos_dir = osp.join(data_root, scene, 'videos', duration)

    video_path = None
    for camera_name in os.listdir(videos_dir):
        if camera_name.startswith(camera_id):
            video_path = osp.join(videos_dir, camera_name)
            break
    if video_path is None:
        raise ValueError(f'video path not found for {video_name}')
    return video_path


def load_annotations(data_root: str, ann_file: str):
    annotations = load(osp.join(data_root, ann_file))

    return annotations


def main(args):
    register_all_modules()

    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # Generate results of tracker
    cfg.load_from = args.checkpoint

    # build the model and load checkpoint
    model = MODELS.build(cfg.model)
    model.init_weights()
    model.to(args.device)

    # prepare data
    data_cfg = cfg.test_dataloader.dataset
    annotations = load_annotations(data_cfg.data_root, data_cfg.ann_file)

    subset_name = osp.splitext(osp.basename(data_cfg.ann_file))[0]
    tracker_name = osp.splitext(osp.basename(args.config))[0]
    code_name = f'{tracker_name}-{subset_name}'

    model.eval()
    data_storage = []
    data_pipeline = data_cfg.pipeline[2:]
    with tqdm(annotations['videos']) as pbar:
        for vid_info in pbar:
            pbar.set_description(f"Processing {vid_info['name']}")

            scene_id, duration_id, camera_id = map(int,
                                                   vid_info['name'].split('/'))
            frames = get_frames(data_cfg.data_root, vid_info['name'],
                                data_cfg.data_prefix.img_path)

            data_iter = iter(frames)
            frame_id_cnt = 0
            done = False
            while not done:
                batched_frames = []
                frame_ids = []

                for _ in range(args.batch_size):
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
                batched_res = batch_inference_mot(model, batched_frames,
                                                  frame_ids, data_pipeline)

                for track_result in batched_res:
                    track_ids = track_result.pred_track_instances.instances_id  # type: ignore
                    track_bboxes = track_result.pred_track_instances.bboxes  # type: ignore
                    track_scores = track_result.pred_track_instances.scores  # type: ignore
                    track_embeds = track_result.pred_track_instances.embeds  # type: ignore
                    assert track_embeds is not None, 'Embedding is not available'

                    # Convert bbox to xywh format
                    track_bboxes[:, 2] -= track_bboxes[:, 0]
                    track_bboxes[:, 3] -= track_bboxes[:, 1]

                    for track_id, track_bbox, track_score, track_embed in zip(
                            track_ids, track_bboxes, track_scores,
                            track_embeds):
                        data_storage.append({
                            'code_version':
                            code_name,
                            'scene_id':
                            scene_id,
                            'duration_id':
                            duration_id,
                            'stream_id':
                            camera_id,
                            'frame_number':
                            track_result.metainfo['frame_id'] + 1,
                            'object': {
                                'box': {
                                    'x': track_bbox[0].item(),
                                    'y': track_bbox[1].item(),
                                    'w': track_bbox[2].item(),
                                    'h': track_bbox[3].item(),
                                },
                                'object_id': track_id.item(),
                                'confidence': track_score.item(),
                                'embedding': track_embed.tolist(),
                            }
                        })

                        if len(data_storage) > args.max_storage:
                            MOT_EXPORTER.export(data_storage)
                            print(f'Exported {len(data_storage)} data')
                            data_storage.clear()

            if len(data_storage) > 0:
                MOT_EXPORTER.export(data_storage)
                print(f'Exported {len(data_storage)} data')
                data_storage.clear()


if __name__ == '__main__':
    args = parse_args()
    main(args)
