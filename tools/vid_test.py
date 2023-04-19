from typing import List, Union, Sequence, Any, Optional
from argparse import ArgumentParser
import os
import os.path as osp

from torch import nn

from mmengine.config import Config, DictAction
from mmtrack.registry import MODELS
from mmtrack.structures import TrackDataSample

from mmengine import load, dump
from mmtrack.utils import register_all_modules

from mmcv import VideoReader, imread
from mmtrack.apis import batch_inference_mot


def parse_args():
    parser = ArgumentParser(description="Test a model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("--checkpoint", help="checkpoint file")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument(
        "--output-dir", default="outputs", help="output directory to save results"
    )

    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)


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
        batched_res = batch_inference_mot(
            model, batched_frames, frame_ids, data_pipeline
        )

        outputs.extend(batched_res)

    return outputs


def get_frames(data_root: str, video_name: str, data_prefix: Optional[str] = None):
    if data_prefix is not None:
        video_name = osp.join(data_prefix, video_name)
    video_path = osp.join(data_root, video_name)
    frames = os.listdir(video_path)
    frames.sort()

    return frames


def get_videos_info(data_root: str, ann_file: str):
    annotations = load(osp.join(data_root, ann_file))["videos"]

    return annotations


def main(args):
    register_all_modules(init_default_scope=False)

    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg.load_from = args.checkpoint

    # build the model and load checkpoint
    model = MODELS.build(cfg.model)
    model.init_weights()

    # prepare data
    data_cfg = cfg.test_dataloader.dataset
    videos_info = get_videos_info(data_cfg.data_root, data_cfg.ann_file)

    pred_results = []
    for vid_info in videos_info:
        frames = get_frames(data_cfg.data_root, vid_info["name"], data_cfg.data_prefix)
        vid_track_results = run_vid_test(
            model, frames, data_cfg.pipeline, args.batch_size
        )

        for track_result in vid_track_results:
            track_ids = track_result.pred_track_instances.instance_id  # type: ignore
            track_bboxes = track_result.pred_track_instances.bboxes  # type: ignore
            track_scores = track_result.pred_track_instances.scores  # type: ignore

            # Convert bbox to xywh format
            track_bboxes[:, 2] -= track_bboxes[:, 0]
            track_bboxes[:, 3] -= track_bboxes[:, 1]

            for track_id, track_bbox, track_score in zip(
                track_ids, track_bboxes, track_scores
            ):
                pred_results.append(
                    {
                        "video_id": vid_info["id"],
                        "frame_id": track_result.metainfo["frame_id"],
                        "track_id": track_id,
                        "bbox": track_bbox.tolist(),
                        "score": track_score.item(),
                    }
                )
        pred_results.extend(vid_track_results)
    dump(
        {"videos": videos_info, "results": pred_results},
        osp.join(args.output_dir, "results.json"),
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
