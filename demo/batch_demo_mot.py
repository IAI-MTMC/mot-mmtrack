# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import tempfile
from argparse import ArgumentParser

import mmcv
import mmengine
from mmengine.dataset import Compose, default_collate

from mmtrack.apis import init_model, batch_inference_mot
from mmtrack.registry import VISUALIZERS
from mmtrack.utils import register_all_modules
from mmtrack.utils.visualization import draw_tracked_instances


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('config', help='config file')
    parser.add_argument('--input', help='input video file or folder')
    parser.add_argument(
        '--output', help='output video file (mp4 format) or folder')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.0,
        help='The threshold of score to filter bboxes.')
    parser.add_argument(
        '--device', default='cuda:0', help='device used for inference')
    parser.add_argument(
        '--show',
        action='store_true',
        help='whether show the results on the fly')
    parser.add_argument('--fps', help='FPS of the output video')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    args = parser.parse_args()
    return args

def main(args):
    assert args.output or args.show
    # load images
    if osp.isdir(args.input):
        imgs = sorted(
            filter(lambda x: x.endswith(('.jpg', '.png', '.jpeg')),
                   os.listdir(args.input)),
            key=lambda x: int(x.split('.')[0]))
        IN_VIDEO = False
    else:
        imgs = mmcv.VideoReader(args.input)
        IN_VIDEO = True

    # define output
    OUT_VIDEO = False
    if args.output is not None:
        if args.output.endswith('.mp4'):
            OUT_VIDEO = True
            out_dir = tempfile.TemporaryDirectory()
            out_path = out_dir.name
            _out = args.output.rsplit(os.sep, 1)
            if len(_out) > 1:
                os.makedirs(_out[0], exist_ok=True)
        else:
            out_path = args.output
            os.makedirs(out_path, exist_ok=True)

    fps = args.fps
    if args.show or OUT_VIDEO:
        if fps is None and IN_VIDEO:
            fps = imgs.fps
        if not fps:
            raise ValueError('Please set the FPS for the output video.')
        fps = int(fps)

    register_all_modules(init_default_scope=True)

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)

    prog_bar = mmengine.ProgressBar(len(imgs))
    # test and show/save the images
    frame_id = 0
    while frame_id < len(imgs):
        batched_imgs = []
        batch_frame_ids = []

        for _ in range(args.batch_size):
            if frame_id >= len(imgs):
                break

            batched_imgs.append(imgs[frame_id])
            batch_frame_ids.append(frame_id)
            frame_id += 1
        
        results = batch_inference_mot(model, batched_imgs, batch_frame_ids)

        for result in results:
            frame_id = result.metainfo['frame_id']

            if args.output is not None:
                if IN_VIDEO or OUT_VIDEO:
                    out_file = osp.join(out_path, f'{frame_id:06d}.jpg')
                else:
                    out_file = osp.join(out_path, imgs[frame_id].rsplit(os.sep, 1)[-1])
            else:
                out_file = None
            
            out_img = draw_tracked_instances(imgs[frame_id].astype('uint8'), result)
            mmcv.imwrite(out_img, out_file)
        
        prog_bar.update(len(results))

    if args.output and OUT_VIDEO:
        print(f'making the output video at {args.output} with a FPS of {fps}')
        mmcv.frames2video(out_path, args.output, fps=fps, fourcc='mp4v')
        out_dir.cleanup()


if __name__ == '__main__':
    args = parse_args()
    main(args)
