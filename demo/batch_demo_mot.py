import os
import os.path as osp
from argparse import ArgumentParser

import cv2
import mmcv
import mmengine

from mmtrack.apis import batch_inference_mot, init_model
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
        '--device', default='cuda:0', help='device used for inference')
    parser.add_argument('--fps', help='FPS of the output video')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument(
        '--vis-pose', type=bool, default=False, help='visualize pose')
    args = parser.parse_args()
    return args


def main(args):
    assert args.output
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
            os.makedirs(osp.dirname(args.output), exist_ok=True)
        else:
            out_path = args.output
            os.makedirs(out_path, exist_ok=True)

    fps = args.fps
    if OUT_VIDEO:
        if fps is None and IN_VIDEO:
            fps = imgs.fps
        if not fps:
            raise ValueError('Please set the FPS for the output video.')
        fps = int(fps)

    register_all_modules(init_default_scope=True)

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    data_pipeline = model.cfg.test_pipeline[2:]

    prog_bar = mmengine.ProgressBar(len(imgs))
    # test and show/save the images
    frame_id_cnt = 0
    outputs = []
    while frame_id_cnt < len(imgs):
        batched_imgs = []
        batch_frame_ids = []

        for _ in range(args.batch_size):
            if frame_id_cnt >= len(imgs):
                break

            batched_imgs.append(imgs[frame_id_cnt])
            batch_frame_ids.append(frame_id_cnt)
            frame_id_cnt += 1

        results = batch_inference_mot(model, batched_imgs, batch_frame_ids,
                                      data_pipeline)
        outputs.extend(results)

        prog_bar.update(len(results))

    print(f'\nmaking the output video at {args.output} with a FPS of {fps}')

    if OUT_VIDEO:
        height, width = outputs[0].metainfo['ori_shape']
        vwriter = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'),
                                  fps, (width, height))

    for result in outputs:
        frame_id = result.metainfo['frame_id']
        out_img = draw_tracked_instances(imgs[frame_id], result, args.vis_pose)
        if OUT_VIDEO:
            vwriter.write(out_img)
        else:
            mmcv.imwrite(out_img, osp.join(out_path, f'{frame_id:06d}.jpg'))

    if OUT_VIDEO:
        vwriter.release()


if __name__ == '__main__':
    args = parse_args()
    main(args)
