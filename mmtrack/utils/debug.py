# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from typing import List

import cv2
import numpy as np
import torch


class ReIDDebugger:

    def __init__(self, out_dir: str):
        self.memo = dict()
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def update(self, frame_id: int, img: torch.Tensor, im_mean: List[float],
               im_std: List[float], embed: torch.Tensor):
        im_mean = torch.tensor(im_mean).to(img).view(3, 1, 1)
        im_std = torch.tensor(im_std).to(img).view(3, 1, 1)
        img = img * im_std + im_mean
        img = img.permute(1, 2, 0).cpu().numpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if frame_id not in self.memo:
            self.memo[frame_id] = []
        self.memo[frame_id].append(dict(img=img, embed=embed))

    def visualize(self):
        if len(self.memo) < 2:
            return

        frame_ids = sorted(self.memo.keys())
        cur_frame_id = frame_ids[-1]
        prev_frame_id = frame_ids[-2]

        cur_frame = self.memo[cur_frame_id]
        prev_frame = self.memo[prev_frame_id]

        for id, info in enumerate(cur_frame):
            dists = dict()
            for prev_id, prev_info in enumerate(prev_frame):
                dist = torch.dist(info['embed'], prev_info['embed']).item()
                dists[prev_id] = dist
            # Sort ids from previous frame by distance to current id
            ordered_ids = sorted(
                list(range(len(prev_frame))), key=lambda x: dists[x])

            imgs = [info['img']]
            for prev_id in ordered_ids:
                prev_img = prev_frame[prev_id]['img'].copy()
                cv2.putText(prev_img, f'Dist: {dists[prev_id]:.2f}', (0, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                imgs.append(prev_img)
            img = np.concatenate(imgs, axis=1)
            cv2.imwrite(
                osp.join(self.out_dir, f'{cur_frame_id}_{id}.jpg'), img)

    def clear(self, keep_current: bool = True):
        if keep_current:
            frame_ids = sorted(self.memo.keys())

            for frame_id in frame_ids[:-1]:
                self.memo.pop(frame_id)
        else:
            self.memo.clear()
