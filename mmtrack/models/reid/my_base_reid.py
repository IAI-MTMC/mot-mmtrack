# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch
from mmcls.models.classifiers import ImageClassifier

from mmtrack.registry import MODELS
from mmtrack.structures import ReIDDataSample


@MODELS.register_module()
class MyBaseReID(ImageClassifier):
    """Base model for re-identification."""

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[ReIDDataSample]] = None,
                mode: str = 'tensor',
                frame_id: int = -1):
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
          tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
          processed to a list of :obj:`ReIDDataSample`.
        - "loss": Forward and return a dict of losses according to the given
          inputs and data samples.

        Note that this method doesn't handle neither back propagation nor
        optimizer updating, which are done in the :meth:`train_step`.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, H, W) or (N, T, C, H, W).
            data_samples (List[ReIDDataSample], optional): The annotation
                data of every sample. It's required if ``mode="loss"``.
                Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of
              :obj:`ReIDDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """

        if len(inputs.size()) == 5:
            assert inputs.size(0) == 1
            inputs = inputs[0]

        self.visactmap(inputs, 'actmap', frame_id)
        return super().forward(inputs, data_samples, mode)

    @torch.no_grad()
    def visactmap(self, imgs, actmap_dir, frame_id, width=128, height=256):

        img_mean = [0.485, 0.456, 0.406]
        img_std = [0.229, 0.224, 0.225]
        GRID_SPACING = 10

        print(imgs.shape)
        outputs = self.backbone(imgs)[0]
        print(outputs.shape)

        import cv2
        import numpy as np
        import torch.nn.functional as F

        # compute activation maps
        outputs = (outputs**2).sum(1)
        print(outputs.shape)

        b, h, w = outputs.size()
        outputs = outputs.view(b, h * w)
        outputs = F.normalize(outputs, p=2, dim=1)
        outputs = outputs.view(b, h, w)

        imgs, outputs = imgs.cpu(), outputs.cpu()

        for j in range(outputs.size(0)):
            # RGB image
            img = imgs[j, ...]
            for t, m, s in zip(img, img_mean, img_std):
                t.mul_(s).add_(m).clamp_(0, 1)
            img_np = np.uint8(np.floor(img.numpy() * 255))
            img_np = img_np.transpose((1, 2, 0))  # (c, h, w) -> (h, w, c)

            # activation map
            am = outputs[j, ...].numpy()
            am = cv2.resize(am, (width, height))
            am = 255 * (am - np.min(am)) / (np.max(am) - np.min(am) + 1e-12)
            am = np.uint8(np.floor(am))
            am = cv2.applyColorMap(am, cv2.COLORMAP_JET)

            # overlapped
            overlapped = img_np * 0.3 + am * 0.7
            overlapped[overlapped > 255] = 255
            overlapped = overlapped.astype(np.uint8)

            # save images in a single figure (add white spacing between images)
            # from left to right: original image, activation map, overlapped image
            grid_img = 255 * np.ones(
                (height, 3 * width + 2 * GRID_SPACING, 3), dtype=np.uint8)
            grid_img[:, :width, :] = img_np[:, :, ::-1]
            grid_img[:, width + GRID_SPACING:2 * width + GRID_SPACING, :] = am
            grid_img[:, 2 * width + 2 * GRID_SPACING:, :] = overlapped
            img_name = actmap_dir + f'/{frame_id}_{j}' + '.jpg'
            print(img_name)
            cv2.imwrite(img_name, grid_img)
