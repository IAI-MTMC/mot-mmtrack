# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Union

import numpy as np
import torch
from mmcv.ops import RoIPool
from mmengine.dataset import Compose, default_collate
from torch import nn

from mmtrack.structures import TrackDataSample


def batch_inference_mot(
        model: nn.Module, imgs: List[np.ndarray], frame_ids: List[int],
        data_pipeline: Union[List[dict], Compose]) -> List[TrackDataSample]:
    """Inference image(s) with the mot model.

    Args:
        model (nn.Module): The loaded mot model.
        imgs (List[np.ndarray]): Loaded images in shape (H x W x C).
        frame_ids (List[int]): List of frame id for each image.
        data_pipeline (Compose): Data processing pipeline.

    Returns:
        SampleList: The tracking data samples.
    """
    if not isinstance(data_pipeline, Compose):
        data_pipeline = Compose(data_pipeline)

    data_samples = []
    for frame_id, img in zip(frame_ids, imgs):
        data = dict(
            img=img.astype(np.float32),
            frame_id=frame_id,
            ori_shape=img.shape[:2])
        data = data_pipeline(data)
        data_samples.append(data)
    data_samples = default_collate(data_samples)

    if not next(model.parameters()).is_cuda:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    # forward the model
    with torch.no_grad():
        data_samples = model.data_preprocessor(data_samples, False)
        results = model.batch_predict(**data_samples)
    return results
