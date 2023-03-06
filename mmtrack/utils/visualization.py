from typing import Union
import numpy as np
import torch
from torchvision.utils import draw_bounding_boxes

from mmtrack.structures import TrackDataSample


def draw_tracked_instances(image: Union[torch.Tensor, np.ndarray],
                           track_sample: TrackDataSample, **kwargs):
    """Draw tracked instances on image.
    Args:
        image (torch.Tensor): The image tensor with shape (C, H, W) and dtype unit8. or
            numpy.ndarray: The image array with shape (H, W, C) and dtype unit8.
        track_sample (TrackDataSample): The track data sample.
    Returns:
        torch.Tensor: The image tensor with shape (C, H, W) or
            numpy.ndarray: The image array with shape (H, W, C) and dtype unit8.
    """
    np_format = False
    if isinstance(image, np.ndarray):
        np_format = True
        image = torch.from_numpy(image).moveaxis(-1, 0)

    pred_track_instances = track_sample.pred_track_instances
    boxes = pred_track_instances.bboxes

    labels = []
    for track_id, track_score in zip(pred_track_instances.instances_ids,
                                     pred_track_instances.scores):
        labels.append(f'{track_id}({track_score:.02f})')

    image = draw_bounding_boxes(image, boxes, labels, **kwargs)

    if np_format:
        image = image.moveaxis(0, -1).numpy()
    return image
