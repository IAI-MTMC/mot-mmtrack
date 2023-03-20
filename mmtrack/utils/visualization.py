# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np
import seaborn as sns

from mmtrack.structures import TrackDataSample


def _random_color(seed):
    """Random a color according to the input seed."""
    np.random.seed(seed)
    colors = sns.color_palette()
    color = colors[np.random.choice(range(len(colors)))]
    color = tuple([int(255 * c) for c in color])
    return color


def draw_tracked_instances(image: np.ndarray,
                           track_sample: TrackDataSample,
                           thickness: int = 2):
    """
    Args:
        image (np.ndarray): The image to draw on.
        track_sample (TrackDataSample): The track sample to draw.
        thickness (int): The thickness of the lines.

    Returns:
        np.ndarray: The image with drawn track instances.
    """

    # Adapt the font size according to the image size
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = min(image.shape[0], image.shape[1]) * 5e-4
    font_thickness = 2
    text_padding = 1

    pred_track_instances = track_sample.pred_track_instances
    boxes = pred_track_instances.bboxes

    labels = []
    for track_id, track_score in zip(pred_track_instances.instances_id,
                                     pred_track_instances.scores):
        labels.append(f'{track_id}-{track_score:.02f}')

    colors = [
        _random_color(track_id)
        for track_id in pred_track_instances.instances_id
    ]

    for box, label, color in zip(boxes, labels, colors):
        box = box.round().int().tolist()
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color,
                      thickness)

        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale,
                                              font_thickness)
        cv2.rectangle(image,
                      (box[0] - text_padding, box[1] - text_h - text_padding),
                      (box[0] + text_w + text_padding, box[1] + text_padding),
                      color, -1)
        cv2.putText(image, label, (box[0], box[1]), font, font_scale,
                    (255, 255, 255), font_thickness)

    return image
