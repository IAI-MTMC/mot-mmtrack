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


def draw_tracked_instances(
    image: np.ndarray,
    track_sample: TrackDataSample,
    vis_pose: bool = False,
    dataset_info: dict = None,
    thickness: int = 2,
):
    """
    Args:
        image (np.ndarray): The image to draw on.
        track_sample (TrackDataSample): The track sample to draw.
        vis_pose (bool): Whether to draw the pose. Default: False.
        thickness (int): The thickness of the lines.

    Returns:
        np.ndarray: The image with drawn track instances.
    """

    # Adapt the font size according to the image size
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = min(image.shape[0], image.shape[1]) * 5e-4
    font_thickness = 1
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
                      (box[0] - text_padding, box[1] + text_h + text_padding),
                      (box[0] + text_w + text_padding, box[1] + text_padding),
                      color, -1)
        cv2.putText(image, label, (box[0], box[1] + text_h), font, font_scale,
                    (255, 255, 255), font_thickness, cv2.LINE_AA)

    if vis_pose:
        image = visualize_pose(image, pred_track_instances.pose, dataset_info)

    return image


def visualize_pose(image, pose_results, dataset_info, thickness=1, radius=2):
    keypoint_info = dataset_info.keypoint_info
    skeleton_info = dataset_info.skeleton_info

    for pose_result in pose_results:
        landmarks = pose_result.pred_instances.keypoints.reshape(-1, 2)

        # draw keypoint
        for keypoint in keypoint_info.values():
            id = keypoint['id']
            color = keypoint['color']
            keypoint = (int(landmarks[id][0]), int(landmarks[id][1]))
            image = cv2.circle(image, keypoint, radius, color, thickness)

        # draw skeleton
        for skeleton in skeleton_info:
            start_keypoint, end_keypoint = skeleton['link']
            start_id = keypoint_info[start_keypoint]['id']
            end_id = keypoint_info[end_keypoint]['id']

            start_point = (int(landmarks[start_id][0]),
                           int(landmarks[start_id][1]))
            end_point = (int(landmarks[end_id][0]), int(landmarks[end_id][1]))
            color = skeleton['color']
            image = cv2.line(image, start_point, end_point, color, thickness)

    return image
