# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from argparse import ArgumentParser
from pathlib import Path

import mmengine


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('annotation_path', help='Path to annotations')
    parser.add_argument(
        '--ratio',
        type=float,
        help='The ratio of the subset size to the dataset size.')

    return parser.parse_args()


def _group_images_by_video(images):
    """Group the images by video.

    Args:
        images (list[dict]): The images.

    Returns:
        dict[int, list[dict]]: The images grouped by video.
    """
    images_by_video = {}
    for img in images:
        video_id = img['video_id']
        if video_id not in images_by_video:
            images_by_video[video_id] = []
        images_by_video[video_id].append(img)
    return images_by_video


def _group_annotations_by_image(annotations):
    """Group the annotations by image.

    Args:
        annotations (list[dict]): The annotations.

    Returns:
        dict[int, list[dict]]: The annotations grouped by image.
    """
    annotations_by_image = {}
    for ann in annotations:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)
    return annotations_by_image


def main(args):
    annotation_path = Path(args.annotation_path)
    annotations = mmengine.load(annotation_path)
    videos = annotations['videos']

    assert 0 < args.ratio < 1, 'The ratio should be in range (0, 1).'
    num_videos = int(len(videos) * args.ratio)
    subset_videos = videos[:num_videos]

    annotations['videos'] = subset_videos

    # Filter images and annotations
    images_by_video = _group_images_by_video(annotations['images'])
    subset_images = []
    for video in subset_videos:
        subset_images.extend(images_by_video[video['id']])
    annotations['images'] = subset_images

    annotations_by_image = _group_annotations_by_image(
        annotations['annotations'])
    subset_annotations = []
    for image in subset_images:
        image_id = image['id']
        if image_id in annotations_by_image:
            subset_annotations.extend(annotations_by_image[image_id])
    annotations['annotations'] = subset_annotations

    ann_name = osp.splitext(annotation_path.name)[0]
    mmengine.dump(
        annotations,
        annotation_path.parent / f'{ann_name}_subset_{args.ratio}.json')


if __name__ == '__main__':
    main(parse_args())
