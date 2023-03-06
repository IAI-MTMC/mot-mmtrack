import os.path as osp
from pathlib import Path
from argparse import ArgumentParser
import mmengine


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("annotation_path", help="Path to annotations")
    parser.add_argument("--ratio", type=float, help="The ratio of the subset size to the dataset size.")
    parser.add_argument("--strategy", type=str, help="The strategy to sample the subset. Can be 'consec' or 'step'")

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
        video_id = img["video_id"]
        if video_id not in images_by_video:
            images_by_video[video_id] = []
        images_by_video[video_id].append(img)
    return images_by_video

def _check_sorted_by_key(l, key: str = "id"):
    """Check if the list is sorted by `key`.

    Args:
        l (list[dict]): The list to check.
        key (str): The key to check

    Returns:
        bool: Whether the list is sorted by `key.
    """
    for i in range(len(l) - 1):
        if l[i + 1][key] < l[i][key]:
            return False
    return True

def main():
    args = parse_args()

    annotation_path = Path(args.annotation_path)

    annotations = mmengine.load(annotation_path)
    images = annotations["images"]

    assert 0 < args.ratio < 1, "The ratio must be in (0, 1)."
    assert args.strategy in ("consec", "step"), "The strategy must be 'consec' or 'step'."
    img_ids = [img["id"] for img in images]
    assert len(set(img_ids)) == len(img_ids), "The image ids are not unique."

    image_grs = _group_images_by_video(images)
    sampled_imgs = []
    for video_id, imgs in image_grs.items():
        assert _check_sorted_by_key(imgs, "frame_id"), "The images are not in order by frame id"

        num_samples = int(len(imgs) * args.ratio)

        if args.strategy == "consec":
            sampled_imgs.extend(imgs[:num_samples])
        elif args.strategy == "step":
            step_size = len(imgs) // num_samples
            sampled_imgs.extend(imgs[::step_size])
        else:
            raise ValueError(f"Unknown strategy {args.strategy}")

    annotations["images"] = sampled_imgs

    # Filter out the annotations that are not in the sampled images.
    assert _check_sorted_by_key(annotations["images"], "id"), "The images are not sorted by id."
    assert _check_sorted_by_key(annotations["annotations"], "id"), "The annotations are not sorted by id."

    anns = []
    img_idx, ann_idx = 0, 0
    while img_idx < len(annotations["images"]) and ann_idx < len(annotations["annotations"]):
        ann = annotations["annotations"][ann_idx]
        img = annotations["images"][img_idx]

        if ann["image_id"] == img["id"]:
            anns.append(ann)
            ann_idx += 1
        elif ann["image_id"] < img["id"]:
            ann_idx += 1
        else:
            img_idx += 1
    annotations["annotations"] = anns # Update the annotations.

    annotation_name = osp.splitext(annotation_path.name)[0]  # type: ignore
    mmengine.dump(
        annotations,
        annotation_path.parent / f"{annotation_name}_subset_{args.ratio}_{args.strategy}.json",
    )


if __name__ == "__main__":
    main()
