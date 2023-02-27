import os.path as osp
from pathlib import Path
from argparse import ArgumentParser
import mmengine


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("annotation_path", help="Path to annotations")
    parser.add_argument("-subset_size", type=int, help="Number of images to sample")

    return parser.parse_args()


def main():
    args = parse_args()

    annotation_path = Path(args.annotation_path)

    annotations = mmengine.load(annotation_path)
    images = annotations["images"]

    assert len(images) >= args.subset_size, "The subset size is larger than the dataset size."
    assert args.subset_size > 1, "The subset size must be larger than 1."

    sampling_rate = len(images) // args.subset_size
    sampled_images = images[::sampling_rate]

    annotations["images"] = sampled_images

    # Filter out the annotations that are not in the sampled images.
    anns = []
    img_idx, ann_idx = 0, 0
    while img_idx < len(annotations["images"]) and ann_idx < len(annotations["annotations"]):
        ann = annotations["annotations"][ann_idx]
        img = annotations["images"][img_idx]

        if ann["image_id"] == img["id"]:
            anns.append(ann)
            ann_idx += 1
        elif ann["image_idx"] < img["id"]:
            ann_idx += 1
        else:
            img_idx += 1
    annotations["annotations"] = anns # Update the annotations.

    annotation_name = osp.splitext(annotation_path.name)[0]  # type: ignore
    mmengine.dump(
        annotations,
        annotation_path.parent / f"{annotation_name}_subset_{args.subset_size}.json",
    )


if __name__ == "__main__":
    main()
