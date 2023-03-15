import os
import os.path as osp
from collections import defaultdict
from argparse import ArgumentParser

import cv2
from tqdm import tqdm
import mmengine

CLASSES = [dict(id=1, name='person')]


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("data-dir", help="Path to the data directory")
    parser.add_argument("--min-box-height", type=int, default=-1, help="Minimum height of the bounding box")
    parser.add_argument("--min-box-width", type=int, default=-1, help="Minimum width of the bounding box")

    return parser.parse_args()

def parse_gts(ann_path: str):
    """
    Read the annotations from the ground-truth file and format them.

    Args:
        ann_path (str): Path to the annotation file.
        Note: Each line in the annotation file is in the following format:
            `<frame_id>,<track_id>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,1,-1,-1,-1`

    Returns:
        list: List of annotations. Each contains keys:
            - `id`: The id of annotation.
            - `category_id`: The id of the category of annotated object.
            - `instance_id`: The id of the instance of annotated object.
            - `bbox`: The bounding box of annotated object with format `[x_left, y_top, w, h]`.
            - `area`: The area of the bounding box.
            - `true_frame_id`: The true frame id of annotated object.
    """
    outs = defaultdict(list)

    with open(ann_path, "r") as f:
        for ann in f:        
            ann = ann.rstrip().split(",")
            true_frame_id, instance_id = map(int, ann[:2])
            bbox = list(map(float, ann[2:6]))
            category_id = 1
            area = bbox[2] * bbox[3]
            
            ann = dict(
                category_id=category_id,
                bbox=bbox,
                area=area,
                iscrowd=False,
                visibility=1.0,
                mot_conf=1.0,
                true_instance_id=instance_id,
                true_frame_id=true_frame_id)

            outs[true_frame_id].append(ann)
    return outs

def get_image_infos(frames_dir: str, video_name: str):
    """
    Get the frames information from the directory containing the frame images.

    Args:
        frames_dir (str): Path to the directory containing the images.
        video_name (str): Name of the video.

    Returns:
        list: List of image information order by frame_id. Each is a dict with keys:
            - `id`: The id of the image.
            - `file_name`: The name of the image file.
            - `height`: The height of the image.
            - `width`: The width of the image.
            - `frame_id`: The frame id of the image.
            - `true_frame_id`: The true frame id of the image.
    """
    outs = defaultdict(list)
    height, width = None, None
    
    prev_frame_id = -1
    for img_path in os.scandir(frames_dir):
        file_name = os.path.splitext(img_path.name)[0]
        true_frame_id = int(file_name)

        assert true_frame_id > prev_frame_id, f"Frame ids are not in order: {true_frame_id} <= {prev_frame_id}"
        prev_frame_id = true_frame_id

        if height is None:
            height, width = cv2.imread(img_path.path).shape[:2]
        
        img = dict(
            file_name=os.path.join(video_name, "imgs", img_path.name),
            height=height,
            width=width,
            frame_id=id,
            true_frame_id=true_frame_id)

        outs[true_frame_id].append(img)
    
    return outs


def main(args):
    vid_id, img_id, ann_id = 0, 0, 0

    for subset in ("train", "validation"):
        subset_anns = {
            "videos": [],
            "images": [],
            "annotations": [],
            "categories": []
        }
        subset_dir = os.path.join(args.data_dir, subset)
        save_dir = os.path.join(args.data_dir, "annotations")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        print(f'Extracting images from {subset} set')

        ins_id = 0
        for scene_dir in tqdm(os.scandir(subset_dir)):
            if scene_dir.is_dir():
                for camera_dir in os.scandir(scene_dir):
                    if camera_dir.is_dir():
                        imgs_dir = os.path.join(camera_dir.path, "imgs")
                        gt_path = os.path.join(camera_dir.path, "label.txt")
                        video_name = os.path.join(scene_dir.name, camera_dir.name)
                        height, width = None, None

                        video = dict(id=vid_id, name=video_name)

                        ins_maps = dict()
                        # Read the annotations
                        gts = parse_gts(gt_path)
                        # Get the image information
                        for frame_id, image in enumerate(os.scandir(imgs_dir)):
                            if height is None:
                                height, width = cv2.imread(image.path).shape[:2]
                            
                            true_frame_id = int(osp.splitext(image.name)[0])
                            image = dict(
                                id=img_id,
                                video_id=vid_id,
                                file_name=image.path,
                                height=height,
                                width=width,
                                frame_id=frame_id,
                                true_frame_id=true_frame_id)
                            
                            for gt in gts[true_frame_id]:
                                if subset == "train":
                                    if gt["bbox"][2] < args.min_box_width or gt["bbox"][3] < args.min_box_height:
                                        continue
    
                                gt.update(id=ann_id, image_id=img_id)
                                true_ins_id = gt['true_instance_id']
                                if true_ins_id not in ins_maps:
                                    ins_maps[true_ins_id] = ins_id
                                    ins_id += 1
                                gt['instance_id'] = ins_maps[true_ins_id]
                                subset_anns["annotations"].append(gt)
                                ann_id += 1
                            subset_anns["images"].append(image)
                            img_id += 1
                        subset_anns["videos"].append(video)
                        vid_id += 1
        subset_anns["num_instances"] = ins_id
        # Add the categories to the subset
        subset_anns["categories"] = CLASSES
        print(f'{subset} has {ins_id} instances.')

        print("Saving annotations...")
        mmengine.dump(subset_anns, os.path.join(save_dir, f"{subset}_cocoformat.json"))

if __name__ == '__main__':
    args = parse_args()
    main(args)
