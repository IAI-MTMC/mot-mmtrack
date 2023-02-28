import os
import cv2
from tqdm import tqdm
from argparse import ArgumentParser
import mmengine

_video_id = 0
_image_id = 0
_annotation_id = 0

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("data_dir", help="Path to the data directory")

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
    global _annotation_id
    outs = []

    with open(ann_path, "r") as f:
        for ann in f:        
            ann = ann.rstrip().split(",")
            true_frame_id, instance_id = map(int, ann[:2])
            bbox = list(map(float, ann[2:6]))
            category_id = 1
            area = bbox[2] * bbox[3]
            
            ann = dict(
                id=_annotation_id,
                category_id=category_id,
                instance_id=instance_id,
                bbox=bbox,
                area=area,
                iscrowd=False,
                visibility=1.0,
                mot_conf=1.0,
                true_frame_id=true_frame_id)
            _annotation_id += 1 # Increment the annotation id.

            outs.append(ann)
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
    global _image_id

    outs = []
    height, width = None, None
    
    prev_frame_id = -1
    for id, img_path in enumerate(os.scandir(frames_dir)):
        file_name, ext = os.path.splitext(img_path.name)
        true_frame_id = int(file_name)

        assert true_frame_id > prev_frame_id, f"Frame ids are not in order: {true_frame_id} <= {prev_frame_id}"
        prev_frame_id = true_frame_id

        if height is None:
            height, width = cv2.imread(img_path.path).shape[:2]
        
        info = dict(
            id=_image_id,
            file_name=os.path.join(video_name, "imgs", img_path.name),
            height=height,
            width=width,
            frame_id=id,
            true_frame_id=true_frame_id)
        _image_id += 1 # Increment the image id

        outs.append(info)
    
    return outs


def main():
    global _video_id
    args = parse_args()

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

        # Reset id counter for each annotation file
        _video_id = 0
        _image_id = 0
        _annotation_id = 0

        print(f'Extracting images from {subset} set')
        for scene_dir in tqdm(os.scandir(subset_dir)):
            if scene_dir.is_dir():
                for camera_dir in os.scandir(scene_dir):
                    if camera_dir.is_dir():
                        imgs_dir = os.path.join(camera_dir.path, "imgs")
                        gt_path = os.path.join(camera_dir.path, "label.txt")
                        video_name = os.path.join(scene_dir.name, camera_dir.name)

                        # Read the annotations
                        anns = parse_gts(gt_path)
                        imgs = get_image_infos(imgs_dir, video_name)

                        # Match the annotations with the image infos
                        # Since frames are not extracted with the true FPS, we need to match the annotations with the image infos
                        ann_id = 0
                        new_anns = []
                        for img in imgs:
                            while ann_id < len(anns):
                                ann = anns[ann_id]
                                if ann["true_frame_id"] < img["true_frame_id"]:
                                    ann_id += 1
                                elif ann["true_frame_id"] == img["true_frame_id"]:
                                    ann["image_id"] = img["id"]
                                    new_anns.append(ann)
                                    ann_id += 1
                                elif ann["true_frame_id"] > img["true_frame_id"]:
                                    break
                        anns = new_anns

                        # Add video_id keys to the annotations
                        for ann in anns:
                            ann["video_id"] = _video_id
                        
                        # Add video_id keys to the image infos
                        for img in imgs:
                            img["video_id"] = _video_id

                        # Add the annotations and image infos to the subset
                        subset_anns["annotations"].extend(anns)
                        subset_anns["images"].extend(imgs)

                        # Add the videos to the subset
                        subset_anns["videos"].append(
                            dict(
                                id=_video_id,
                                name=video_name,))

                        _video_id += 1 # Increment the video id
                        
        # Add the categories to the subset
        subset_anns["categories"].append(dict(id=1, name="person"))

        print("Saving annotations...")
        mmengine.dump(subset_anns, os.path.join(save_dir, f"{subset}_cocoformat.json"))

if __name__ == '__main__':
    main()
