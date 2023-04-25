import numpy as np
from mmengine.dataset import Compose
from mmpose.datasets.transforms import LoadImage, GetBBoxCenterScale, PackPoseInputs
from .pose_embedding import FullBodyPoseEmbedder

class PosePipeline:
    def __init__(self):
        self.pose_embedder = FullBodyPoseEmbedder()
        self.pose_pipeline = Compose(
            [LoadImage(),
             GetBBoxCenterScale(padding=1.0),
             PackPoseInputs()])
    
    @property
    def embedding_size():
        return 46

    def prepare_pose_data(self, img, bboxes, scores, crops):
            print('prepare_pose_data')
            pose_data = []

            for bbox, score, crop in zip(bboxes, scores, crops):
                data = self.pose_pipeline(dict(img=img,
                                            bbox=bbox[None]))  # shape (1, 4)
                pds = data['data_samples']
                pds.gt_instances.bbox_scores = score.reshape(1)
                pds.set_field(
                    (crop.shape[2], crop.shape[1]),  # w, h
                    'input_size',
                    field_type='metainfo')
                pds.set_field(
                    (0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15),
                    'flip_indices',
                    field_type='metainfo')

                pose_data.append(pds)
            return pose_data
    
    def draw_img(self, bboxes, img, pose_results):
        print('draw_img')
        import cv2
        mean = np.array([[[123.675, 116.28, 103.53]]])
        std = np.array([[[58.395, 57.12, 57.375]]])
        img = img * std + mean

        cv2.imwrite('image.jpg', img[:, :, ::-1])
        img = cv2.imread('image.jpg')

        color = (255, 255, 0)
        thickness = 2

        for k in range(bboxes.shape[0]):
            start_point = (int(bboxes[k][0]), int(bboxes[k][1]))
            end_point = (int(bboxes[k][2]), int(bboxes[k][3]))
            img = cv2.rectangle(img, start_point, end_point, color, thickness)

            landmarks = pose_results[k].pred_instances.keypoints.reshape(-1, 2)
            for i in range(landmarks.shape[0]):
                center_coordinates = (int(landmarks[i][0]),
                                      int(landmarks[i][1]))
                radius = 3
                color = (100, 255, 100)
                thickness = 1
                img = cv2.circle(img, center_coordinates, radius, color,
                                 thickness)

        cv2.imwrite('image1.jpg', img)

    def get_pose_embedded(self, bboxes, scores, metainfo, img, crops,
                          pose_estimator):

        bboxes = bboxes.detach().cpu().numpy()
        scores = scores.detach().cpu().numpy()
        img = img.squeeze().detach().moveaxis(0, -1).cpu().numpy()

        factor_x, factor_y = metainfo['scale_factor']
        bboxes_scale = bboxes[:, :4] * np.array(
            [factor_x, factor_y, factor_x, factor_y])

        pose_data = self.prepare_pose_data(img, bboxes_scale, scores, crops)
        pose_results = pose_estimator.predict(crops, pose_data)
        # self.draw_img(bboxes_scale, img, pose_results)

        pose_embedded = self.pose_embbedder(pose_results, bboxes_scale)

        for k in range(len(pose_results)):
            keypoints = pose_results[k].pred_instances.keypoints[0]
            keypoints /= np.array([factor_x, factor_y])
        return pose_results, pose_embedded