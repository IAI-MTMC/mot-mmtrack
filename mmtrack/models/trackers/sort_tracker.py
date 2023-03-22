# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import numpy as np
import torch
from mmdet.structures.bbox import bbox_overlaps
from mmengine.structures import InstanceData
from motmetrics.lap import linear_sum_assignment
from torch import Tensor

from mmtrack.registry import MODELS
from mmtrack.structures import TrackDataSample
from mmtrack.structures.bbox import bbox_xyxy_to_cxcyah
from mmtrack.utils import OptConfigType, imrenormalize
from .base_tracker import BaseTracker
from mmtrack.models.motion.klt_tracker import KLTTracker
from mmtrack.models.motion.utils import Track
import cv2

flow_cfg = {
    "bg_feat_scale_factor": [0.1, 0.1],
    "opt_flow_scale_factor": [0.5, 0.5],
    "feat_density": 0.005,
    "feat_dist_factor": 0.06,
    "ransac_max_iter": 500,
    "ransac_conf": 0.99,
    "max_error": 100,
    "inlier_thresh": 4,
    "bg_feat_thresh": 10,
    "obj_feat_params": {
        "maxCorners": 1000,
        "qualityLevel": 0.06,
        "blockSize": 3
    },
    "opt_flow_params": {
        "winSize": [5, 5],
        "maxLevel": 5,
        "criteria": [3, 10, 0.03]
    }
}


@MODELS.register_module()
class SORTTracker(BaseTracker):
    """Tracker for SORT/DeepSORT.

    Args:
        obj_score_thr (float, optional): Threshold to filter the objects.
            Defaults to 0.3.
        reid (dict, optional): Configuration for the ReID model.
            - num_samples (int, optional): Number of samples to calculate the
                feature embeddings of a track. Default to 10.
            - image_scale (tuple, optional): Input scale of the ReID model.
                Default to (256, 128).
            - img_norm_cfg (dict, optional): Configuration to normalize the
                input. Default to None.
            - match_score_thr (float, optional): Similarity threshold for the
                matching process. Default to 2.0.
        match_iou_thr (float, optional): Threshold of the IoU matching process.
            Defaults to 0.7.
        num_tentatives (int, optional): Number of continuous frames to confirm
            a track. Defaults to 3.
    """

    def __init__(self,
                 obj_score_thr: float = 0.3,
                 reid: dict = dict(
                     num_samples=10,
                     img_scale=(256, 128),
                     img_norm_cfg=None,
                     match_score_thr=2.0),
                 match_iou_thr: float = 0.7,
                 num_tentatives: int = 3,
                 **kwargs):
        super().__init__(**kwargs)
        self.obj_score_thr = obj_score_thr
        self.reid = reid
        self.match_iou_thr = match_iou_thr
        self.num_tentatives = num_tentatives
        # from mmtrack.utils.debug import ReIDDebugger
        # self.reid_debugger = ReIDDebugger("reid_debug_low")

    @property
    def confirmed_ids(self) -> List:
        """Confirmed ids in the tracker."""
        ids = [id for id, track in self.tracks.items() if not track.tentative]
        return ids

    def init_track(self, id: int, obj: Tuple[Tensor]) -> None:
        """Initialize a track."""
        super().init_track(id, obj)
        self.tracks[id].tentative = True
        bbox = bbox_xyxy_to_cxcyah(self.tracks[id].bboxes[-1])  # size = (1, 4)
        assert bbox.ndim == 2 and bbox.shape[0] == 1
        bbox = bbox.squeeze(0).cpu().numpy()
        self.tracks[id].mean, self.tracks[id].covariance = self.kf.initiate(
            bbox)

    def update_track(self, id: int, obj: Tuple[Tensor]) -> None:
        """Update a track."""
        super().update_track(id, obj)
        if self.tracks[id].tentative:
            if len(self.tracks[id]['bboxes']) >= self.num_tentatives:
                self.tracks[id].tentative = False
        bbox = bbox_xyxy_to_cxcyah(self.tracks[id].bboxes[-1])  # size = (1, 4)
        assert bbox.ndim == 2 and bbox.shape[0] == 1
        bbox = bbox.squeeze(0).cpu().numpy()
        self.tracks[id].mean, self.tracks[id].covariance = self.kf.update(
            self.tracks[id].mean, self.tracks[id].covariance, bbox)

    def pop_invalid_tracks(self, frame_id: int) -> None:
        """Pop out invalid tracks."""
        invalid_ids = []
        for k, v in self.tracks.items():
            # case1: disappeared frames >= self.num_frames_retrain
            case1 = frame_id - v['frame_ids'][-1] >= self.num_frames_retain
            # case2: tentative tracks but not matched in this frame
            case2 = v.tentative and v['frame_ids'][-1] != frame_id
            if case1 or case2:
                invalid_ids.append(k)
        for invalid_id in invalid_ids:
            self.tracks.pop(invalid_id)

    def track(self,
              model: torch.nn.Module,
              img: Tensor,
              feats: List[Tensor],
              data_sample: TrackDataSample,
              data_preprocessor: OptConfigType = None,
              rescale: bool = False,
              **kwargs) -> InstanceData:
        """Tracking forward function.

        Args:
            model (nn.Module): MOT model.
            img (Tensor): of shape (T, C, H, W) encoding input image.
                Typically these should be mean centered and std scaled.
                The T denotes the number of key images and usually is 1 in
                SORT method.
            feats (list[Tensor]): Multi level feature maps of `img`.
            data_sample (:obj:`TrackDataSample`): The data sample.
                It includes information such as `pred_det_instances`.
            data_preprocessor (dict or ConfigDict, optional): The pre-process
               config of :class:`TrackDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
            rescale (bool, optional): If True, the bounding boxes should be
                rescaled to fit the original scale of the image. Defaults to
                False.

        Returns:
            :obj:`InstanceData`: Tracking results of the input images.
            Each InstanceData usually contains ``bboxes``, ``labels``,
            ``scores`` and ``instances_id``.
        """
        metainfo = data_sample.metainfo
        bboxes = data_sample.pred_det_instances.bboxes
        labels = data_sample.pred_det_instances.labels
        scores = data_sample.pred_det_instances.scores

        frame_id = metainfo.get('frame_id', -1)
        img_np_clone = np.ascontiguousarray(img[0].cpu().moveaxis(0, -1).numpy(), dtype=np.uint8)
        img_np = img[0].cpu().moveaxis(0, -1).numpy() # BGR

        if not hasattr(self, 'motion'):
            self.motion = KLTTracker((img_np.shape[1], img_np.shape[0]), **flow_cfg)
            self.motion.init(img_np)

        if frame_id == 0:
            self.reset()
        if not hasattr(self, 'kf'):
            self.kf = model.motion

        if self.with_reid:
            if self.reid.get('img_norm_cfg', False):
                img_norm_cfg = dict(
                    mean=data_preprocessor.get('mean', [0.0, 0.0, 0.0]),
                    std=data_preprocessor.get('std', [1.0, 1.0, 1.0]),
                    to_bgr=data_preprocessor.get('rgb_to_bgr', False))
                reid_img = imrenormalize(img, img_norm_cfg,
                                         self.reid['img_norm_cfg'])
            else:
                reid_img = img.clone()

        valid_inds = scores > self.obj_score_thr
        bboxes = bboxes[valid_inds]
        labels = labels[valid_inds]
        scores = scores[valid_inds]

        if self.empty or bboxes.size(0) == 0:
            num_new_tracks = bboxes.size(0)
            ids = torch.arange(
                self.num_tracks,
                self.num_tracks + num_new_tracks,
                dtype=torch.long).to(bboxes.device)
            self.num_tracks += num_new_tracks
            if self.with_reid:
                crops = self.crop_imgs(reid_img, metainfo, bboxes.clone(),
                                       rescale)
                if crops.size(0) > 0:
                    embeds = model.reid(crops, mode='tensor')
                else:
                    embeds = crops.new_zeros((0, model.reid.head.out_channels))
        else:
            ids = torch.full((bboxes.size(0), ), -1,
                             dtype=torch.long).to(bboxes.device)

            active_ids = self.confirmed_ids
            if self.with_reid:
                crops = self.crop_imgs(reid_img, metainfo, bboxes.clone(),
                                       rescale)
                embeds = model.reid(crops, mode='tensor')

                # for crop, embed in zip(crops, embeds):
                #     reid_norm_cfg = self.reid['img_norm_cfg']
                #     self.reid_debugger.update(frame_id, crop, reid_norm_cfg['mean'], reid_norm_cfg['std'], embed)
                # self.reid_debugger.visualize()
                # self.reid_debugger.clear(keep_current=True)

                # reid
                if len(active_ids) > 0:
                    track_embeds = self.get(
                        'embeds',
                        active_ids,
                        self.reid.get('num_samples', None),
                        behavior='mean')
                    reid_dists = torch.cdist(track_embeds, embeds)

                    # support multi-class association
                    track_labels = torch.tensor([
                        self.tracks[id]['labels'][-1] for id in active_ids
                    ]).to(bboxes.device)
                    cate_match = labels[None, :] == track_labels[:, None]
                    cate_cost = (1 - cate_match.int()) * 1e6
                    reid_dists = (reid_dists + cate_cost).cpu().numpy()
                    print(reid_dists)

                    # valid_inds = [list(self.ids).index(_) for _ in active_ids]
                    # reid_dists[~np.isfinite(costs[valid_inds, :])] = np.nan

                    row, col = linear_sum_assignment(reid_dists)
                    for r, c in zip(row, col):
                        dist = reid_dists[r, c]
                        if not np.isfinite(dist):
                            continue
                        if dist <= self.reid['match_score_thr']:
                            ids[c] = active_ids[r]

            active_ids = [
                id for id in self.ids if id not in ids
                # and self.tracks[id].frame_ids[-1] == frame_id - 1
            ]
            if len(active_ids) > 0:
                active_dets = torch.nonzero(ids == -1).squeeze(1)

                motion_tracks: List[Track] = []
                for id in active_ids:
                    assert self.tracks[id]['last_frame'] == frame_id - 1
                    motion_tracks.append(self.tracks[id]['motion_track'])
                track_bboxes, homography = self.motion.predict(img_np, motion_tracks)

                active_ids = list(track_bboxes.keys())
                if len(active_ids) > 0:
                    for id in active_ids:
                        self.tracks[id]['motion_track'].tlbr = track_bboxes[id]
                        self.tracks[id]['last_frame'] = frame_id

                    track_bboxes = [track_bboxes[id] for id in active_ids]
                    track_bboxes = np.stack(track_bboxes, axis=0)
                    track_bboxes = torch.from_numpy(track_bboxes).to(bboxes)
                    ious = bbox_overlaps(track_bboxes, bboxes[active_dets])

                    # support multi-class association
                    track_labels = torch.tensor([
                        self.tracks[id]['labels'][-1] for id in active_ids
                    ]).to(bboxes.device)
                    cate_match = labels[None, active_dets] == track_labels[:, None]
                    cate_cost = (1 - cate_match.int()) * 1e6

                    dists = (1 - ious + cate_cost).cpu().numpy()

                    row, col = linear_sum_assignment(dists)
                    for r, c in zip(row, col):
                        dist = dists[r, c]
                        if dist < 1 - self.match_iou_thr:
                            ids[active_dets[c]] = active_ids[r]

            new_track_inds = ids == -1
            ids[new_track_inds] = torch.arange(
                self.num_tracks,
                self.num_tracks + new_track_inds.sum(),
                dtype=torch.long).to(bboxes.device)
            self.num_tracks += new_track_inds.sum()

        self.update(
            ids=ids,
            bboxes=bboxes,
            scores=scores,
            labels=labels,
            embeds=embeds if self.with_reid else None,
            frame_ids=frame_id)
        
        # Save the motion boxes
        motion_bboxes = bboxes[:, :4].clone()
        if rescale:
            factor_x, factor_y = metainfo['scale_factor']
            motion_bboxes = motion_bboxes * torch.tensor(
                [factor_x, factor_y, factor_x, factor_y]).to(bboxes.device)

        for id, bbox, score in zip(ids.tolist(), motion_bboxes, scores.tolist()):
            if id in self.ids:
                if 'motion_track' in self.tracks[id]:
                    self.tracks[id]['motion_track'].tlbr = bbox.cpu().numpy()
                else:
                    self.tracks[id]['motion_track'] = Track(bbox.cpu().numpy(), score, id)
                self.tracks[id]['last_frame'] = frame_id

        # update pred_track_instances
        pred_track_instances = InstanceData()
        # pred_track_instances.bboxes = bboxes
        # pred_track_instances.labels = labels
        # pred_track_instances.scores = scores
        # pred_track_instances.instances_id = ids
        try:
            pred_track_instances.bboxes = torch.cat((track_bboxes.to(bboxes), bboxes))
            pred_track_instances.labels = torch.cat((torch.zeros(track_bboxes.shape[0]).to(labels), labels))
            pred_track_instances.scores = torch.cat((torch.zeros(track_bboxes.shape[0]).to(scores), scores))
            pred_track_instances.instances_id = torch.cat((torch.tensor(active_ids).to(ids), ids))
        except Exception as e:
            pred_track_instances.bboxes = bboxes
            pred_track_instances.labels = labels
            pred_track_instances.scores = scores
            pred_track_instances.instances_id = ids
            print(e)

        cv2.imwrite(f'debug/{frame_id}.jpg', img_np_clone)
        return pred_track_instances
