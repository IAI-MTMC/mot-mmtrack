# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import numpy as np
import torch
from mmdet.structures.bbox import bbox_overlaps
from mmengine.structures import InstanceData
from motmetrics.lap import linear_sum_assignment
from torch import Tensor

from mmtrack.models.pose import PosePipeline
from mmtrack.registry import MODELS
from mmtrack.structures import TrackDataSample
from mmtrack.structures.bbox import bbox_xyxy_to_cxcyah, expanse_bbox
from mmtrack.structures.bbox.transforms import bbox_cxcyah_to_xyxy
from mmtrack.utils import OptConfigType, imrenormalize
from .base_tracker import BaseTracker


@MODELS.register_module()
class MySORTTracker(BaseTracker):
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
                     pose=False,
                     reid=True,
                     num_samples=10,
                     img_scale=(256, 128),
                     img_norm_cfg=None,
                     match_score_thr=2.0,
                     return_embedding=False),
                 match_iou_thr: float = 0.7,
                 num_tentatives: int = 3,
                 biou: int = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.obj_score_thr = obj_score_thr
        self.reid = reid
        self.match_iou_thr = match_iou_thr
        self.num_tentatives = num_tentatives
        self.biou = biou

        self.pose_pipeline = PosePipeline()

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

        print()
        print('bboxes:', bboxes.shape[0], '---', 'scores:', scores)
        print('frame_id:', frame_id, '---', 'reid_image:', reid_img.shape)

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
                    embeds = torch.zeros((crops.shape[0], 0),
                                         device=crops.device)
                    if self.reid.get('reid', None):
                        embeds_reid = model.reid(
                            crops, mode='tensor', frame_id=frame_id)
                        embeds = torch.cat((embeds, embeds_reid), dim=1)
                    if self.reid.get('pose', None):
                        pose_results, pose_embedded = self.pose_pipeline.get_pose_embedded(
                            bboxes.clone(), scores.clone(), metainfo, reid_img,
                            crops, model.pose)
                        embeds = torch.cat(
                            (embeds, pose_embedded.to(embeds.device)), dim=1)
                else:
                    embeds = torch.zeros((0, 0), device=crops.device)
                    if self.reid.get('reid', None):
                        embeds_reid = crops.new_zeros(
                            (0, model.reid.head.out_channels))
                        embeds = torch.cat((embeds, embeds_reid), dim=1)
                    if self.reid.get('pose', None):
                        pose_embedded = crops.new_zeros(
                            (0, self.pose_pipeline.embedding_size))
                        pose_results = []
                        embeds = torch.cat(
                            (embeds, pose_embedded.to(embeds.device)), dim=1)
        else:
            ids = torch.full((bboxes.size(0), ), -1,
                             dtype=torch.long).to(bboxes.device)

            # motion
            if model.with_motion:
                # self.tracks, costs = model.motion.track(
                #     self.tracks, bbox_xyxy_to_cxcyah(bboxes))

                kf_bboxes = []

                for id, track in self.tracks.items():
                    if track.frame_ids[-1] != frame_id - 1:
                        track.mean[7] = 0

                    track.mean, track.covariance = model.motion.predict(
                        track.mean, track.covariance)

                    kf_bboxes.append(torch.from_numpy(track.mean[:4]))

                if len(kf_bboxes) > 0:
                    kf_bboxes = torch.stack(kf_bboxes, dim=0).to(bboxes)
                    kf_bboxes = bbox_cxcyah_to_xyxy(kf_bboxes)
                    kf_ids = torch.tensor(
                        self.confirmed_ids,
                        dtype=torch.long,
                        device=bboxes.device)
                print('kf_bboxes:', len(kf_bboxes))

            active_ids = self.confirmed_ids

            if self.with_reid:
                crops = self.crop_imgs(reid_img, metainfo, bboxes.clone(),
                                       rescale)

                embeds = torch.zeros((crops.shape[0], 0), device=crops.device)
                if self.reid.get('reid', None):
                    embeds_reid = model.reid(
                        crops, mode='tensor', frame_id=frame_id)
                    embeds = torch.cat((embeds, embeds_reid), dim=1)

                if self.reid.get('pose', None):
                    pose_results, pose_embedded = self.pose_pipeline.get_pose_embedded(
                        bboxes.clone(), scores.clone(), metainfo, reid_img,
                        crops, model.pose)
                    embeds = torch.cat(
                        (embeds, pose_embedded.to(embeds.device)), dim=1)

                print('reid_mtaching')
                print('activate_ids:', active_ids, '---', 'self.ids:',
                      self.ids)

                if len(active_ids) > 0:
                    track_embeds = self.get(
                        'embeds',
                        active_ids,
                        self.reid.get('num_samples', None),
                        behavior='mean')

                    print('track_embeds:', track_embeds.shape, '---',
                          'embeds:', embeds.shape)
                    reid_dists = torch.cdist(track_embeds, embeds)

                    print('reid_dist')
                    print(reid_dists)

                    # support multi-class association
                    track_labels = torch.tensor([
                        self.tracks[id]['labels'][-1] for id in active_ids
                    ]).to(bboxes.device)
                    cate_match = labels[None, :] == track_labels[:, None]
                    cate_cost = (1 - cate_match.int()) * 1e6
                    reid_dists = (reid_dists + cate_cost).cpu().numpy()

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
                # track_bboxes = self.get('bboxes', active_ids)
                print('iou_mtaching')
                print('active_ids:', active_ids, '---', 'active_dets:',
                      active_dets)

                track_bboxes = np.zeros((0, 4))
                for id in active_ids:
                    track_bboxes = np.concatenate(
                        (track_bboxes, self.tracks[id].mean[:4][None]), axis=0)
                track_bboxes = torch.from_numpy(track_bboxes).to(bboxes.device)
                track_bboxes = bbox_cxcyah_to_xyxy(track_bboxes)

                if self.biou is not None:
                    ious = bbox_overlaps(
                        expanse_bbox(track_bboxes, self.biou),
                        expanse_bbox(bboxes[active_dets], self.biou))

                else:
                    ious = bbox_overlaps(track_bboxes, bboxes[active_dets])

                # support multi-class association
                track_labels = torch.tensor([
                    self.tracks[id]['labels'][-1] for id in active_ids
                ]).to(bboxes.device)
                cate_match = labels[None, active_dets] == track_labels[:, None]
                cate_cost = (1 - cate_match.int()) * 1e6

                dists = (1 - ious + cate_cost).cpu().numpy()

                print('bbox dists:')
                print(dists)
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

        print('match id:', ids)

        # update pred_track_instances
        # try:
        #     print('kf: ', kf_bboxes.shape, kf_ids.shape)
        #     pred_track_instances = InstanceData()
        #     pred_track_instances.bboxes = torch.cat((kf_bboxes, bboxes))
        #     pred_track_instances.labels = torch.cat(
        #         (torch.zeros(kf_bboxes.size(0)).to(labels), labels))
        #     pred_track_instances.scores = torch.cat(
        #         (torch.zeros(kf_bboxes.size(0)).to(scores), scores))
        #     pred_track_instances.instances_id = torch.cat(
        #         (kf_ids.to(ids), ids))
        #     print('done')
        # except:
        #     print('exception')
        #     pred_track_instances = InstanceData()
        #     pred_track_instances.bboxes = bboxes
        #     pred_track_instances.labels zzzz= labels
        #     pred_track_instances.scores = scores
        #     pred_track_instances.instances_id = ids

        pred_track_instances = InstanceData()
        pred_track_instances.bboxes = bboxes
        pred_track_instances.labels = labels
        pred_track_instances.scores = scores
        pred_track_instances.instances_id = ids
        if self.with_reid and self.reid.get('pose', False):
            pred_track_instances.pose = pose_results
        if self.with_reid and self.reid.get('return_embedding', False):
            pred_track_instances.embeds = embeds

        return pred_track_instances
