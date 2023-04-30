# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple

import lap
import numpy as np
import torch
from mmdet.structures.bbox import bbox_overlaps
from mmengine.structures import InstanceData

from mmtrack.registry import MODELS
from mmtrack.structures import TrackDataSample
from mmtrack.structures.bbox import bbox_cxcyah_to_xyxy, bbox_xyxy_to_cxcyah
from mmtrack.utils import OptConfigType, imrenormalize
from .base_tracker import BaseTracker


@MODELS.register_module()
class DeepByteTracker(BaseTracker):
    """Tracker for ByteTrack.

    Args:
        obj_score_thrs (dict): Detection score threshold for matching objects.
            - high (float): Threshold of the first matching. Defaults to 0.6.
            - low (float): Threshold of the second matching. Defaults to 0.1.
        init_track_thr (float): Detection score threshold for initializing a
            new tracklet. Defaults to 0.7.
        weight_iou_with_det_scores (bool): Whether using detection scores to
            weight IOU which is used for matching. Defaults to True.
        match_iou_thrs (dict): IOU distance threshold for matching between two
            frames.
            - high (float): Threshold of the first matching. Defaults to 0.1.
            - low (float): Threshold of the second matching. Defaults to 0.5.
            - tentative (float): Threshold of the matching for tentative
                tracklets. Defaults to 0.3.
        num_tentatives (int, optional): Number of continuous frames to confirm
            a track. Defaults to 3.
    """

    def __init__(self,
                 obj_score_thrs: dict = dict(high=0.6, low=0.1),
                 init_track_thr: float = 0.7,
                 reid: dict = dict(
                     num_samples=10,
                     img_scale=(256, 128),
                     img_norm_cfg=None),
                 weight_iou_with_det_scores: bool = True,
                 match_iou_thrs: dict = dict(high=0.1, low=0.5, tentative=0.3),
                 weight_assoc_embed: float = 0.75,
                 embed_cost_diff_limit: float = 0.5,
                 embed_momentum_factor: float = 0.5,
                 update_embed_thr: float = 0.5,
                 num_tentatives: int = 3,
                 **kwargs):
        super().__init__(**kwargs)
        self.obj_score_thrs = obj_score_thrs
        self.init_track_thr = init_track_thr

        self.reid = reid

        self.weight_iou_with_det_scores = weight_iou_with_det_scores
        self.match_iou_thrs = match_iou_thrs

        self.num_tentatives = num_tentatives

        assert self.momentums is None, 'Does not support momentums.'
        self.momentums = dict(embeds=embed_momentum_factor)
        self.update_embed_thr = update_embed_thr
        self.weight_assoc_embed = weight_assoc_embed
        self.embed_cost_diff_limit = embed_cost_diff_limit

    @property
    def confirmed_ids(self) -> List:
        """Confirmed ids in the tracker."""
        ids = [id for id, track in self.tracks.items() if not track.tentative]
        return ids

    @property
    def unconfirmed_ids(self) -> List:
        """Unconfirmed ids in the tracker."""
        ids = [id for id, track in self.tracks.items() if track.tentative]
        return ids

    def init_track(self, id: int, obj: Tuple[torch.Tensor]) -> None:
        """Initialize a track."""
        super().init_track(id, obj)
        if self.tracks[id].frame_ids[-1] == 0:
            self.tracks[id].tentative = False
        else:
            self.tracks[id].tentative = True
        bbox = bbox_xyxy_to_cxcyah(self.tracks[id].bboxes[-1])  # size = (1, 4)
        assert bbox.ndim == 2 and bbox.shape[0] == 1
        bbox = bbox.squeeze(0).cpu().numpy()
        self.tracks[id].mean, self.tracks[id].covariance = self.kf.initiate(
            bbox)

    def update_track(self, id: int, obj: Tuple[torch.Tensor]) -> None:
        """Update a track."""
        for k, v in zip(self.memo_items, obj):
            v = v[None]
            if k == 'embeds':
                assert k in self.momentums
                m = self.momentums[k]
                scores = obj[self.memo_items.index('scores')]

                trust = (scores - self.update_embed_thr) / (
                    1 - self.update_embed_thr)
                m = m + (1 - m) * (1 - trust)
                self.tracks[id][k] = m * self.tracks[id][k] + (1 - m) * v
            else:
                self.tracks[id][k].append(v)

        if self.tracks[id].tentative:
            if len(self.tracks[id]['bboxes']) >= self.num_tentatives:
                self.tracks[id].tentative = False
        bbox = bbox_xyxy_to_cxcyah(self.tracks[id].bboxes[-1])  # size = (1, 4)
        assert bbox.ndim == 2 and bbox.shape[0] == 1
        bbox = bbox.squeeze(0).cpu().numpy()
        track_label = self.tracks[id]['labels'][-1]
        label_idx = self.memo_items.index('labels')
        obj_label = obj[label_idx]
        assert obj_label == track_label
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

    def compute_aw_matrix(self,
                          embed_costs: torch.Tensor,
                          w_assoc_embed: float,
                          max_diff: float = 0.5):
        w_emb = torch.full_like(embed_costs, w_assoc_embed)

        # Needs two columns at least to make sense to boost
        if embed_costs.size(1) >= 2:
            for i in range(embed_costs.size(0)):
                inds = torch.argsort(-embed_costs[i])
                # Row weight is difference between top / second top
                row_weight = 1 - max((embed_costs[i, inds[1]] / embed_costs[i, inds[0]]) - max_diff, 0) / (1 - max_diff)
                # Add to row
                w_emb[i] *= row_weight

        if embed_costs.size(0) >= 2:
            for j in range(embed_costs.size(1)):
                inds = torch.argsort(embed_costs[:, j])
                # Row weight is difference between top / second top
                col_weight = 1 - max((embed_costs[inds[1], j] / embed_costs[inds[0], j]) - max_diff, 0) / (1 - max_diff)
                # Add to row
                w_emb[:, j] *= col_weight

        return w_emb

    def assign_ids(
            self,
            ids: List[int],
            det_bboxes: torch.Tensor,
            det_labels: torch.Tensor,
            det_scores: torch.Tensor,
            det_embeds: Optional[torch.Tensor] = None,
            weight_iou_with_det_scores: Optional[bool] = False,
            match_iou_thr: Optional[float] = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Assign ids.

        Args:
            ids (list[int]): Tracking ids.
            det_bboxes (Tensor): of shape (N, 4)
            det_labels (Tensor): of shape (N,)
            det_scores (Tensor): of shape (N,)
            det_embeds (Tensor, optional): Appearance embeddings of shape (N, C).
            weight_iou_with_det_scores (bool, optional): Whether using
                detection scores to weight IOU which is used for matching.
                Defaults to False.
            match_iou_thr (float, optional): Matching threshold.
                Defaults to 0.5.

        Returns:
            tuple(np.ndarray, np.ndarray): The assigning ids.
        """
        # get track_bboxes
        track_bboxes = np.zeros((0, 4))
        for id in ids:
            track_bboxes = np.concatenate(
                (track_bboxes, self.tracks[id].mean[:4][None]), axis=0)
        track_bboxes = torch.from_numpy(track_bboxes).to(det_bboxes)
        track_bboxes = bbox_cxcyah_to_xyxy(track_bboxes)

        # compute distance
        ious = bbox_overlaps(track_bboxes, det_bboxes)

        if min(ious.shape) > 0:
            if weight_iou_with_det_scores:
                ious *= det_scores

            if det_embeds is not None and len(ids) > 0:
                track_embeds = torch.cat([
                    self.tracks[id]['embeds'] for id in ids
                ]).to(det_embeds.device)
                embed_costs = torch.cosine_similarity(track_embeds, det_embeds)
                weighted_matrix = self.compute_aw_matrix(
                    embed_costs, 
                    self.weight_assoc_embed, 
                    self.embed_cost_diff_limit)
                embed_costs *= weighted_matrix

                embed_costs = embed_costs.cpu().numpy()
            else:
                embed_costs = 0

            # support multi-class association
            track_labels = torch.tensor([
                self.tracks[id]['labels'][-1] for id in ids
            ]).to(det_bboxes.device)
            cate_match = det_labels[None, :] == track_labels[:, None]
            # to avoid det and track of different categories are matched
            cate_cost = (1 - cate_match.int()) * 1e6

            final_costs = -(ious + embed_costs) + cate_cost.cpu().numpy()
            cost, row, col = lap.lapjv(final_costs)

            # filter out matches with low IoU
            for track_idx in range(len(row)):
                if track_idx != -1:
                    matched_det_idx = row[track_idx]
                    if ious[track_idx, matched_det_idx] < match_iou_thr:
                        row[track_idx] = -1
                        col[matched_det_idx] = -1
        else:
            row = np.zeros(len(ids)).astype(np.int32) - 1
            col = np.zeros(len(det_bboxes)).astype(np.int32) - 1

        return row, col

    def track(self, 
              model: torch.nn.Module, 
              img: torch.Tensor,
              feats: List[torch.Tensor], 
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
                ByteTrack method.
            feats (list[Tensor]): Multi level feature maps of `img`.
            data_sample (:obj:`TrackDataSample`): The data sample.
                It includes information such as `pred_det_instances`.

        Returns:
            :obj:`InstanceData`: Tracking results of the input images.
            Each InstanceData usually contains ``bboxes``, ``labels``,
            ``scores`` and ``instances_id``.
        """
        assert self.with_reid, 'ReID is required for tracking.'

        metainfo = data_sample.metainfo
        bboxes = data_sample.pred_det_instances.bboxes
        labels = data_sample.pred_det_instances.labels
        scores = data_sample.pred_det_instances.scores

        frame_id = metainfo.get('frame_id', -1)
        if frame_id == 0:
            self.reset()
        if not hasattr(self, 'kf'):
            self.kf = model.motion

        if self.reid.get('img_norm_cfg', False):
            img_norm_cfg = dict(
                mean=data_preprocessor.get('mean', [0.0, 0.0, 0.0]),
                std=data_preprocessor.get('std', [1.0, 1.0, 1.0]),
                to_bgr=data_preprocessor.get('rgb_to_bgr', False))
            reid_img = imrenormalize(img, img_norm_cfg,
                                        self.reid['img_norm_cfg'])
        else:
            reid_img = img.clone()

        if self.empty or bboxes.size(0) == 0:
            valid_inds = scores > self.init_track_thr
            scores = scores[valid_inds]
            bboxes = bboxes[valid_inds]
            labels = labels[valid_inds]
            num_new_tracks = bboxes.size(0)
            ids = torch.arange(self.num_tracks,
                               self.num_tracks + num_new_tracks).to(labels)
            self.num_tracks += num_new_tracks

            crops = self.crop_imgs(reid_img, metainfo, bboxes.clone(),
                                    rescale)
            if crops.size(0) > 0:
                embeds = model.reid(crops, mode='tensor')
            else:
                embeds = crops.new_zeros((0, model.reid.head.out_channels))
        else:
            # 0. init
            ids = torch.full((bboxes.size(0), ),
                             -1,
                             dtype=labels.dtype,
                             device=labels.device)
            
            crops = self.crop_imgs(reid_img, metainfo, bboxes.clone(), rescale)
            embeds = model.reid(crops, mode='tensor')

            # get the detection bboxes for the first association
            first_det_inds = scores > self.obj_score_thrs['high']
            first_det_bboxes = bboxes[first_det_inds]
            first_det_labels = labels[first_det_inds]
            first_det_scores = scores[first_det_inds]
            first_det_ids = ids[first_det_inds]
            first_det_embeds = embeds[first_det_inds]

            # get the detection bboxes for the second association
            second_det_inds = (~first_det_inds) & (
                scores > self.obj_score_thrs['low'])
            second_det_bboxes = bboxes[second_det_inds]
            second_det_labels = labels[second_det_inds]
            second_det_scores = scores[second_det_inds]
            second_det_ids = ids[second_det_inds]

            # 1. use Kalman Filter to predict current location
            for id in self.confirmed_ids:
                # track is lost in previous frame
                if self.tracks[id].frame_ids[-1] != frame_id - 1:
                    self.tracks[id].mean[7] = 0
                (self.tracks[id].mean,
                 self.tracks[id].covariance) = self.kf.predict(
                     self.tracks[id].mean, self.tracks[id].covariance)

            # 2. first match
            first_match_track_inds, first_match_det_inds = self.assign_ids(
                self.confirmed_ids, 
                first_det_bboxes, 
                first_det_labels,
                first_det_scores, 
                first_det_embeds,
                self.weight_iou_with_det_scores,
                self.match_iou_thrs['high'])
            # '-1' mean a detection box is not matched with tracklets in
            # previous frame
            valid = first_match_det_inds > -1
            first_det_ids[valid] = torch.tensor(
                self.confirmed_ids)[first_match_det_inds[valid]].to(labels)

            first_match_det_bboxes = first_det_bboxes[valid]
            first_match_det_labels = first_det_labels[valid]
            first_match_det_scores = first_det_scores[valid]
            first_match_det_ids = first_det_ids[valid]
            assert (first_match_det_ids > -1).all()

            first_unmatch_det_bboxes = first_det_bboxes[~valid]
            first_unmatch_det_labels = first_det_labels[~valid]
            first_unmatch_det_scores = first_det_scores[~valid]
            first_unmatch_det_ids = first_det_ids[~valid]
            assert (first_unmatch_det_ids == -1).all()

            # 3. use unmatched detection bboxes from the first match to match
            # the unconfirmed tracks
            (tentative_match_track_inds,
             tentative_match_det_inds) = self.assign_ids(
                 self.unconfirmed_ids, 
                 first_unmatch_det_bboxes,
                 first_unmatch_det_labels, 
                 first_unmatch_det_scores,
                 None,
                 self.weight_iou_with_det_scores,
                 self.match_iou_thrs['tentative'])
            valid = tentative_match_det_inds > -1
            first_unmatch_det_ids[valid] = torch.tensor(self.unconfirmed_ids)[
                tentative_match_det_inds[valid]].to(labels)

            # 4. second match for unmatched tracks from the first match
            first_unmatch_track_ids = []
            for i, id in enumerate(self.confirmed_ids):
                # tracklet is not matched in the first match
                case_1 = first_match_track_inds[i] == -1
                # tracklet is not lost in the previous frame
                case_2 = self.tracks[id].frame_ids[-1] == frame_id - 1
                if case_1 and case_2:
                    first_unmatch_track_ids.append(id)

            second_match_track_inds, second_match_det_inds = self.assign_ids(
                first_unmatch_track_ids, 
                second_det_bboxes, 
                second_det_labels,
                second_det_scores,
                None, 
                False,
                self.match_iou_thrs['low'])
            valid = second_match_det_inds > -1
            second_det_ids[valid] = torch.tensor(first_unmatch_track_ids)[
                second_match_det_inds[valid]].to(ids)

            # 5. gather all matched detection bboxes from step 2-4
            # we only keep matched detection bboxes in second match, which
            # means the id != -1
            valid = second_det_ids > -1
            bboxes = torch.cat(
                (first_match_det_bboxes, first_unmatch_det_bboxes), dim=0)
            bboxes = torch.cat((bboxes, second_det_bboxes[valid]), dim=0)

            labels = torch.cat(
                (first_match_det_labels, first_unmatch_det_labels), dim=0)
            labels = torch.cat((labels, second_det_labels[valid]), dim=0)

            scores = torch.cat(
                (first_match_det_scores, first_unmatch_det_scores), dim=0)
            scores = torch.cat((scores, second_det_scores[valid]), dim=0)

            ids = torch.cat((first_match_det_ids, first_unmatch_det_ids),
                            dim=0)
            ids = torch.cat((ids, second_det_ids[valid]), dim=0)

            # 6. assign new ids
            new_track_inds = ids == -1
            ids[new_track_inds] = torch.arange(
                self.num_tracks,
                self.num_tracks + new_track_inds.sum()).to(labels)
            self.num_tracks += new_track_inds.sum()

        self.update(
            ids=ids,
            bboxes=bboxes,
            scores=scores,
            labels=labels,
            embeds=embeds,
            frame_ids=frame_id)

        # update pred_track_instances
        pred_track_instances = InstanceData()
        pred_track_instances.bboxes = bboxes
        pred_track_instances.labels = labels
        pred_track_instances.scores = scores
        pred_track_instances.instances_id = ids

        return pred_track_instances
