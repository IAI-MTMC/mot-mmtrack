from typing import List

import lap
import numpy as np
import torch
from torch import Tensor
from mmengine.structures import InstanceData
from mmdet.structures.bbox import bbox_overlaps

from mmtrack.structures.bbox import bbox_cxcyah_to_xyxy, bbox_xyxy_to_cxcyah
from mmtrack.structures import TrackDataSample
from mmtrack.registry import MODELS
from mmtrack.utils import OptConfigType
from mmtrack.utils.image import imrenormalize
from .ocsort_tracker import OCSORTTracker


@MODELS.register_module()
class DeepOCSORTTracker(OCSORTTracker):
    """DeepOCSORTTracker."""
    def __init__(self,
                 det_momentum_thr: float = 0.8,
                 embed_momentum_factor: float = 0.95,
                 **kwargs):
        super().__init__(**kwargs)
        self.det_momentum_thr = det_momentum_thr

        assert self.momentums is None, "DeepOCSORT does not support momentums."
        self.embed_momentum_factor = embed_momentum_factor
        self.w_assoc_embed = 0.75
    
    def init_track(self, id, obj):
        """Initialize a track."""
        super().init_track(id, obj)

        self.tracks[id].embed_momentum = self.tracks[id]['embeds'][0]
    
    def update_track(self, id, obj):
        """Update a track."""
        super().update_track(id, obj)
        # update momentum factor of embeds
        obj_score = self.tracks[id]['scores'][-1]
        det_conf = 1 - (obj_score - self.det_momentum_thr) / (1 - self.det_momentum_thr)
        embed_mometum_factor = self.embed_momentum_factor + \
            (1 - self.embed_momentum_factor) * det_conf
        embed_mometum_factor = min(embed_mometum_factor, 1) # clip to 1

        self.tracks[id].embed_momentum = embed_mometum_factor * self.tracks[id].embed_momentum + \
            (1 - embed_mometum_factor) * self.tracks[id]['embeds'][-1]
        
    def compute_adaptive_weighted_matrix(self, embed_costs, w_assoc_embed, max_diff=0.5):
        w_emb = torch.full_like(embed_costs, w_assoc_embed)
        w_emb_bonus = torch.full_like(embed_costs, 0.0)

        # Needs two columns at least to make sense to boost
        if embed_costs.size(1) >= 2:
            for i in range(embed_costs.size(0)):
                inds = torch.argsort(embed_costs[i], descending=True)
                # Row weight is difference between top / second top
                row_weight = min(embed_costs[i, inds[0]] - embed_costs[i, inds[1]], max_diff)
                # Add to row
                w_emb_bonus[i] += row_weight / 2
            
        if embed_costs.size(0) >= 2:
            for j in range(embed_costs.size(1)):
                inds = torch.argsort(embed_costs[:, j], descending=True)
                # Row weight is difference between top / second top
                col_weight = min(embed_costs[inds[0], j] - embed_costs[inds[1], j], max_diff)
                # Add to row
                w_emb_bonus[:, j] += col_weight / 2
        
        return w_emb + w_emb_bonus
        
    def ocm_assign_ids(self,
                       ids,
                       det_bboxes,
                       det_scores,
                       det_embeds=None,
                       weight_iou_with_det_scores=False,
                       match_iou_thr=0.5):
        """Apply Observation-Centric Momentum (OCM) to assign ids.

        OCM adds movement direction consistency into the association cost
        matrix. This term requires no additional assumption but from the
        same linear motion assumption as the canonical Kalman Filter in SORT.

        Args:
            ids (list[int]): Tracking ids.
            det_bboxes (Tensor): of shape (N, 4)
            det_scores: (Tensor): of shape (N, )
            det_embeds: (Tensor, optional): of shape (N, embed_dim)
            weight_iou_with_det_scores (bool, optional): Whether using
                detection scores to weight IOU which is used for matching.
                Defaults to False.
            match_iou_thr (float, optional): Matching threshold.
                Defaults to 0.5.

        Returns:
            tuple(int): The assigning ids.

        OC-SORT uses velocity consistency besides IoU for association
        """
        # get track_bboxes
        track_bboxes = np.zeros((0, 4))
        for id in ids:
            track_bboxes = np.concatenate(
                (track_bboxes, self.tracks[id].mean[:4][None]), axis=0)
        track_bboxes = torch.from_numpy(track_bboxes).to(det_bboxes.device)
        track_bboxes = bbox_cxcyah_to_xyxy(track_bboxes)

        # compute distance
        ious = bbox_overlaps(track_bboxes, det_bboxes)
        if weight_iou_with_det_scores:
            ious *= det_scores[None]
        dists = (1 - ious).cpu().numpy()

        if len(ids) > 0 and len(det_bboxes) > 0:
            track_velocities = torch.stack(
                [self.tracks[id].velocity for id in ids]).to(det_bboxes.device)
            k_step_observations = torch.stack([
                self.k_step_observation(self.tracks[id]) for id in ids
            ]).to(det_bboxes.device)
            # valid1: if the track has previous observations to estimate speed
            # valid2: if the associated observation k steps ago is a detection
            valid1 = track_velocities.sum(dim=1) != -2
            valid2 = k_step_observations.sum(dim=1) != -4
            valid = valid1 & valid2

            vel_to_match = self.vel_direction_batch(k_step_observations[:, :4],
                                                    det_bboxes)
            track_velocities = track_velocities[:, None, :].repeat(
                1, det_bboxes.shape[0], 1)

            angle_cos = (vel_to_match * track_velocities).sum(dim=-1)
            angle_cos = torch.clamp(angle_cos, min=-1, max=1)
            angle = torch.acos(angle_cos)  # [0, pi]
            norm_angle = (angle - np.pi / 2.) / np.pi  # [-0.5, 0.5]
            valid_matrix = valid[:, None].int().repeat(1, det_bboxes.shape[0])
            # set non-valid entries 0
            valid_norm_angle = norm_angle * valid_matrix

            dists += valid_norm_angle.cpu().numpy() * self.vel_consist_weight

            # compute appearance distance
            if det_embeds is not None:
                track_embeds = torch.stack(
                    [self.tracks[id].embed_momentum for id in ids]).to(
                        det_embeds.device)
                embed_dists = torch.cdist(track_embeds, det_embeds)
                weighted_matrix = self.compute_adaptive_weighted_matrix(
                    embed_dists,
                    self.w_assoc_embed)
                embed_dists *= weighted_matrix

                dists += embed_dists.cpu().numpy()

        # bipartite match
        if dists.size > 0:
            cost, row, col = lap.lapjv(
                dists, extend_cost=True, cost_limit=1 - match_iou_thr)
        else:
            row = np.zeros(len(ids)).astype(np.int32) - 1
            col = np.zeros(len(det_bboxes)).astype(np.int32) - 1
        return row, col

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
        
        if self.empty or bboxes.size(0) == 0:
            valid_inds = scores > self.init_track_thr
            bboxes = bboxes[valid_inds]
            labels = labels[valid_inds]
            scores = scores[valid_inds]
            
            num_new_tracks = bboxes.size(0)
            ids = torch.arange(
                self.num_tracks,
                self.num_tracks + num_new_tracks,
                dtype=torch.long).to(bboxes.device)
            self.num_tracks += num_new_tracks
            if self.with_reid:
                det_crops = self.crop_imgs(reid_img, metainfo, bboxes.clone(),
                                       rescale)
                if det_crops.size(0) > 0:
                    det_embeds = model.reid(det_crops, mode='tensor')
                else:
                    det_embeds = det_crops.new_zeros((0, model.reid.head.out_channels))
        else:
            # 0. init
            ids = torch.full((bboxes.size(0), ), -1,
                             dtype=torch.long).to(bboxes.device)
            
            # get the detection bboxes for the first association
            det_inds = scores > self.obj_score_thr
            det_bboxes = bboxes[det_inds]
            det_labels = labels[det_inds]
            det_scores = scores[det_inds]
            det_ids = ids[det_inds]
            
            # 1. predict by Kalman Filter
            for id in self.confirmed_ids:
                # track is lost in previous frame
                if self.tracks[id].frame_ids[-1] != frame_id - 1:
                    self.tracks[id].mean[7] = 0
                if self.tracks[id].tracked:
                    self.tracks[id].saved_attr.mean = self.tracks[id].mean
                    self.tracks[id].saved_attr.covariance = self.tracks[
                        id].covariance
                (self.tracks[id].mean,
                 self.tracks[id].covariance) = self.kf.predict(
                     self.tracks[id].mean, self.tracks[id].covariance)

            if self.with_reid:
                det_crops = self.crop_imgs(reid_img, metainfo, det_bboxes.clone(),
                                           rescale)
                det_embeds = model.reid(det_crops, mode='tensor')
            else:
                det_embeds = None

            # 2. match detections and tracks' predicted locations
            match_track_inds, raw_match_det_inds = self.ocm_assign_ids(
                self.confirmed_ids,
                det_bboxes,
                det_scores,
                det_embeds,
                self.weight_iou_with_det_scores,
                self.match_iou_thr)
            # '-1' mean a detection box is not matched with tracklets in
            # previous frame
            valid = raw_match_det_inds > -1
            det_ids[valid] = torch.tensor(
                self.confirmed_ids)[raw_match_det_inds[valid]].to(labels)

            match_det_bboxes = det_bboxes[valid]
            match_det_labels = det_labels[valid]
            match_det_scores = det_scores[valid]
            match_det_ids = det_ids[valid]
            assert (match_det_ids > -1).all()

            # unmatched tracks and detections
            unmatch_det_bboxes = det_bboxes[~valid]
            unmatch_det_labels = det_labels[~valid]
            unmatch_det_scores = det_scores[~valid]
            unmatch_det_ids = det_ids[~valid]
            assert (unmatch_det_ids == -1).all()

            # 3. use unmatched detection bboxes from the first match to match
            # the unconfirmed tracks
            (tentative_match_track_inds,
             tentative_match_det_inds) = self.ocm_assign_ids(
                 self.unconfirmed_ids,
                 unmatch_det_bboxes,
                 unmatch_det_scores,
                 None,
                 self.weight_iou_with_det_scores, 
                 self.match_iou_thr)
            valid = tentative_match_det_inds > -1
            unmatch_det_ids[valid] = torch.tensor(self.unconfirmed_ids)[
                tentative_match_det_inds[valid]].to(labels)

            match_det_bboxes = torch.cat(
                (match_det_bboxes, unmatch_det_bboxes[valid]), dim=0)
            match_det_labels = torch.cat(
                (match_det_labels, unmatch_det_labels[valid]), dim=0)
            match_det_scores = torch.cat(
                (match_det_scores, unmatch_det_scores[valid]), dim=0)
            match_det_ids = torch.cat((match_det_ids, unmatch_det_ids[valid]),
                                      dim=0)
            assert (match_det_ids > -1).all()

            unmatch_det_bboxes = unmatch_det_bboxes[~valid]
            unmatch_det_labels = unmatch_det_labels[~valid]
            unmatch_det_scores = unmatch_det_scores[~valid]
            unmatch_det_ids = unmatch_det_ids[~valid]
            assert (unmatch_det_ids == -1).all()

            all_track_ids = [id for id, _ in self.tracks.items()]
            unmatched_track_inds = torch.tensor(
                [ind for ind in all_track_ids if ind not in match_det_ids])

            if len(unmatched_track_inds) > 0:
                # 4. still some tracks not associated yet, perform OCR
                last_observations = []
                for id in unmatched_track_inds:
                    last_box = self.last_obs(self.tracks[id.item()])
                    last_observations.append(last_box)
                last_observations = torch.stack(last_observations)

                remain_det_ids = torch.full((unmatch_det_bboxes.size(0), ),
                                            -1,
                                            dtype=labels.dtype,
                                            device=labels.device)

                _, ocr_match_det_inds = self.ocr_assign_ids(
                    last_observations, 
                    unmatch_det_bboxes, 
                    unmatch_det_scores,
                    self.weight_iou_with_det_scores,
                    self.match_iou_thr)

                valid = ocr_match_det_inds > -1
                remain_det_ids[valid] = unmatched_track_inds.clone()[
                    ocr_match_det_inds[valid]].to(labels)

                ocr_match_det_bboxes = unmatch_det_bboxes[valid]
                ocr_match_det_labels = unmatch_det_labels[valid]
                ocr_match_det_scores = unmatch_det_scores[valid]
                ocr_match_det_ids = remain_det_ids[valid]
                assert (ocr_match_det_ids > -1).all()

                ocr_unmatch_det_bboxes = unmatch_det_bboxes[~valid]
                ocr_unmatch_det_labels = unmatch_det_labels[~valid]
                ocr_unmatch_det_scores = unmatch_det_scores[~valid]
                ocr_unmatch_det_ids = remain_det_ids[~valid]
                assert (ocr_unmatch_det_ids == -1).all()

                unmatch_det_bboxes = ocr_unmatch_det_bboxes
                unmatch_det_labels = ocr_unmatch_det_labels
                unmatch_det_scores = ocr_unmatch_det_scores
                unmatch_det_ids = ocr_unmatch_det_ids
                match_det_bboxes = torch.cat(
                    (match_det_bboxes, ocr_match_det_bboxes), dim=0)
                match_det_labels = torch.cat(
                    (match_det_labels, ocr_match_det_labels), dim=0)
                match_det_scores = torch.cat(
                    (match_det_scores, ocr_match_det_scores), dim=0)
                match_det_ids = torch.cat((match_det_ids, ocr_match_det_ids),
                                          dim=0)

            # 5. summarize the track results
            for i in range(len(match_det_ids)):
                det_bbox = match_det_bboxes[i]
                track_id = match_det_ids[i].item()
                if not self.tracks[track_id].tracked:
                    # the track is lost before this step
                    self.online_smooth(self.tracks[track_id], det_bbox)

            for track_id in all_track_ids:
                if track_id not in match_det_ids:
                    self.tracks[track_id].tracked = False
                    self.tracks[track_id].obs.append(None)

            bboxes = torch.cat((match_det_bboxes, unmatch_det_bboxes), dim=0)
            labels = torch.cat((match_det_labels, unmatch_det_labels), dim=0)
            scores = torch.cat((match_det_scores, unmatch_det_scores), dim=0)
            ids = torch.cat((match_det_ids, unmatch_det_ids), dim=0)

            # 6. assign new ids
            new_track_inds = ids == -1
            ids[new_track_inds] = torch.arange(
                self.num_tracks,
                self.num_tracks + new_track_inds.sum()).to(labels)
            self.num_tracks += new_track_inds.sum()

        self.update(
            ids=ids,
            bboxes=bboxes,
            labels=labels,
            embeds=det_embeds if self.with_reid else None,
            frame_ids=frame_id)

        # update pred_track_instances
        pred_track_instances = InstanceData()
        pred_track_instances.bboxes = bboxes
        pred_track_instances.labels = labels
        pred_track_instances.scores = scores
        pred_track_instances.instances_id = ids

        return pred_track_instances

                
