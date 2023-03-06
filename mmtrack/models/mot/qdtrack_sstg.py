# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, Optional

import torch
from torch import Tensor

from mmtrack.registry import MODELS
from mmtrack.structures import TrackDataSample
from mmtrack.utils import OptConfigType, OptMultiConfig, SampleList
from .base import BaseMultiObjectTracker

@MODELS.register_module()
class QDTrackSSTG(BaseMultiObjectTracker):
    """Quasi-Dense Similarity Learning Single Stage for Multiple Object Tracking.

    This multi object tracker is the implementation of `QDTrack
    <https://arxiv.org/abs/2006.06664>`_.

    Args:
        detector (dict): Configuration of detector. Defaults to None.
        track_head (dict): Configuration of track head. Defaults to None.
        tracker (dict): Configuration of tracker. Defaults to None.
        freeze_detector (bool): If True, freeze the detector weights.
            Defaults to False.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
           config of :class:`TrackDataPreprocessor`.  it usually includes,
            ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (dict or list[dict]): Configuration of initialization.
            Defaults to None.
    """

    def __init__(self,
                 detector: Optional[dict] = None,
                 track_head: Optional[dict] = None,
                 tracker: Optional[dict] = None,
                 freeze_detector: bool = False,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 train_cfg: OptMultiConfig = None,
                 test_cfg: OptMultiConfig = None):
        super().__init__(data_preprocessor, init_cfg)
        if detector is not None:
            self.detector = MODELS.build(detector)

        if track_head is not None:
            self.track_head = MODELS.build(track_head)

        if tracker is not None:
            self.tracker = MODELS.build(tracker)

        self.freeze_detector = freeze_detector
        if self.freeze_detector:
            self.freeze_module('detector')

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def loss(self, inputs: Dict[str, Tensor], data_samples: SampleList,
             **kwargs) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Dict[str, Tensor]): of shape (N, T, C, H, W) encoding
                input images. Typically these should be mean centered and std
                scaled. The N denotes batch size.The T denotes the number of
                key/reference frames.
                - img (Tensor) : The key images.
                - ref_img (Tensor): The reference images.
            data_samples (list[:obj:`TrackDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance`.

        Returns:
            dict: A dictionary of loss components.
        """
        # modify the inputs shape to fit mmdet
        img = inputs['img']
        assert img.dim() == 5, 'The img must be 5D Tensor (N, T, C, H, W).'
        assert img.size(1) == 1, \
            'QDTrack can only have 1 key frame and 1 reference frame.'
        img = img[:, 0]

        ref_img = inputs['ref_img']
        assert ref_img.dim() == 5, 'The img must be 5D Tensor (N, T, C, H, W).'
        assert ref_img.size(1) == 1, \
            'QDTrack can only have 1 key frame and 1 reference frame.'
        ref_img = ref_img[:, 0]

        x = self.detector.extract_feat(img)
        ref_x = self.detector.extract_feat(ref_img)

        losses = dict()

        detect_losses, proposal_results = self.detector.bbox_head.loss_and_predict(
            x, data_samples, self.detector.train_cfg, **kwargs)

        losses.update(detect_losses)

        # adjust the key of ref_img in data_samples
        ref_data_samples = []
        for data_sample in data_samples:
            ref_data_sample = TrackDataSample()
            ref_data_sample.set_metainfo(
                metainfo=dict(
                    img_shape=data_sample.metainfo['ref_img_shape'],
                    scale_factor=data_sample.metainfo['ref_scale_factor']))
            ref_data_samples.append(ref_data_sample)
        
        # Since mmdet does not support predict with custom test_cfg, 
        # we need to change the test_cfg of bbox_head temporarily.
        backup_cfg = self.detector.bbox_head.test_cfg
        self.detector.bbox_head.test_cfg = self.detector.train_cfg
        ref_proposal_results = self.detector.bbox_head.predict(
            ref_x, ref_data_samples, **kwargs)
        self.detector.bbox_head.test_cfg = backup_cfg

        losses_track = self.track_head.loss(x, ref_x, proposal_results,
                                            ref_proposal_results, data_samples,
                                            **kwargs)
        losses.update(losses_track)

        return losses

    def predict(self,
                inputs: Dict[str, Tensor],
                data_samples: SampleList,
                rescale: bool = True,
                **kwargs) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (Dict[str, Tensor]): of shape (N, T, C, H, W) encoding
                input images. Typically these should be mean centered and std
                scaled. The N denotes batch size.The T denotes the number of
                key/reference frames.
                - img (Tensor) : The key images.
                - ref_img (Tensor): The reference images.
            data_samples (list[:obj:`TrackDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance`.
            rescale (bool, Optional): If False, then returned bboxes and masks
                will fit the scale of img, otherwise, returned bboxes and masks
                will fit the scale of original image shape. Defaults to True.

        Returns:
            SampleList: Tracking results of the input images.
            Each TrackDataSample usually contains ``pred_det_instances``
            or ``pred_track_instances``.
        """
        img = inputs['img']
        assert img.dim() == 5, 'The img must be 5D Tensor (N, T, C, H, W).'
        assert img.size(1) == 1, \
            'QDTrack can only have 1 key frame.'
        img = img[:, 0]

        assert len(data_samples) == 1, \
            'QDTrack only support 1 batch size per gpu for now.'
        metainfo = data_samples[0].metainfo
        frame_id = metainfo.get('frame_id', -1)
        if frame_id == 0:
            self.tracker.reset()

        x = self.detector.extract_feat(img)
        det_results = self.detector.bbox_head.predict(x, data_samples, rescale=rescale)

        track_data_sample = data_samples[0]
        track_data_sample.pred_det_instances = \
            det_results[0].clone()

        pred_track_instances = self.tracker.track(
            model=self,
            img=img,
            feats=x,
            data_sample=track_data_sample,
            rescale=rescale,
            **kwargs)
        track_data_sample.pred_track_instances = pred_track_instances

        return [track_data_sample]

    def batch_predict(self,
                inputs: Dict[str, Tensor],
                data_samples: SampleList,
                rescale: bool = True,
                **kwargs) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (Dict[str, Tensor]): of shape (N, T, C, H, W) encoding
                input images. Typically these should be mean centered and std
                scaled. The N denotes batch size.The T denotes the number of
                key/reference frames.
                - img (Tensor) : The key images.
                - ref_img (Tensor): The reference images.
            data_samples (list[:obj:`TrackDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance`.
            rescale (bool, Optional): If False, then returned bboxes and masks
                will fit the scale of img, otherwise, returned bboxes and masks
                will fit the scale of original image shape. Defaults to True.

        Returns:
            SampleList: Tracking results of the input images.
            Each TrackDataSample usually contains ``pred_det_instances``
            or ``pred_track_instances``.
        """
        imgs = inputs['img']
        assert imgs.dim() == 5, 'The img must be 5D Tensor (N, T, C, H, W).'
        assert imgs.size(1) == 1, \
            'QDTrack can only have 1 key frame.'
        imgs = imgs.squeeze_(1)

        track_data_samples = data_samples

        x = self.detector.extract_feat(imgs)
        det_results = self.detector.bbox_head.predict(x, data_samples, rescale=rescale)
        # import pdb; pdb.set_trace()
        for i in range(len(data_samples)):
            metainfo = data_samples[i].metainfo
            frame_id = metainfo.get('frame_id', -1)
            if frame_id == 0:
                self.tracker.reset()

            track_data_sample = track_data_samples[i]
            track_data_sample.pred_det_instances = \
                det_results[i].clone()
            
            img_feats = tuple(feat[i].unsqueeze(0) for feat in x)
            pred_track_instances = self.tracker.track(
                model=self,
                img=imgs[i].unsqueeze(0),
                feats=img_feats,
                data_sample=track_data_sample,
                rescale=rescale,
                **kwargs)
            track_data_sample.pred_track_instances = pred_track_instances

        return track_data_samples
