import torch
from mmdet.models.builder import DETECTORS, HEADS
from mmdet.models.detectors import SingleStageDetector
from mmdet.models.builder import build_backbone, build_head
from mmcv.runner import BaseModule

from yolov7.utils.general import non_max_suppression, scale_coords


# NOTE: This is a trick to make our pretrained model work with mmtrack.
@HEADS.register_module()
class YOLOv7Head(BaseModule):
    def __init__(
        self,
        num_classes: int,
        conf_thresh: float = 0.2,
        iou_thres: float = 0.6,
        init_cfg=None,
        test_cfg=None,
        **kwargs,
    ):
        super().__init__(init_cfg)

        self.num_classes = num_classes
        self.conf_thresh = conf_thresh
        self.iou_thres = iou_thres

        self.test_cfg = test_cfg

    def forward(self, val_outs: torch.Tensor):
        return (val_outs,)

    def get_bboxes(
        self,
        val_outs: torch.Tensor,
        img_metas=None,
        rescale=None,
    ):
        val_outs = val_outs[0]
        outs = non_max_suppression(val_outs, self.conf_thresh, self.iou_thres)

        assert len(outs) == 1, "Only support batch size 1."
        preds = outs[0]

        bboxes = preds[:, :5]
        labels = preds[:, 5].long()

        if img_metas:
            *img_size, _ = img_metas[0]["img_shape"]
            org_img = img_metas[0]["ori_shape"]

            bboxes = scale_coords(img_size, bboxes, org_img).round()

        return [[bboxes, labels]]


@DETECTORS.register_module()
class YOLOv7(SingleStageDetector):
    def __init__(
        self,
        backbone,
        bbox_head,
        **kwargs,
    ):
        super().__init__(
            backbone=backbone,
            bbox_head=bbox_head,
            **kwargs,
        )

    def extract_feat(self, img):
        assert self.training is False, "Only support inference mode."
        return self.backbone(img)
