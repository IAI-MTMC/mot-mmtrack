from mmdet.structures import SampleList
from torch import Tensor
from mmtrack.registry import MODELS
from mmdet.models import BaseDetector


@MODELS.register_module()
class GtDetector(BaseDetector):
    """
    Ground-truth detector is tricky way to evaluate the performance of trackers without detection module.
    This module take the ground-truth bounding boxes as input and output the same bounding boxes.
    """
    def predict(self, batch_inputs: Tensor, batch_data_samples: SampleList) -> SampleList:
        return batch_data_samples
