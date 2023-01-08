from torchreid.reid.utils import FeatureExtractor
from mmcv.runner import BaseModule, auto_fp16

from ..builder import REID


@REID.register_module()
class MyReID(BaseModule):
    def __init__(self, model_name: str, model_path: str, device: str = "cuda"):
        super().__init__()

        self.reid = FeatureExtractor(
            model_name=model_name, model_path=model_path, device=device
        )

    def forward(self, inputs):
        return self.reid(inputs)

    @auto_fp16(apply_to=('img', ), out_fp32=True)
    def simple_test(self, img, **kwargs):
        """Test without augmentation."""
        if img.nelement() > 0:
            feats = self(img)
            return feats
        else:
            return img.new_zeros(0, self.reid.model.out_channels)
