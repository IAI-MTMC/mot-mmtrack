from mmengine.model import BaseModel

from mmtrack.registry import MODELS

from torchreid.reid.models.osnet import OSNet, init_pretrained_weights, OSBlock
from torchreid.reid.utils import (check_isfile, load_pretrained_weights)


def osnet_x1_0(num_classes=1000,
               pretrained=True,
               loss='softmax',
               feature_dim=256,
               use_gpu=True):
    # standard size (width x1.0)
    model = OSNet(
        num_classes,
        blocks=[OSBlock, OSBlock, OSBlock],
        layers=[2, 2, 2],
        channels=[64, 256, 384, 512],
        loss=loss,
        feature_dim=feature_dim,
        use_gpu=use_gpu)
    if pretrained:
        init_pretrained_weights(model, key='osnet_x1_0')
    return model


@MODELS.register_module()
class MyReID(BaseModel):

    def __init__(self, model_name: str, model_path: str, device: str):
        super().__init__()

        pretrained = (model_path and check_isfile(model_path))
        self.model = osnet_x1_0(
            num_classes=1,
            pretrained=not pretrained,
            loss='triplet',
            feature_dim=256,
            use_gpu=device.startswith('cuda'))
        self.model.eval()
        if pretrained:
            load_pretrained_weights(self.model, model_path)

        print('+++++++++++++++')
        print('pretrained: ', pretrained)
        print('model_path: ', model_path)
        print(self.model.feature_dim)
        print('+++++++++++++++')

    @property
    def head(self):

        class Head:
            out_channels = self.reid.model.feature_dim

        return Head()

    def forward(self, inputs, mode: str = 'tensor'):
        assert mode == 'tensor', "Only support tensor mode"

        outs = self.model(inputs)
        return outs
