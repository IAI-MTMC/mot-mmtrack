# Source: https://github.com/IAI-MTMC/MOT/tree/main/src/models/det/components/yolov7_net.py

from torch import nn

from mmdet.models import BACKBONES
from yolov7.models.yolo import Model


@BACKBONES.register_module()
class YOLOv7Net(nn.Module):
    def __init__(
        self,
        cfg_path: str,
    ):
        """
        Args:
            cfg_path (str): path to the yaml config file to instantiate YOLOv7
        """
        super().__init__()
        self.cfg_path = cfg_path

        self.net = self.load_net()

    def load_net(self):
        from torch.utils.model_zoo import load_url
        ckpt = load_url("https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt")

        net = Model(self.cfg_path)
        net.load_state_dict(ckpt["model"].state_dict(), strict=False)

        return net

    def forward(self, x):
        return self.net(x)
