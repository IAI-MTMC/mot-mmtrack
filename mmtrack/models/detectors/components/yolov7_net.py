# Source: https://github.com/IAI-MTMC/MOT/tree/main/src/models/det/components/yolov7_net.py

from typing import Optional
import torch
from torch import nn
from torch.utils.model_zoo import load_url

from mmdet.models import BACKBONES


@BACKBONES.register_module()
class YOLOv7Net(nn.Module):
    def __init__(
        self,
        cfg_path: str,
        pretrained_path: Optional[str] = None,
    ):
        """
        Args:
            cfg_path (str): path to the yaml config file to instantiate YOLOv7
            pretrained_path (str): path to the pretrained weights
        """
        super().__init__()
        self.cfg_path = cfg_path
        self.pretrained_path = pretrained_path

        self.net = self.load_net()

    def load_net(self):
        # from pathlib import Path

        # path = Path(__file__).parents[1].resolve().joinpath("modules/yolov7").resolve()

        net = torch.hub.load("IAI-MTMC/yolov7", "yolov7_custom", cfg=self.cfg_path)

        assert isinstance(net, nn.Module)

        if self.pretrained_path is not None:
            if self.pretrained_path.startswith("http"):
                ckpt = load_url(self.pretrained_path)
            else:
                ckpt = torch.load(self.pretrained_path)
            net.load_state_dict(ckpt["model"].state_dict())

        return net

    def forward(self, x):
        return self.net(x)
