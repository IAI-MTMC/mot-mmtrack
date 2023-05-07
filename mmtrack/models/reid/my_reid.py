# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.model import BaseModel
from torchreid.reid.models.osnet import OSBlock, OSNet, init_pretrained_weights
from torchreid.reid.utils import check_isfile, load_pretrained_weights

from mmtrack.registry import MODELS


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

    def __init__(self, model_name: str, model_path: str, device: str,
                 feature_dim: int):
        super().__init__()

        pretrained = (model_path and check_isfile(model_path))
        self.model = osnet_x1_0(
            num_classes=1,
            pretrained=not pretrained,
            loss='triplet',
            feature_dim=feature_dim,
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
            out_channels = self.model.feature_dim

        return Head()

    def forward(self, inputs, mode: str = 'tensor', frame_id=-1):
        print('reid by osnet_x1_0')
        self.visualize_crop_images(inputs, frame_id)
        # self.test_reid()
        assert mode == 'tensor', 'Only support tensor mode'
        features = self.model(inputs)
        print('features:', features.shape)
        return features

    def visualize_crop_images(self, inputs, frame_id):
        import os
        import cv2
        import numpy as np

        mean = np.array([[[123.675, 116.28, 103.53]]])
        std = np.array([[[58.395, 57.12, 57.375]]])

        try:
            os.makedirs("images")
        except FileExistsError:
            pass

        print('crop images: ', inputs.shape)
        for i, img in enumerate(inputs):
            img = img.detach().moveaxis(0, -1).cpu().numpy()
            img = img * std + mean
            img_path = 'images/image_' + str(frame_id) + '_' + str(i) + '.jpg'
        cv2.imwrite(img_path, img[..., ::-1])

    def test_reid(self):
        print('------------ test -----------')
        import glob

        import cv2
        import torch
        from torchvision.transforms import Compose, Normalize, Resize, ToTensor

        transform = Compose([
            ToTensor(),
            Resize((256, 128)),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        images = []

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3)
        images_path = glob.glob('../reid/data/*')
        images_path.sort()
        for image_path in images_path:
            image = cv2.imread(image_path)
            image = image[..., ::-1]
            image = transform(image.copy())

            print(image.min(), image.max())
            images.append(image)
            print(image_path)
            img = image.clone().moveaxis(0, -1)
            img = (img * std + mean) * 255
            img = img.numpy()
            img = img[..., ::-1]
            # cv2.imwrite(image_path.split('/')[-1], img)

        images = torch.stack(images, dim=0)
        images = images.cuda()
        print(images.shape)
        features = self.model(images)
        print(features.min(), features.max())
        print(features.shape)

        print(torch.cdist(features, features))

        print('------------ test -----------')
