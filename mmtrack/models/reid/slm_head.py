import warnings
from typing import Union, List, Optional, Tuple
from math import sqrt, log

import torch
from torch import nn
from mmcls.evaluation.metrics import Accuracy
from mmengine.model import BaseModule

from mmtrack.registry import MODELS
from mmtrack.structures import ReIDDataSample


@MODELS.register_module()
class SLMHead(BaseModule):
    """Similarity Learning Module for re-identification."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_slices: int = 4,
                 num_attn_heads: int = 8,
                 num_classes: Union[int, None] = None,
                 shared_head: Optional[dict] = None,
                 loss_cls: Optional[dict] = None,
                 loss_triplet: Optional[dict] = None,
                 init_cfg: Union[dict, List[dict]] = None):
        super().__init__(init_cfg=init_cfg)

        if loss_cls is None:
            if isinstance(num_classes, int):
                warnings.warn('Since cross entropy is not set, '
                              'the num_classes will be ignored.')
            if loss_triplet is None:
                raise ValueError('Please choose at least one loss in '
                                 'triplet loss and cross entropy loss.')
        elif not isinstance(num_classes, int):
            raise TypeError('The num_classes must be a current number, '
                            'if there is cross entropy loss.')
        self.loss_cls = MODELS.build(loss_cls) if loss_cls else None
        self.loss_triplet = MODELS.build(loss_triplet) \
            if loss_triplet else None
        
        self.shared_head = MODELS.build(shared_head) if shared_head else None

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_slices = num_slices
        self.num_attn_heads = num_attn_heads

        self._init_layers()
    
    def _init_layers(self):
        self.qkv = nn.Conv2d(self.in_channels, self.in_channels * 3, kernel_size=1)
        self.fc_out = nn.Linear(self.out_channels, self.out_channels)
        self.register_buffer('pos_enc', self.positional_encoding(self.in_channels))
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        if self.loss_cls:
            self.bn = nn.BatchNorm1d(self.out_channels)
            self.classifier = nn.Linear(self.out_channels, self.num_classes)
        

    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The forward process."""
        # Multiple stage inputs are acceptable
        # but only the last stage will be used.
        feats = feats[-1] # Shape: (N, C, H, W)

        # 1. Split the input feature map into multiple slices.
        # Shape: (N, C, H, W) -> (N, C, num_slices, h, w)
        feats = self.split(feats, self.num_slices)

        # 2. Apply positional encoding to each slice.
        pos_emb = self.pos_enc[:self.num_slices] # Shape: (num_slices, 1, C)
        feats = feats + pos_emb.permute(1, 2, 0)

        # 3. Apply self-attention and cross-attention to each slide.
        _, N, C, h, w = feats.shape
        feats = feats.permute(2, 0, 1, 3, 4).contiguous() # Shape: (num_slices, N, C, h, w)
        for slice_idx in range(self.num_slices):
            feats[slice_idx] = (self.qkv(feats[slice_idx])
                                .view(N, 3, C, h * w)
                                .permute(1, 0, 3, 2)) # Shape: (3, N, h * w, C)
        
        outputs = []
        for slice_idx in range(self.num_slices):
            attn_out = 0
            q = feats[slice_idx][0] # Shape: (N, h * w, C)
            for slice_idx in range(self.num_slices):
                k, v = feats[slice_idx][1:]
                attn_out += self.attention(q, k, v, self.num_attn_heads)
            outputs.append(attn_out)
        
        # 4. Concatenate the outputs of all slices and group back to original shape.
        feats = torch.stack(outputs, dim=1) # Shape: (N, num_slices, h * w, C)
        feats = feats.permute(0, 3, 1, 2).contiguous().view(N, C, self.num_slices, h, w)
        feats = self.group(feats) # Shape: (N, C, H, W)

        if self.shared_head:
            feats = self.shared_head(feats)

        feats = self.global_avgpool(feats) # Shape: (N, C)
        feats = self.fc_out(feats) # Shape: (N, out_channels)

        return feats
    
    def loss(self, feats: Tuple[torch.Tensor],
             data_samples: List[ReIDDataSample]) -> dict:
        """Calculate losses.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
            data_samples (List[ReIDDataSample]): The annotation data of
                every samples.

        Returns:
            dict: a dictionary of loss components
        """
        # The part can be traced by torch.fx
        feats = self(feats)

        # The part can not be traced by torch.fx
        losses = self.loss_by_feat(feats, data_samples)
        return losses
    
    def loss_by_feat(self, feats: torch.Tensor,
                     data_samples: List[ReIDDataSample]) -> dict:
        """Unpack data samples and compute loss."""
        losses = dict()
        gt_label = torch.cat([i.gt_label.label for i in data_samples])

        if self.loss_triplet:
            losses['triplet_loss'] = self.loss_triplet(feats, gt_label)

        if self.loss_cls:
            feats_bn = self.bn(feats)
            cls_score = self.classifier(feats_bn)
            losses['ce_loss'] = self.loss_cls(cls_score, gt_label)
            acc = Accuracy.calculate(cls_score, gt_label, topk=self.topk)
            losses.update(
                {f'accuracy_top-{k}': a
                 for k, a in zip(self.topk, acc)})

        return losses

    def predict(
            self,
            feats: Tuple[torch.Tensor],
            data_samples: List[ReIDDataSample] = None) -> List[ReIDDataSample]:
        """Inference without augmentation.

        Args:
            feats (Tuple[Tensor]): The features extracted from the backbone.
                Multiple stage inputs are acceptable but only the last stage
                will be used.
            data_samples (List[ReIDDataSample], optional): The annotation
                data of every samples. If not None, set ``pred_label`` of
                the input data samples. Defaults to None.

        Returns:
            List[ReIDDataSample]: A list of data samples which contains the
            predicted results.
        """
        # The part can be traced by torch.fx
        feats = self(feats)

        # The part can not be traced by torch.fx
        data_samples = self.predict_by_feat(feats, data_samples)

        return data_samples

    def predict_by_feat(
            self,
            feats: torch.Tensor,
            data_samples: List[ReIDDataSample] = None) -> List[ReIDDataSample]:
        """Add prediction features to data samples."""
        if data_samples is not None:
            for data_sample, feat in zip(data_samples, feats):
                data_sample.pred_feature = feat
        else:
            data_samples = []
            for feat in feats:
                data_sample = ReIDDataSample()
                data_sample.pred_feature = feat
                data_samples.append(data_sample)

        return data_samples

    def split(x: torch.Tensor, num_slices: int = 4):
        """Split the input feature map into multiple slices."""
        N, C, H, W = x.shape
        grid_size = sqrt(num_slices)
        assert grid_size.is_integer(), 'The number of slices must be a square number.'
        grid_size = int(grid_size)

        x = x.view(N, C, grid_size, H // grid_size, -1)
        x = x.transpose_(-1, -2).contiguous().view(N, C, num_slices, W // grid_size, H // grid_size)
        x = x.transpose_(-1, -2).contiguous()
        return x
    
    def group(slices: torch.Tensor):
        """Group all slices into a single feature map."""
        N, C, num_slices, h, w = slices.shape
        grid_size = sqrt(num_slices)
        assert grid_size.is_integer(), 'The number of slices must be a square number.'
        grid_size = int(grid_size)

        slices = slices.transpose_(-1, -2).contiguous().view(N, C, grid_size, grid_size * w, h)
        slices = slices.transpose_(-1, -2).contiguous().view(N, C, grid_size * h, grid_size * w)
        
        return slices

    def positional_encoding(self, feature_dim: int, max_seq_len: int = 1000):
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, feature_dim, 2) * (-log(10000.0) / feature_dim))
        pe = torch.zeros(max_seq_len, 1, feature_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        return pe

    def attention(
            self,
            q: torch.Tensor, 
            k: torch.Tensor, 
            v: torch.Tensor,
            num_heads: int = 8,
            dropout: float = 0.1):
        """Attention module.

        Args:
            q (torch.Tensor): query. Shape: (N, S, D)
            k (torch.Tensor): key. Shape: (N, S, D)
            v (torch.Tensor): value. Shape: (N, S, D)
            num_heads (int): number of attention heads. Default: 8
        """
        N, S, D = q.shape
        q = q.view(N, S, num_heads, -1).contiguous().permute(0, 2, 1, 3)
        k = k.view(N, S, num_heads, -1).contiguous().permute(0, 2, 1, 3)
        v = v.view(N, S, num_heads, -1).contiguous().permute(0, 2, 1, 3)

        scale = 1 / sqrt(sqrt(D))
        # logits.shape = (N, num_heads, S, S)
        logits = torch.matmul(q * scale, k.transpose(-1, -2) * scale)
        # subtracting a constant value from the tensor won't change the output of softmax.
        # apply the subtraction to avoid value overflow in torch.nn.functional.softmax.
        logits = logits - logits.max(dim=-1, keepdim=True)[0]
        probs = torch.softmax(logits, dim=-1)
        probs = torch.nn.functional.dropout(probs, p=dropout, training=self.training, inplace=True)

        outputs = torch.matmul(probs, v)
        outputs = outputs.permute(0, 2, 1, 3).contiguous().view(N, S, -1)

        return outputs
