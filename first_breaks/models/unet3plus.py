from first_breaks.const import raise_if_no_torch
raise_if_no_torch()

from math import copysign
from typing import List

import torch
from torch import nn, Tensor
from torchvision import models


class Unet3Plus(nn.Module):
    def __init__(
            self,
            resnet_type: str,
            in_channels: int,
            out_channels: int,
            inter_channels: int,
            pretrained: bool,
    ):
        super().__init__()
        self.encoder = UnetEncoder(resnet_type, pretrained, in_channels)
        self.decoder = Unet3PlusDecoder(self.encoder.encoder_channels, out_channels_decoder_block=inter_channels)
        self.outconv = ReshapeConv2D(len(self.encoder.encoder_channels) * inter_channels, out_channels, -2)

    def forward(self, x: Tensor) -> Tensor:
        features = self.encoder(x)
        mask = self.decoder(features)
        mask = self.outconv(mask)
        return mask


class UnetEncoder(nn.Module):
    def __init__(self, resnet_type: str, pretrained: bool = True, in_channels: int = 1):
        super().__init__()
        self.encoder = models.__dict__[resnet_type](pretrained=pretrained)
        if in_channels != 3:
            self.encoder.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            )

        del self.encoder.fc
        del self.encoder.maxpool
        del self.encoder.avgpool

        self.encoder_channels = self.get_num_channels_feature()

    def get_num_channels_feature(self) -> List[int]:
        layer_name_list = [f"layer{k}" for k in range(1, 5)]
        num_channels = []
        for layer_name in layer_name_list:
            layer = getattr(self.encoder, layer_name)

            for module in layer.modules():
                if isinstance(module, nn.Conv2d):
                    last_conv = module
            num_channels.append(last_conv.out_channels)

        return num_channels

    def forward(self, x: Tensor) -> List[Tensor]:
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        features = []
        for layer in [self.encoder.layer1, self.encoder.layer2, self.encoder.layer3, self.encoder.layer4]:
            x = layer(x)
            features.append(x)
        return features


class Unet3PlusDecoder(nn.Module):
    def __init__(self, encoder_channels: List[int], out_channels_decoder_block: int):
        super().__init__()
        self.decoder_blocks = []
        self.num_encoder_blocks = len(encoder_channels)

        for idx in range(len(encoder_channels) - 1):
            self.decoder_blocks.append(Unet3PlusDecoderBlock(encoder_channels, idx, out_channels_decoder_block))

        self.decoder_blocks.reverse()

        self.decoder_blocks = nn.ModuleList(self.decoder_blocks)

    def forward(self, features: List[Tensor]) -> Tensor:
        for idx, decoder_block in enumerate(self.decoder_blocks):
            idx_feature_to_replace = self.num_encoder_blocks - idx - 2
            x = decoder_block(features)
            features[idx_feature_to_replace] = x
        return x


class Unet3PlusDecoderBlock(nn.Module):
    def __init__(self, encoder_channels: List[int], direct_connection_order: int, out_channels: int = 64):
        super().__init__()
        num_features_blocks = len(encoder_channels)
        assert direct_connection_order < num_features_blocks
        scales = [direct_connection_order - block for block in range(num_features_blocks)]
        scales_factors = [int(copysign(1, sc) * 2 ** abs(sc)) for sc in scales]

        self.reshaping_blocks = []

        for idx, (scale, in_channels) in enumerate(zip(scales_factors, encoder_channels)):
            if idx == num_features_blocks - 1:
                pass
            elif idx > direct_connection_order:
                in_channels = out_channels * num_features_blocks

            self.reshaping_blocks.append(ReshapeConv2D(in_channels, out_channels, scale))

        self.reshaping_blocks = nn.ModuleList(self.reshaping_blocks)

        self.stacking_conv = Conv2dReLU(out_channels * num_features_blocks,
                                        out_channels * num_features_blocks,
                                        kernel_size=(3, 3),
                                        padding=(1, 1))

    def forward(self, features: List[Tensor]) -> Tensor:
        reshaped_features = []
        for tensor, reshape_block in zip(features, self.reshaping_blocks):
            x = reshape_block(tensor)
            reshaped_features.append(x)
        x = torch.cat(reshaped_features, dim=1)
        x = self.stacking_conv(x)
        return x


class Conv2dReLU(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        out_channels = args[1] if len(args) > 1 else kwargs["out_channels"]
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = super().forward(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ReshapeFeatures(nn.Module):
    def __init__(self, scale_factor: int):
        super().__init__()
        if scale_factor > 0:
            self.layer = nn.MaxPool2d(kernel_size=scale_factor)
        elif scale_factor < 0:
            self.layer = nn.UpsamplingBilinear2d(scale_factor=abs(scale_factor))
        else:
            self.layer = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x)


class ReshapeConv2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, scale_factor: int):
        super().__init__()
        self.reshape = ReshapeFeatures(scale_factor)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x: Tensor) -> Tensor:
        x = self.reshape(x)
        x = self.conv(x)
        return x

