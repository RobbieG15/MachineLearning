#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This file contains the UNET neural network architecture."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()

        # Contracting path (encoder)
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Expansive path (decoder)
        self.upconv4 = self.upconv_block(1024, 512)
        self.decoder4 = self.conv_block(1024, 512)

        self.upconv3 = self.upconv_block(512, 256)
        self.decoder3 = self.conv_block(512, 256)

        self.upconv2 = self.upconv_block(256, 128)
        self.decoder2 = self.conv_block(256, 128)

        self.upconv1 = self.upconv_block(128, 64)
        self.decoder1 = self.conv_block(128, 64)

        # Final output layer
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        # Encoder (downsampling)
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, 2))
        enc3 = self.encoder3(F.max_pool2d(enc2, 2))
        enc4 = self.encoder4(F.max_pool2d(enc3, 2))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))

        # Decoder (upsampling)
        up4 = self.upconv4(bottleneck)
        dec4 = self.decoder4(torch.cat((up4, enc4), dim=1))

        up3 = self.upconv3(dec4)
        dec3 = self.decoder3(torch.cat((up3, enc3), dim=1))

        up2 = self.upconv2(dec3)
        dec2 = self.decoder2(torch.cat((up2, enc2), dim=1))

        up1 = self.upconv1(dec2)
        dec1 = self.decoder1(torch.cat((up1, enc1), dim=1))

        return self.final_conv(dec1)


# Example usage:
# model = UNet(in_channels=1, out_channels=1)
# input_tensor = torch.randn(1, 1, 256, 256)  # Batch size of 1, single channel, 256x256 image
# output = model(input_tensor)
