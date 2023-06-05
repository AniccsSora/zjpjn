import sys
sys.path.append('../../')
#
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd.variable import Variable
from torchvision.utils import save_image
import torchvision.utils as vutils
import os
import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
# from self
from utils.dataloaders.single_qrCode_field_dataset import QRCodeDataset, get_QRCode_dataloader
#
from pathlib import Path
from utils.F import ensure_folder, timestamp


# 3
# layer_depth = [512, 256, 128, 64, num_output_channels]
# feature_map_ratio = [8, 4, 2, 1, 1]
# layer_kernel_size = [4, 4, 4, 4, 4]
# layer_stride = [1, 2, 2, 2, 2]
# layer_padding = [0, 1, 1, 1, 1]

class Generator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Generator, self).__init__()

        layer_depth = [512, 256, 128, 64, 32, 3]
        feature_map_ratio = [8, 4, 2, 1, 1, 1]
        layer_kernel_size = [4, 4, 4, 4, 4, 4]
        layer_stride = [1, 2, 2, 2, 2, 2]
        layer_padding = [0, 1, 1, 1, 1, 1]

        #
        layers = []
        num_layers = len(layer_depth)

        for i in range(num_layers-1):  # -1 是因為不要去迭代最後一層
            out_channels = layer_depth[i]
            ratio = feature_map_ratio[i]
            ks = layer_kernel_size[i]
            pad = layer_padding[i]
            stride = layer_stride[i]
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, ks, stride, pad, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(True))

            in_channels = out_channels

        layers.append(nn.ConvTranspose2d(in_channels, 3, layer_kernel_size[num_layers-1],
                                         layer_stride[num_layers-1],
                                         layer_padding[num_layers-1], bias=False)
                      )
        layers.append(nn.Tanh())

        self.main = nn.Sequential(*layers)

    def forward(self, input):
        return self.main(input)

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    g = Generator(128, 3).to(device)

    # generate fake batch with 128 dimension
    z = torch.randn(128, 128, 1, 1, device=device)



    print("a")
    pass