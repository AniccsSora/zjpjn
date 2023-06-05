import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # todo: wait to fix
        self.downsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels//2, kernel_size=1, stride=stride, bias=False),
            nn.ConvTranspose2d(out_channels//2, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out



class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()

        self.config = config

        self.layers = nn.ModuleList()
        in_channels = config['in_channels']

        layer_depth = config['layer_depth']
        feature_map_ratio = config['feature_map_ratio']
        layer_kernel_size = config['layer_kernel_size']
        layer_stride = config['layer_stride']
        layer_padding = config['layer_padding']

        num_layers = len(layer_depth)

        for i in range(num_layers - 1):  # -1 是因為不要去迭代最後一層
            out_channels = layer_depth[i]
            ratio = feature_map_ratio[i]
            ks = layer_kernel_size[i]
            pad = layer_padding[i]
            stride = layer_stride[i]

            self.layers.append(ResidualBlock(in_channels, out_channels, ks, stride, pad))
            in_channels = out_channels

        self.layers.append(
            nn.ConvTranspose2d(in_channels, config['out_channels'], layer_kernel_size[num_layers - 1],
                               layer_stride[num_layers - 1], layer_padding[num_layers - 1], bias=False)
        )
        self.layers.append(nn.Tanh())

    def forward(self, input):
        x = input
        for layer in self.layers:
            x = layer(x)
        return x


if __name__ == "__main__":
    # 定義超參數配置字典
    config = {
        'in_channels': 128,                    # 輸入通道數
        'out_channels': 3,                     # 輸出通道數
        'layer_depth': [512, 256, 128, 64, 32, 3],          # 每層的通道數
        'feature_map_ratio': [8, 4, 2, 1, 1, 1],           # 特徵圖的比例
        'layer_kernel_size': [4, 4, 4, 4, 4, 4],         # 卷積核尺寸
        'layer_stride': [1, 2, 2, 2, 2, 2],              # 卷積步長
        'layer_padding': [0, 1, 1, 1, 1, 1],             # 卷積填充數
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 初始化 Generator
    generator = Generator(config).to(device)

    # 生成假批次，輸入維度為128
    z = torch.randn(128, 128, 1, 1, device=device)

    # 生成圖像
    output = generator(z)

    print(output.shape)
