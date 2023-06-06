import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()

        self.config = config

        self.layers = nn.ModuleList()
        in_channels = config['in_channels']
        layer_depth = config['layer_depth']
        layer_kernel_size = config['layer_kernel_size']
        layer_stride = config['layer_stride']
        layer_padding = config['layer_padding']

        num_layers = len(layer_depth)

        for i in range(num_layers - 1):  # -1 是因為不要去迭代最後一層
            out_channels = layer_depth[i]
            pad = layer_padding[i]
            
            if isinstance(layer_kernel_size, list):
                ks = layer_kernel_size[i]
            else:
                ks = layer_kernel_size
            
            if isinstance(layer_stride, list):
                stride = layer_stride[i]
            else:
                stride = layer_stride[i]

            self.layers.append(nn.ConvTranspose2d(in_channels, out_channels, ks, stride, pad, bias=False))
            self.layers.append(nn.BatchNorm2d(out_channels))
            self.layers.append(nn.ReLU(True))

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
    # 輸入通道數 
    G_input_latent_dims = 256
    
    # 定義超參數配置字典
    G_config = {
        'in_channels': G_input_latent_dims,                    
        'out_channels': 3,                     # 最後一層輸出通道數
        'layer_depth': [1024, 512, 256, 32, 16, 3],       # 每層的輸出通道數量
        'layer_kernel_size': [4, 4, 4, 4, 4, 4, 4, 4],         # 卷積核尺寸
        'layer_stride': [2, 2, 2, 2, 2, 2, 2, 2],              # 卷積步長
        'layer_padding': [0, 1, 1, 1, 1, 1, 1, 1],             # 卷積填充數
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 初始化 Generator
    generator = Generator(G_config).to(device)

    # 生成假批次，輸入維度為 input_latent_dims
    z = torch.randn(32, G_input_latent_dims, 1, 1, device=device)
    print("input.shape :")
    print(z.shape)
    
    # 生成圖像
    output = generator(z)

    print()
    print("output.shape :")
    print(output.shape)

