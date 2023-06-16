import torch
import matplotlib.pyplot as plt
import tqdm
import torch.nn as nn

class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1)):

        super(ResidualBlock, self).__init__()

        self.residual_block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                            kernel_size=kernel_size, stride=stride, padding=1),
            torch.nn.BatchNorm2d(in_channels),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
            torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                            kernel_size=kernel_size, stride=stride, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x):
        return x + self.residual_block(x)


class ResNetDiscriminator(torch.nn.Module):
    def __init__(self, config):
        #
        n_ResidualBlock = config['n_ResidualBlock']  # offcial default = 8
        n_levels = config['n_levels']  # offcial default = 4
        input_ch = config.get('input_ch')  # offcial default = 3
        if input_ch is None:
            input_ch = config['input_shape'][2]
        self.z_dim = config['z_dim']  # offcial default = 10
        bUseMultiResSkips = config['bUseMultiResSkips']  # offcial default = True
        #
        super(ResNetDiscriminator, self).__init__()

        self.max_filters = 2 ** (n_levels+3)
        self.n_levels = n_levels
        self.bUseMultiResSkips = bUseMultiResSkips

        self.conv_list = torch.nn.ModuleList()
        self.res_blk_list = torch.nn.ModuleList()
        self.multi_res_skip_list = torch.nn.ModuleList()

        self.input_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=input_ch, out_channels=8,
                            kernel_size=(3, 3), stride=(1, 1), padding=1),
            torch.nn.BatchNorm2d(8),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        for i in range(n_levels):
            n_filters_1 = 2 ** (i + 3)
            n_filters_2 = 2 ** (i + 4)
            ks = 2 ** (n_levels - i)

            self.res_blk_list.append(
                torch.nn.Sequential(*[ResidualBlock(n_filters_1, n_filters_1)
                                      for _ in range(n_ResidualBlock)])
            )

            self.conv_list.append(
                torch.nn.Sequential(
                    torch.nn.Conv2d(n_filters_1, n_filters_2,
                                    kernel_size=(2, 2), stride=(2, 2), padding=0),
                    torch.nn.BatchNorm2d(n_filters_2),
                    torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
                )
            )

            if bUseMultiResSkips:
                self.multi_res_skip_list.append(
                    torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=n_filters_1, out_channels=self.max_filters,
                                        kernel_size=(ks, ks), stride=(ks, ks), padding=0),
                        torch.nn.BatchNorm2d(self.max_filters),
                        torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    )
                )

        self.output_conv = torch.nn.Conv2d(in_channels=self.max_filters, out_channels=self.z_dim,
                                           kernel_size=(3, 3), stride=(1, 1), padding=1)
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fc1 = nn.Linear(1024, self.z_dim, bias=False, device=_device)
        self.fc2 = nn.Linear(self.z_dim, 1, bias=False, device=_device)

    def forward(self, x):
        b_size = x.shape[0]
        x = self.input_conv(x)

        skips = []
        for i in range(self.n_levels):
            x = self.res_blk_list[i](x)
            if self.bUseMultiResSkips:
                skips.append(self.multi_res_skip_list[i](x))
            x = self.conv_list[i](x)

        if self.bUseMultiResSkips:
            x = sum([x] + skips)

        x = self.output_conv(x)

        # go flatten
        x = x.view(b_size, -1)
        # 第一個參數跟著 config 改
        x = self.fc1(x)
        x = self.fc2(x)
        return torch.sigmoid(x)


if __name__ == "__main__":
    # 這裡的 config 會影響模型的輸出 形狀，請注意!!!
    discriminator_config = {
        'n_ResidualBlock': 8,
        'n_levels': 4,
        'input_ch': 3,
        'z_dim': 4,  # the same between E(), D()
        'bUseMultiResSkips': True
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D = ResNetDiscriminator(config=discriminator_config).to(device)
    print("ResNetDiscriminator layers =", sum(1 for _ in D.named_modules()))

    test_batch = torch.randn((4, 3, 256, 256)).to(device)

    print("input shape =", test_batch.shape)
    print("output shape =", D(test_batch).shape)

    # export to onnx
    input_names = ["actual_input"]
    output_names = ["output_D"]

    torch.onnx.export(model=D, args=test_batch, f="ResD.onnx", verbose=True,
                      input_names=input_names,
                      output_names=output_names)


