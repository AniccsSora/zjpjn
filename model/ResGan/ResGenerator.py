import torch
import torch.nn as nn

class GenResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1)):

        super(GenResidualBlock, self).__init__()

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



class ResNetGenerator(torch.nn.Module):
    def __init__(self, config):
        """
        official default:
            n_ResidualBlock=8,
            n_levels=4,
            z_dim=10,
            output_channels=3,
            bUseMultiResSkips=True
        """
        #
        assert config['input_shape'][0] == config['input_shape'][1]
        n_ResidualBlock = config['n_ResidualBlock']
        n_levels = config['n_levels']
        self.z_dim = config['z_dim']
        output_channels = config['output_channels']
        bUseMultiResSkips = config['bUseMultiResSkips']
        self.img_latent_dim = config['input_shape'][0] // (2 ** n_levels)
        #

        super(ResNetGenerator, self).__init__()

        self.max_filters = 2 ** (n_levels + 3)
        self.n_levels = n_levels
        self.bUseMultiResSkips = bUseMultiResSkips

        self.conv_list = torch.nn.ModuleList()
        self.res_blk_list = torch.nn.ModuleList()
        self.multi_res_skip_list = torch.nn.ModuleList()

        self.input_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=self.z_dim, out_channels=self.max_filters,
                            kernel_size=(3, 3), stride=(1, 1), padding=1),
            torch.nn.BatchNorm2d(self.max_filters),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        for i in range(n_levels):
            n_filters_0 = 2 ** (self.n_levels - i + 3)
            n_filters_1 = 2 ** (self.n_levels - i + 2)
            ks = 2 ** (i + 1)

            self.res_blk_list.append(
                torch.nn.Sequential(*[GenResidualBlock(n_filters_1, n_filters_1)
                                      for _ in range(n_ResidualBlock)])
            )

            self.conv_list.append(
                torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(n_filters_0, n_filters_1,
                                             kernel_size=(2, 2), stride=(2, 2), padding=0),
                    torch.nn.BatchNorm2d(n_filters_1),
                    torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
                )
            )

            if bUseMultiResSkips:
                self.multi_res_skip_list.append(
                    torch.nn.Sequential(
                        torch.nn.ConvTranspose2d(in_channels=self.max_filters, out_channels=n_filters_1,
                                                 kernel_size=(ks, ks), stride=(ks, ks), padding=0),
                        torch.nn.BatchNorm2d(n_filters_1),
                        torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    )
                )

        self.output_conv = torch.nn.Conv2d(in_channels=n_filters_1, out_channels=output_channels,
                                           kernel_size=(3, 3), stride=(1, 1), padding=1)

    def forward(self, z):
        bottleneck_dim = 256
        self.fc1 = torch.nn.Linear(self.z_dim * self.img_latent_dim * self.img_latent_dim, bottleneck_dim, device=z.device)
        self.fc2 = torch.nn.Linear(bottleneck_dim, self.z_dim * self.img_latent_dim * self.img_latent_dim, device=z.device)
        z = self.fc1(z.view(-1, self.z_dim * self.img_latent_dim * self.img_latent_dim))
        z = z_top = self.input_conv(z.view(-1, self.z_dim,16, 16))

        for i in range(self.n_levels):
            z = self.conv_list[i](z)
            z = self.res_blk_list[i](z)
            if self.bUseMultiResSkips:
                z += self.multi_res_skip_list[i](z_top)

        z = self.output_conv(z)

        # 學 原本 AE Decoder 的 sigmoid
        return torch.sigmoid(z)


if __name__ == "__main__":
    latent_dims = 256

    ResGenerator_config = {
        "input_shape": (256, 256, 3),
        "n_ResidualBlock": 4,
        "n_levels": 4,
        "z_dim": 1,  # fixed
        "output_channels": 3,
        "bUseMultiResSkips": True
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = ResNetGenerator(ResGenerator_config).to(device)
    print("ResNetGenerator layers =", sum(1 for _ in G.named_modules()))
    test_batch = torch.randn(2, latent_dims, 1, 1).to(device)

    print("input shape =", test_batch.shape)
    print("output shape =", G(test_batch).shape)