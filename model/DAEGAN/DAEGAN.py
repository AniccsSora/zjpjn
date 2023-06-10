from model.Autoencoder.model import ResNetAE_Conductor
from model.Gan.Discriminator import Discriminator
from model.Gan.Generator import Generator

import torch



if __name__== "__main__":

    image_size = 256
    image_channels = 3

    # 鑑別器輸入大小，應與 AE latent vector size 維度相當。
    bottle_neck_dims = 256

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ResAE
    res_ae_config = {
        'input_shape': (image_size, image_size, image_channels),
        'n_ResidualBlock': 2,
        'n_levels': 2,
        'z_dim': 10,
        'bottleneck_dim': bottle_neck_dims,  # 真正的 latent vector size
        'bUseMultiResSkips': True
    }

    discrimitor_config = {
        'in_channels': image_channels,                      # 輸入的圖片通道數
        'out_channels': 1,                     # 最後一層輸出通道數
        'layer_depth': [4, 8, 16, 32, 64, 128, 256, 512], # 每層的輸出通道數量
        'layer_kernel_size': [4, 4, 4, 4, 4, 4, 4, 4],      # 卷積核尺寸
        'layer_stride': [2, 2, 2, 2, 2, 2, 2, 2],           # 卷積步長
        'layer_padding': [1, 1, 1, 1, 1, 1, 1, 1],          # 卷積填充數
    }

    G_config = {
        'in_channels': bottle_neck_dims,
        'out_channels': 3,  # 最後一層輸出通道數
        'layer_depth': [1024, 512, 256, 32, 16, 3],  # 每層的輸出通道數量
        'layer_kernel_size': [4, 4, 4, 4, 4, 4, 4, 4],  # 卷積核尺寸
        'layer_stride': [2, 2, 2, 2, 2, 2, 2, 2],  # 卷積步長
        'layer_padding': [0, 1, 1, 1, 1, 1, 1, 1],  # 卷積填充數
    }

    res_ae = ResNetAE_Conductor(config=res_ae_config).to(device)
    # 訓練AE的圖片們
    ae_test_input_batch = torch.randn((32, 3, 256, 256)).to(device)

    discriminator = Discriminator(config=discrimitor_config).to(device)
    # 輸入給 discrimitor 的圖片們，形狀應等同 AE 所輸出的。
    ae_test_input_batch = torch.randn_like(ae_test_input_batch).to(device)

    generator = Generator(config=G_config).to(device)
    # 輸入給 generator 的 latent vector。
    g_test_input_batch = torch.randn((32, bottle_neck_dims, 1, 1)).to(device)
    #
    generator_output = generator(g_test_input_batch)

