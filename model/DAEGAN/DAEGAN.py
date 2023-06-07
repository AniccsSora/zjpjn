from model.Autoencoder.model import ResNetAE_Conductor
from model.Gan.Discriminator import Discriminator




if __name__== "__main__":

    image_size = 256
    image_channels = 3

    # 鑑別器輸入 大小，需要與 laten vetor size 相當。
    discrimitor_input_channels = 256

    # ResAE
    res_ae_config={
        'input_shape': (image_size, image_size, image_channels),
        'n_ResidualBlock': 4,
        'n_levels': 4,
        'z_dim': 10,
        'bottleneck_dim': discrimitor_input_channels,  # 真正的 latent vector size
        'bUseMultiResSkips': True
    }

    discrimitor_config = {
        'in_channels': discrimitor_input_channels,  # 輸入通道數
        'out_channels': 1,  # 最後一層輸出通道數
        'layer_depth': [4, 8, 16, 32, 64, 128, 256, 512],  # 每層的輸出通道數量
        'layer_kernel_size': [4, 4, 4, 4, 4, 4, 4, 4],  # 卷積核尺寸
        'layer_stride': [2, 2, 2, 2, 2, 2, 2, 2],  # 卷積步長
        'layer_padding': [1, 1, 1, 1, 1, 1, 1, 1],  # 卷積填充數
    }

    res_ae = ResNetAE_Conductor(config=res_ae_config)

    discriminator = Discriminator(config=discrimitor_config)