import pytest
import torch
from model.Gan.Generator import Generator

@pytest.fixture(scope="function")
def setup_generator(request):
    test_params = request.param

    batch = test_params['batch']
    # 網路定義
    config = {
        'in_channels': test_params['in_channels'],  # 輸入通道數
        'out_channels': test_params['out_channels'],  # 最後一層輸出通道數
        'layer_depth': test_params['layer_depth'],  # 每層的輸出通道數量
        'layer_kernel_size': test_params['layer_kernel_size'],  # 卷積核尺寸
        'layer_stride': test_params['layer_stride'],  # 卷積步長
        'layer_padding': test_params['layer_padding'],  # 卷積填充數
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 初始化 Generator
    generator = Generator(config).to(device)
                                                    
    # 生成假批次，輸入維度為 (batch_size, <G_接受維度>, 1, 1)
    z = torch.randn(batch, test_params['in_channels'], 1, 1, device=device)
    output = generator(z)


    return output, test_params

test_configs = [
    {   # case 1 : in: 'in_channels': 256,> out: 'assert_image_size': 128
        'batch': 5,
        #  in_channels Generator 接受的 1維輸入
        'in_channels': 256,   # 1維vector,輸入通道數，G 吃的一維向量大小
        'out_channels': 3,                     # 最後一層輸出通道數
        'layer_depth': [1024, 512, 256, 32, 16, 3],       # 每層的輸出通道數量
        'layer_kernel_size': [4, 4, 4, 4, 4, 4, 4, 4],         # 卷積核尺寸
        'layer_stride': [2, 2, 2, 2, 2, 2, 2, 2],              # 卷積步長
        'layer_padding': [0, 1, 1, 1, 1, 1, 1, 1],             # 卷積填充數
        # 以下參數是預期的 model out image size
        'assert_image_size': 128
    },
    {   # case 2 : in: 'in_channels': 128,> out: 'assert_image_size': 128
        'batch': 2,
        #  in_channels Generator 接受的 1維輸入
        'in_channels': 128,   # 1維vector,輸入通道數，G 吃的一維向量大小
        'out_channels': 3,                     # 最後一層輸出通道數
        'layer_depth': [1024, 512, 256, 32, 16, 3],       # 每層的輸出通道數量
        'layer_kernel_size': [4, 4, 4, 4, 4, 4, 4, 4],         # 卷積核尺寸
        'layer_stride': [2, 2, 2, 2, 2, 2, 2, 2],              # 卷積步長
        'layer_padding': [0, 1, 1, 1, 1, 1, 1, 1],             # 卷積填充數
        # 以下參數是預期的 model out image size
        'assert_image_size': 128
    },
    {   # case 3 : in: 'in_channels': 128> out: 'assert_image_size': 32
        'batch': 3,
        #  in_channels Generator 接受的 1維輸入
        'in_channels': 128,   # 1維vector,輸入通道數，G 吃的一維向量大小
        'out_channels': 3,                     # 最後一層輸出通道數
        'layer_depth': [512, 256, 128, 64],       # 每層的輸出通道數量， * 加上越多層數，圖片越大
        'layer_kernel_size': [4, 4, 4, 4, 4, 4],         # 卷積核尺寸
        'layer_stride': [2, 2, 2, 2, 2, 2],              # 卷積步長
        'layer_padding': [0, 1, 1, 1, 1, 1],             # 卷積填充數
        # 以下參數是預期的 model out image size
        'assert_image_size': 32
    },
]

@pytest.mark.parametrize('setup_generator', test_configs, indirect=True)
def test_generator_output_shape(setup_generator):
    output, test_params = setup_generator
    # batch dims
    assert output.dim() == 4
    # batch size check
    assert output.size()[0] == test_params['batch']
    # channel[1], width[2], height[3]
    assert output.size()[1] == test_params['out_channels']
    assert output.size()[2] == test_params['assert_image_size']
    assert output.size()[3] == test_params['assert_image_size']
