import glob
from my_utils.useYolo import detect_qr_field
from model.Autoencoder.model import ResNetAE_Conductor
from model.Gan.Discriminator import Discriminator
from model.Gan.Generator import Generator
import torch
import os
from torchvision import transforms
import numpy as np
import cv2
from my_utils.F import ensure_folder, timestamp

if __name__ == "__JUST_TEST__":
    # 讀取資料夾內的所有圖片到 list 等待後續處理
    dir = r"../my_utils\data\raw_qr"
    assert os.path.isdir(dir)
    images = glob.glob(os.path.join(dir, "*.*"))

    for image in images:
        res = detect_qr_field(image)
        print(res)

mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.5, 0.5, 0.5])

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((256, 256)),  # 修改圖像大小
    transforms.ToTensor(),  # 轉換為張量
    transforms.Normalize(mean, std)  # 正規化
])

if __name__ == "__main__":

    # rebuild save path
    save_path="../output/rebuild_result_"+timestamp()
    ensure_folder(save_path, remake=False)

    # 這 model 會用到
    image_size = 256
    image_channels = 3

    # 鑑別器輸入大小，應與 AE latent vector size 維度相當。
    bottle_neck_dims = 256

    #
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
        'in_channels': image_channels,  # 輸入的圖片通道數
        'out_channels': 1,  # 最後一層輸出通道數
        'layer_depth': [4, 8, 16, 32, 64, 128, 256, 512],  # 每層的輸出通道數量
        'layer_kernel_size': [4, 4, 4, 4, 4, 4, 4, 4],  # 卷積核尺寸
        'layer_stride': [2, 2, 2, 2, 2, 2, 2, 2],  # 卷積步長
        'layer_padding': [1, 1, 1, 1, 1, 1, 1, 1],  # 卷積填充數
    }

    G_config = {
        'in_channels': bottle_neck_dims,
        'out_channels': 3,  # 最後一層輸出通道數
        'layer_depth': [1024, 512, 256, 32, 16, 8, 3],  # 每層的輸出通道數量
        'layer_kernel_size': [4, 4, 4, 4, 4, 4, 4, 4, 4],  # 卷積核尺寸
        'layer_stride': [2, 2, 2, 2, 2, 2, 2, 2, 2],  # 卷積步長
        'layer_padding': [0, 1, 1, 1, 1, 1, 1, 1, 1],  # 卷積填充數
    }

    #
    netAE = ResNetAE_Conductor(config=res_ae_config).to(device)
    netD = Discriminator(config=discrimitor_config).to(device)
    netG = Generator(config=G_config).to(device)

    #
    weight_ae = r"..."
    weight_d = r"..."
    weight_g = r"..."

    # load weight
    netAE.load_state_dict(torch.load(weight_ae))
    netD.load_state_dict(torch.load(weight_d))
    netG.load_state_dict(torch.load(weight_g))

    # eval mode
    netAE.eval()
    netD.eval()
    netG.eval()

    # Generator epoches
    g_epoches = 10

    #
    fake_label = 0.

    # load rebuild data
    dir = r"../my_utils\data\raw_qr"
    assert os.path.isdir(dir)
    images = glob.glob(os.path.join(dir, "*.*"))




    for image in images:
        # is QR code???
        unknow_fields_list=detect_qr_field(image)

        if len(unknow_fields_list) == 0:
            print("pass a image.")
            continue

        for field in unknow_fields_list:
            print("image shape: ", field.shape)

            # make image to a tensor batch

            tensor_field = transform(field)

            # check transform size = 3x256x256 ???
            assert tensor_field.shape == (3, 256, 256)

            # make it to a batch
            a_batch = tensor_field.unsqueeze(0)

            with torch.no_grad():
                # use AE's Encoder to generate latent vector
                latent_vector = netAE.encode(a_batch)

                # use G to generate image
                fake_image = netG(latent_vector)

                # use discrimitor to check
                res_D_loss = netD(fake_image)

                print("D loss: ", res_D_loss)

                # update G
                netG.zero_grad()
                res_D_loss.backward()
                netG.step()

            # cehck out batch size is 1
            assert fake_image.shape[0] == 1

            rebuild_image = fake_image[0].cpu().numpy()

            # make it to 0~255
            rebuild_image = (rebuild_image * 255).astype(np.uint8)

            # make is (w,h,3)
            rebuild_image = np.transpose(rebuild_image, (1, 2, 0))

            # save np image
            cv2.imwrite(os.path.join(save_path, "rebuild_image.jpg"), rebuild_image)
