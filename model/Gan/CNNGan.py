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
from my_utils.dataloaders.single_qrCode_field_dataset import QRCodeDataset, get_QRCode_dataloader
#
from pathlib import Path
from my_utils.F import ensure_folder, timestamp
# 超參數設置
batch_size = 128
num_epochs = 100000
latent_size = 128

# G, D num of feature maps
ngf = 64
ndf = 64

# dataset 的 transform 超參數.
dataset_im_size = 128  # 這數值會影響到 鑑別器的輸出，鑑別器的輸出要是 'scalar', if 為 128，D(x)輸出 5x5
num_output_channels = 3

lr = 0.0002

# Beta1 hyperparameter for Adam optimizers
beta1 = 0.5

# for save
save_loss_png_period_of_epoehes = 10
save_weight_each_epoch = 500

# custom weights initialization called on ``netG`` and ``netD``
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def save_best_model(netG, netD, best_loss, current_loss, save_dir):
    if current_loss < best_loss:
        print("更新 best weight..., best =", current_loss)
        torch.save(netG.state_dict(), os.path.join(save_dir, 'best_generator.pth'))
        torch.save(netD.state_dict(), os.path.join(save_dir, 'best_discriminator.pth'))
        return current_loss
    else:
        return best_loss

# 生成器模型
class Generator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Generator, self).__init__()

        # layer_depth = [512, 256, 128, 64, num_output_channels]
        # feature_map_ratio = [8, 4, 2, 1, 1]
        # layer_kernel_size = [4, 4, 4, 4, 4]
        # layer_stride = [1, 2, 2, 2, 2]
        # layer_padding = [0, 1, 1, 1, 1]

        layer_depth = [512, 256, 128, 64, 32, num_output_channels]
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

        layers.append(nn.ConvTranspose2d(in_channels, num_output_channels, layer_kernel_size[num_layers-1],
                                         layer_stride[num_layers-1],
                                         layer_padding[num_layers-1], bias=False)
                      )
        layers.append(nn.Tanh())

        self.main = nn.Sequential(*layers)
        # for-loop 的 hard code 如下:
        # self.main = nn.Sequential(
        #     # input is Z, going into a convolution
        #     nn.ConvTranspose2d(latent_size, ngf * 8, 4, 1, 0, bias=False),
        #     nn.BatchNorm2d(ngf * 8),
        #     nn.ReLU(True),
        #     # state size. ``(ngf*8) x 4 x 4``
        #     nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ngf * 4),
        #     nn.ReLU(True),
        #     # state size. ``(ngf*4) x 8 x 8``
        #     nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ngf * 2),
        #     nn.ReLU(True),
        #     # state size. ``(ngf*2) x 16 x 16``
        #     nn.ConvTranspose2d(ngf * 2, ngf * 1, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ngf),
        #     nn.ReLU(True),
        #     # state size. ``(ngf) x 32 x 32``
        #     nn.ConvTranspose2d(ngf, num_output_channels, 4, 2, 1, bias=False),
        #     nn.Tanh()
        #     # state size. ``(nc) x 64 x 64``
        # )

    def forward(self, input):
        return self.main(input)

# 鑑別器模型
class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(in_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Flatten(),
            nn.Linear(5*5, 1),  # 強制輸出 scalar
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


def draw_loss(g_loss, d_loss, save_dir, name, idx):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(g_loss, label="G")
    plt.plot(d_loss, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    #plt.show()
    if Path(save_dir).is_dir():
        plt.savefig(f"{save_dir}/{name}_{idx}.png")
    else:
        os.makedirs(Path(save_dir))
        plt.savefig(f"{save_dir}/{name}_{idx}.png")

assert torch.cuda.is_available()

# check cuda is work
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 檢查是否有可用的 GPU，否則使用 CPU

# 初始化生成器
netG = Generator(in_channels=latent_size, out_channels=3).to(device)

#  初始化鑑別器
netD = Discriminator(in_channels=3).to(device)

# 隨機化
netG.apply(weights_init)
netD.apply(weights_init)

# torch transforms 正規化相關 參數
mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.5, 0.5, 0.5])


# 生成器  判別器 權重 如不繼承訓練就給None
__RESUME__WEIGHT__ = True
if __RESUME__WEIGHT__:
    __save_path = "../../output/CNNGan_train/weight_2023-06-04_PM_09h56m45s"
    generator_weight = torch.load(__save_path+"/best_generator.pth")
    discriminator_weight = torch.load(__save_path+"/best_discriminator.pth")
    netG.load_state_dict(generator_weight)
    netD.load_state_dict(discriminator_weight)


if __name__ == "__main__":
    # 資料夾路徑和轉換器
    data_folder_raw = '../../data/processed/qrCodes'
    assert os.path.isdir(data_folder_raw)
    #
    data_loader = get_QRCode_dataloader(data_folder_raw, batch_size=batch_size, im_size=dataset_im_size,
                                        num_output_channels=3)

    # 定義損失函數和優化器
    criterion = nn.BCELoss()

    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # 創建將用於visualization，用在生成器  64 V
    fixed_noise = torch.randn(dataset_im_size, latent_size, 1, 1, device=device)

    # 在訓練期間的 真假 標籤約定
    real_label = 1.
    fake_label = 0.

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    SAVE_TIME_STAMP = timestamp()
    CUS_NAME = "AFTER_1000_ET_"
    # loss 圖
    loss_png_save_path = f"../../output/CNNGan_train/{CUS_NAME}_{SAVE_TIME_STAMP}/loss"
    ensure_folder(loss_png_save_path, remake=True)

    # 訓練時的權重
    training_weight_save_path = f"../../output/CNNGan_train/{CUS_NAME}_{SAVE_TIME_STAMP}/weight"
    ensure_folder(training_weight_save_path, remake=True)

    #
    best_loss = float('inf')

    print("Starting Training Loop...")
    # 訓練 GAN
    for epoch in range(num_epochs):
        for batch_idx, real_images in enumerate(data_loader):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = real_images.to(device)
            b_size = real_cpu.size(0)
            # 標籤的
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D, D 的輸出必須是純量，須確保網路結構會讓一張圖片輸出一個純量
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, latent_size, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if batch_idx % 1 == 0:
                logg_str = ('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' %
                            (epoch, num_epochs, batch_idx, len(data_loader),
                             errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                with open(f"{loss_png_save_path}/log.txt", 'a', encoding='utf-8') as file:
                    file.write(logg_str+"\n")
                print(logg_str)

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (batch_idx == len(data_loader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1
        # draw
        if epoch % save_loss_png_period_of_epoehes == 0:
            draw_loss(g_loss=G_losses, d_loss=D_losses,
                      save_dir=loss_png_save_path, name="loss", idx=epoch)

        # 保存模型_訓練時
        if epoch % save_weight_each_epoch == 0:
            torch.save(netG.state_dict(), '{}/generator_model_{}.pth'.format(training_weight_save_path, str(epoch).zfill(4)))
            torch.save(netD.state_dict(), '{}/discriminator_model_{}.pth'.format(training_weight_save_path, str(epoch).zfill(4)))

        # 每個 epoch 結束後檢查是否最佳
        best_loss = save_best_model(netG, netD, best_loss, D_losses[-1], training_weight_save_path)
        print("=====")
        print("----> best: {%.4f} / current: {%.4f}" % (best_loss, D_losses[-1]))
        print("=====\n")
    # 保存模型 最後結束時
    torch.save(netG.state_dict(), f'{training_weight_save_path}/generator_model_last.pth')
    torch.save(netD.state_dict(), f'{training_weight_save_path}/discriminator_model_last.pth')
    # 最後畫一張
    draw_loss(g_loss=G_losses, d_loss=D_losses,
              save_dir=loss_png_save_path, name="!loss_final", idx=epoch)
