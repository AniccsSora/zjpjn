from model.Autoencoder.model import ResNetAE_Conductor
from model.Gan.Discriminator import Discriminator
from model.Gan.Generator import Generator
from utils.F import ensure_folder, timestamp, save_best_model, write_log
from utils.dataloaders.single_qrCode_field_dataset import get_QRCode_dataloader
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import tqdm
from pathlib import Path
import os


def draw_loss_GD(g_loss, d_loss, save_dir, name, idx):
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

def draw_loss_AEGD(ae_loss, g_loss, d_loss, save_dir, name, idx):
    plt.figure(figsize=(10, 5))
    plt.title("Autoencoder, Generator and Discriminator Loss During Training")
    plt.plot(ae_loss, label="AE")
    plt.plot(g_loss, label="G")
    plt.plot(d_loss, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    if Path(save_dir).is_dir():
        plt.savefig(f"{save_dir}/{name}_{idx}.png")
    else:
        os.makedirs(Path(save_dir))
        plt.savefig(f"{save_dir}/{name}_{idx}.png")

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


__RESUME__WEIGHT__ = True
def  GO__RESUME__WEIGHT():
    __save_path = "../../output/123132132"
    generator_weight = torch.load(__save_path+"/best_generator.pth")
    discriminator_weight = torch.load(__save_path+"/best_discriminator.pth")
    netG.load_state_dict(generator_weight)
    netD.load_state_dict(discriminator_weight)
    netAE.load_state_dict(discriminator_weight)

if __name__== "__main__":

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
        'layer_depth': [1024, 512, 256, 32, 16, 8, 3],  # 每層的輸出通道數量
        'layer_kernel_size': [4, 4, 4, 4, 4, 4, 4, 4, 4],  # 卷積核尺寸
        'layer_stride': [2, 2, 2, 2, 2, 2, 2, 2, 2],  # 卷積步長
        'layer_padding': [0, 1, 1, 1, 1, 1, 1, 1, 1],  # 卷積填充數
    }

    # dataloader Parameters
    data_folder_raw = '../data/processed/qrCodes'

    # 影響 tranning 的參數
    batch_size = 32
    dataset_im_size = 32

    lr = 0.0002

    # Beta1 hyperparameter for Adam optimizers
    beta1 = 0.5

    # 訓練次數
    epochs = 1000

    """
    =======================================================================
    ========================= Hyper Params End ============================
    =======================================================================
    """
    #
    netAE = ResNetAE_Conductor(config=res_ae_config).to(device)
    netD = Discriminator(config=discrimitor_config).to(device)
    netG = Generator(config=G_config).to(device)

    # 隨機化
    netG.apply(weights_init)
    netD.apply(weights_init)
    """
    =======================================================================
    ============================= Model End ===============================
    =======================================================================
    """


    data_loader = get_QRCode_dataloader(data_folder_raw, batch_size=batch_size, im_size=image_size,
                                        num_output_channels=3)
    """
    =======================================================================
    ============================ Training Data ============================
    =======================================================================
    """

    criterion_gan = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    criterion_ae = nn.MSELoss()
    optimizerAE = optim.Adam(netAE.parameters(), lr=lr, betas=(beta1, 0.999))

    # 在訓練期間的 真假 標籤約定
    real_label = 1.
    fake_label = 0.

    # 多久繪製一張圖片
    save_loss_png_period_of_epoehes = 10
    # 多久寫一次 loss 到 file 內
    save_loss2log = 10
    # 多久強制存 model 一次
    save_weight_each_epoch = 50

    # 決定最佳時，存檔用的
    best_D_loss = float('inf')
    best_G_loss = float('inf')
    best_AE_loss = float('inf')

    """
    =======================================================================
    ============================ Training Use  ============================
    =======================================================================
    """

    # Training Save Path
    save_path = '../output/DAEGAN/' + timestamp()
    ensure_folder(save_path)

    # history weight save path
    history_weight_save_path = os.path.join(save_path, 'history_weight')
    ensure_folder(history_weight_save_path)

    # loss 圖片用的路徑
    loss_png_save_path = os.path.join(save_path, 'loss_png')
    ensure_folder(loss_png_save_path)

    """
    ===============================================================
    ============================ PATH  ============================
    ===============================================================
    """

    # 訓練時期的數據紀錄:
    G_losses = []
    D_losses = []
    AE_losses = []

    #pbar = tqdm.tqdm(range(epochs))

    # Training Loop
    for epoch in range(epochs):
        #pbar.set_description("Processing Epoch {}".format(epoch))
        #inner_pbar = tqdm.tqdm(data_loader)
        for batch_idx, real_images in enumerate(data_loader):
            #inner_pbar.set_description("== Processing Batch {}".format(batch_idx))
            """ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            ===============  AE Session Start ===============
            """
            # Train AutoEncoder
            netAE.zero_grad()

            # Format batch
            real_on_device = real_images.to(device)

            # Forward pass real batch through AE
            output = netAE(real_on_device)

            # Calculate loss on all-real batch
            loss_AE = criterion_ae(output, real_on_device)

            # Calculate gradients for AE in backward pass
            loss_AE.backward()
            #
            # AE loss
            AE_x = loss_AE.mean().item()

            optimizerAE.step()
            """ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            ===============  GAN Session Start ===============
            """
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch 前面有了這邊就不用
            # real_on_device = real_images.to(device)
            # 鑑別器看了所有圖片，所以都是 1
            # 標籤要用。因為是真實圖片，所以都用 real_label, (通常=1)
            b_size = real_on_device.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D, D 的輸出必須是純量，須確保網路結構會讓一張圖片輸出一個純量
            output_D = netD(real_on_device).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion_gan(output_D, label)
            errD_real.backward()
            D_x = output_D.mean().item()  # netD 的輸出平均
            #  D network: 第二階段處理 Fake Data
            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, bottle_neck_dims, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            # G() 所生出來的圖片要 "斷開鎖鏈"，重給到 D()，所以 detach()。
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion_gan(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()
            """
            ===============  Discriminator Session Done ===============
            """
            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion_gan(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()
            """
            ===============  Generator Session Done ===============
            """

            # Output training stats
            if batch_idx % 1 == 0:
                logg_str = ('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f\t Loss_AE: %.4f' %
                            (epoch, epochs, batch_idx, len(data_loader),
                             errD.item(), errG.item(), D_x, D_G_z1, D_G_z2, loss_AE.item()))
                write_log(logg_str, f"{save_path}/log.txt")
                print(logg_str)

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            AE_losses.append(loss_AE.item())
        #
        #  A Batch End
        #

        # draw
        if epoch % save_loss_png_period_of_epoehes == 0:
            draw_loss_GD(g_loss=G_losses, d_loss=D_losses,
                      save_dir=loss_png_save_path, name="loss", idx=epoch)
        # 繪製 有 AE_loss, G_losses, D_losses 的圖
        if epoch % save_loss_png_period_of_epoehes == 0:
            draw_loss_AEGD(ae_loss=AE_losses, g_loss=G_losses, d_loss=D_losses,
                           save_dir=loss_png_save_path, name="loss", idx=epoch)

        # 保存模型_訓練時
        if epoch % save_weight_each_epoch == 0:
            torch.save(netG.state_dict(),
                       '{}/generator_model_{}.pt'.format(history_weight_save_path, str(epoch).zfill(4)))
            torch.save(netD.state_dict(),
                       '{}/discriminator_model_{}.pt'.format(history_weight_save_path, str(epoch).zfill(4)))
            torch.save(netAE.state_dict(),
                       '{}/autoencoder_model_{}.pt'.format(history_weight_save_path, str(epoch).zfill(4)))

        # epoch 結束後，決定是否是最佳 model 更新他們
        # (net, best_loss, current_loss, save_dir, save_pt_name)
        best_D_loss, best_G_loss, best_AE_loss = \
            save_best_model(netD, best_D_loss, D_losses[-1], save_path, "best_D.pt"), \
            save_best_model(netG, best_G_loss, G_losses[-1], save_path, "best_G.pt"), \
            save_best_model(netAE, best_AE_loss, AE_losses[-1], save_path, "best_AE.pt")

        print("=== Best Update ===")
        if best_D_loss != D_losses[-1]:
            print("-- D --> best: {%.4f} / current: {%.4f}" % (best_D_loss, D_losses[-1]))
        else:
            print(" D no update")
        if best_G_loss != G_losses[-1]:
            print("-- G --> best: {%.4f} / current: {%.4f}" % (best_G_loss, G_losses[-1]))
        else:
            print(" G no update")
        if best_AE_loss != AE_losses[-1]:
            print("-- AE -> best: {%.4f} / current: {%.4f}" % (best_AE_loss, AE_losses[-1]))
        else:
            print(" AE no update")
        print("=====\n")
    #
    #  ALL Epochs End
    #
    # 最後畫一張
    draw_loss(g_loss=G_losses, d_loss=D_losses,
              save_dir=loss_png_save_path, name="!loss_final", idx=epoch)


