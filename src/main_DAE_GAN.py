from model.Autoencoder.model import ResNetAE_Conductor
from model.Gan.Discriminator import Discriminator
from model.Gan.Generator import Generator
from my_utils.F import ensure_folder, timestamp, save_best_model, write_log
from my_utils.F import create_grid_image, create_grid_image2
from my_utils.dataloaders.single_qrCode_field_dataset import get_QRCode_dataloader
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import tqdm
from pathlib import Path
import os
import numpy as np
import json


# THE SEED!!!
seed = 999
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

def get_optimizer(use_optim, model, lr, Adam_beta1=0.5):
    # torch optimizers list

    if use_optim=='Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=(Adam_beta1, 0.999))
    elif use_optim=='SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif use_optim=='RMSprop':
        """
        alpha = 衰減率，越低可以越快的調整學習率。
        weight_decay = L2正則化，防止過擬合。增加他會減少模型福砸度 提高泛化能力。
        """
        alpha = 0.99
        weight_decay = 0.00005
        momentum = 0.9
        optimizer = optim.RMSprop(model.parameters(), lr=lr, alpha=alpha, eps=1e-08,
                                  weight_decay=weight_decay,
                                  momentum=momentum,
                                  centered=False)
    else:
        assert False

    return optimizer

def draw_loss_AE(ae_loss, save_dir, name, idx):
    plt.figure(figsize=(10, 5))
    plt.title("Autoencoder Loss During Training")
    plt.plot(ae_loss, label="AE")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    #plt.show()
    if Path(save_dir).is_dir():
        plt.savefig(f"{save_dir}/{name}_{idx}.png")
    else:
        os.makedirs(Path(save_dir))
        plt.savefig(f"{save_dir}/{name}_{idx}.png")

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


def  GO__RESUME__WEIGHT(resume_ae, resume_g, resume_d):
    __save_path = "./"
    generator_weight = torch.load(__save_path+"/best_best_G.pt")  # 這是 custom
    discriminator_weight = torch.load(__save_path+"/best_best_D.pt")  # 這是 custom
    ae_weight = torch.load(__save_path + "/best_best_AE.pt")  # 這是 custom

    # hard code below
    if resume_g:
        netG.load_state_dict(generator_weight)
    if resume_d:
        netD.load_state_dict(discriminator_weight)
    if resume_ae:
        netAE.load_state_dict(ae_weight)

if __name__ == "__main__":
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
    #data_folder_raw = '../data/processed/qrCodes'
    data_folder_raw = '../data/processed/qrcode_6000'

    # 影響 tranning 的參數
    batch_size = 32
    dataset_im_size = 32

    lr = 0.0001

    # Beta1 hyperparameter for Adam optimizers
    beta1 = 0.5

    # 訓練次數
    epochs = 20000

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
    netAE.apply(weights_init)

    # 是否續練?
    __RESUME__WEIGHT__ = False

    #
    if __RESUME__WEIGHT__:
        print("Load previouse weight...")
        GO__RESUME__WEIGHT(resume_ae=True, resume_g=True, resume_d=False)
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
    #optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerD = get_optimizer('RMSprop', netD, lr=lr)
    #optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = get_optimizer('RMSprop', netG, lr=lr)

    criterion_ae = nn.MSELoss()
    #optimizerAE = optim.Adam(netAE.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerAE = get_optimizer('RMSprop', netAE, lr=lr)

    # 在訓練期間的 真假 標籤約定
    real_label = 1.
    fake_label = 0.

    # 多久繪製一張圖片
    save_loss_png_period_of_epoehes = 5
    # 多久寫一次 loss 到 file 內
    save_loss2log = 5
    # 多久強制存 model 一次
    save_weight_each_epoch = 25
    # 多久存一次 visualized image
    save_visualized_image_each_epoch = 5

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

    # 視覺化圖片存檔用資料夾
    visulize_G_save_path = os.path.join(save_path, 'visulize/Gen_Images')
    ensure_folder(visulize_G_save_path)

    # 視覺化圖片存檔用資料夾
    visulize_AE_save_path = os.path.join(save_path, 'visulize/AE_Images')
    ensure_folder(visulize_AE_save_path)
    """
    ===============================================================
    ============================ PATH  ============================
    ===============================================================
    """

    # write optimizer state dict to log file.
    optimizer_state_dict_D = optimizerD.state_dict()
    optimizer_state_dict_G = optimizerG.state_dict()
    optimizer_state_dict_AE = optimizerAE.state_dict()

    _name_list = ['discriminator_optm_params', 'generator_optm_params', 'autoencoder_optm_params']
    _opti_list = [optimizer_state_dict_D, optimizer_state_dict_G, optimizer_state_dict_AE]
    for n, o in zip(_name_list, _opti_list):
        write_log("vvvvvvvvvvvvvvvvvvvv {} PARAMS START vvvvvvvvvvvvvvvvvvv".format(n), f"{save_path}/log.txt")
        formatted_dict = json.dumps(o, indent=4)
        write_log(formatted_dict, f"{save_path}/log.txt")
        write_log("^^^^^^^^^^^^^^^^^^^^ {} PARAMS END ^^^^^^^^^^^^^^^^^^^^".format(n), f"{save_path}/log.txt")

    """
    ============================================================================
    ============================ before start logging ==========================
    ============================================================================
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

        G_batch_losses = []
        D_batch_losses = []
        AE_batch_losses = []
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
            # noise = torch.randn(b_size, bottle_neck_dims, 1, 1, device=device)
            #
            # 從 Encoder 來的 noise
            with torch.no_grad():
                # (<b_size>, bottle_neck_dims) -> (<b_size>, bottle_neck_dims, 1, 1)
                noise = netAE.encode(real_on_device).unsqueeze(-1).unsqueeze(-1)

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
            G_batch_losses.append(errG.item())
            D_batch_losses.append(errD.item())
            AE_batch_losses.append(loss_AE.item())
        #
        #  A Batch End
        #

        # Append average loss
        G_losses.append(np.array(G_batch_losses).mean())
        D_losses.append(np.array(D_batch_losses).mean())
        AE_losses.append(np.array(AE_batch_losses).mean())

        # draw
        if epoch % save_loss_png_period_of_epoehes == 0:
            draw_loss_GD(g_loss=G_losses, d_loss=D_losses,
                      save_dir=loss_png_save_path+"/GD", name="loss", idx=epoch)
        # 繪製 有 AE_loss, G_losses, D_losses 的圖
        if epoch % save_loss_png_period_of_epoehes == 0:
            draw_loss_AEGD(ae_loss=AE_losses, g_loss=G_losses, d_loss=D_losses,
                           save_dir=loss_png_save_path+"/AEGD", name="loss", idx=epoch)
        # 單純繪製只有 AE_loss 的圖
        if epoch % save_loss_png_period_of_epoehes == 0:
            draw_loss_AE(ae_loss=AE_losses,
                         save_dir=loss_png_save_path+"/AE", name="loss", idx=epoch)
        # 保存模型_訓練時
        if epoch % save_weight_each_epoch == 0:
            torch.save(netG.state_dict(),
                       '{}/generator_model_{}.pt'.format(history_weight_save_path, str(epoch).zfill(4)))
            torch.save(netD.state_dict(),
                       '{}/discriminator_model_{}.pt'.format(history_weight_save_path, str(epoch).zfill(4)))
            torch.save(netAE.state_dict(),
                       '{}/autoencoder_model_{}.pt'.format(history_weight_save_path, str(epoch).zfill(4)))
        # Save visualized images
        if epoch % save_visualized_image_each_epoch == 0:
            with torch.no_grad():
                # get batch from data_loader
                real_on_device = next(iter(data_loader))
                # input to autoencoder's encoder part
                latent_from_Encoder = netAE.encode(real_on_device.to(device)).unsqueeze(-1).unsqueeze(-1)
                val_generated_images = netG(latent_from_Encoder).detach().cpu()
                ae_rebuild_images = netAE(real_on_device.to(device)).detach().cpu()
                # save images to 'visulize_G_save_path'
                create_grid_image(val_generated_images, grid_size=(4, 4), dpi=150,
                                  save_path=visulize_G_save_path+"/epoch_{}.png".format(epoch))
                # Save image for AutoEncoder
                create_grid_image2(real_on_device, ae_rebuild_images, grid_size=(4, 4), dpi=150,
                                   save_path=visulize_AE_save_path + "/epoch_{}.png".format(epoch))


        # epoch 結束後，決定是否是最佳 model 更新他們
        # (net, best_loss, current_loss, save_dir, save_pt_name)
        best_D_loss, best_G_loss, best_AE_loss = \
            save_best_model(netD, best_D_loss, D_losses[-1], save_path, "best_D"), \
            save_best_model(netG, best_G_loss, G_losses[-1], save_path, "best_G"), \
            save_best_model(netAE, best_AE_loss, AE_losses[-1], save_path, "best_AE")

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


