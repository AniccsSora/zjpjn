import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
from torchvision.utils import save_image
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import random
import matplotlib.pyplot as plt

# 超參數設置
batch_size = 64
num_epochs = 100
latent_size = 100
image_size = 28 * 28
hidden_size = 256


# 生成器模型
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, image_size),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.model(x)
        return x

# 鑑別器模型
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(image_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        return x


# 初始化生成器和鑑別器
generator = Generator()
discriminator = Discriminator()

# 定義損失函數和優化器
criterion = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=0.0002)
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002)


if __name__ == "__main__":


    # 資料夾路徑和轉換器
    data_folder_raw = '../../data/processed/qrCodes'

    assert os.path.isdir(data_folder_raw)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((128, 128)),  # 修改圖像大小
        transforms.ToTensor(),  # 轉換為張量
        transforms.Normalize(mean, std)  # 正規化
    ])
    # 訓練 GAN
    for epoch in range(num_epochs):
        for batch_idx, (real_images, _) in enumerate(data_loader):
            batch_size = real_images.size(0)

            # 將真實圖像轉換為平坦張量
            real_images = real_images.view(-1, image_size)

            # 創建標籤：真實圖像標籤為1，生成圖像標籤為0
            real_labels = Variable(torch.ones(batch_size, 1))
            fake_labels = Variable(torch.zeros(batch_size, 1))

            # 在訓練鑑別器時最小化真實圖像的損失
            optimizer_d.zero_grad()

            # 計算鑑別器對真實圖像的損失
            outputs = discriminator(real_images)
            real_loss = criterion(outputs, real_labels)
            real_score = outputs

            # 在訓練鑑別器時最小化生成圖像的損失
            z = Variable(torch.randn(batch_size, latent_size))
            fake_images = generator(z)
            outputs = discriminator(fake_images)
            fake_loss = criterion(outputs, fake_labels)
            fake_score = outputs

            # 總鑑別器損失為真實圖像損失加上生成圖像損失
            d_loss = real_loss + fake_loss

            # 反向傳播和優化鑑別器
            d_loss.backward()
            optimizer_d.step()

            # 在訓練生成器時最大化生成圖像的損失
            optimizer_g.zero_grad()
            z = Variable(torch.randn(batch_size, latent_size))
            fake_images = generator(z)
            outputs = discriminator(fake_images)
            g_loss = criterion(outputs, real_labels)

            # 反向傳播和優化生成器
            g_loss.backward()
            optimizer_g.step()

            # 每 200 個批次輸出訓練信息
            if (batch_idx + 1) % 200 == 0:
                print('Epoch [{}/{}], Step [{}/{}], 鑑別器損失: {:.4f}, 生成器損失: {:.4f}, '
                      '鑑別器得分: {:.2f}, 生成器得分: {:.2f}'
                      .format(epoch, num_epochs, batch_idx + 1, len(data_loader),
                              d_loss.item(), g_loss.item(), real_score.mean().item(), fake_score.mean().item()))

        # 每個 epoch 完成後保存生成的圖像
        if (epoch + 1) % 10 == 0:
            fake_images = fake_images.view(-1, 1, 28, 28)
            save_image(fake_images, 'generated_images-{}.png'.format(epoch + 1))

    # 保存模型
    torch.save(generator.state_dict(), 'generator_model.pth')
    torch.save(discriminator.state_dict(), 'discriminator_model.pth')
