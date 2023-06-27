import os
import sys
# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
# sys.path.insert(0, parent_dir)
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import random
import matplotlib.pyplot as plt

class QRCodeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_list = os.listdir(root_dir)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image


def get_QRCode_dataloader(data_folder: str, batch_size=32, im_size=128, num_output_channels=3,
                          shuffle=False, num_workers=0,mean=np.array([0.5, 0.5, 0.5]), std=np.array([0.5, 0.5, 0.5])):
    """

    :param data_folder: qrcode 資料夾根目錄
    :param batch_size: dataloader size
    :param im_size: datasset image size
    :param num_output_channels: 轉換的 channel
    :param mean: 正規化的參數
    :param std: 正規化的參數
    :return: torch.Dataloader
    """
    assert os.path.isdir(data_folder)

    # 資料夾路徑和轉換器
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=num_output_channels),
        transforms.Resize((im_size, im_size)),  # 修改圖像大小
        transforms.ToTensor(),  # 轉換為張量
        transforms.Normalize(mean, std)  # 正規化
    ])

    # 建立自定義資料集
    dataset = QRCodeDataset(data_folder, transform=transform)

    # 創建資料載入器
    # pin_memory: 可以加快東西載入
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle
                                              , pin_memory=True, num_workers=num_workers)

    return data_loader

if __name__ == "__main__":
    # 資料夾路徑和轉換器
    data_folder = '../../data/processed/qrCodes'
    # mean = np.array([0.5, 0.5, 0.5])
    # std = np.array([0.5, 0.5, 0.5])
    #
    # assert os.path.isdir(data_folder)
    # transform = transforms.Compose([
    #     transforms.Grayscale(num_output_channels=3),
    #     transforms.Resize((128, 128)),  # 修改圖像大小
    #     transforms.ToTensor(),  # 轉換為張量
    #     transforms.Normalize(mean, std)  # 正規化
    # ])
    #
    # # 建立自定義資料集
    # dataset = QRCodeDataset(data_folder, transform=transform)
    #
    # # 創建資料載入器
    # batch_size = 32
    # data_loader = torch.my_utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    #
    # print("dataset length = ", len(dataset))
    #
    # print("stop")
    #
    batch_size = 32

    # 拿 Dataloader
    dataloader = get_QRCode_dataloader(data_folder, batch_size=batch_size, im_size=128, num_output_channels=3)

    # 拿一批
    batch_data = next(iter(dataloader))

    # A tensor
    array = batch_data[0]
    #
    std = np.array([0.5, 0.5, 0.5])
    mean = np.array([0.5, 0.5, 0.5])

    # 反向標準化
    denormalized_array = (array * std[:, None, None]) + mean[:, None, None]
    denormalized_array = np.clip(denormalized_array, 0, 1)
    denormalized_array = denormalized_array.numpy().transpose(1, 2, 0)

    plt.imshow(denormalized_array)  # 0 ~ 1
    plt.show()




