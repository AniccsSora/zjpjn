import os
import shutil
from datetime import datetime
import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import random
import qrcode

def create_grid_image2(left_images, right_images, grid_size, save_path=None, dpi=300):
    assert left_images.shape[0] >= grid_size[0] * grid_size[1]
    assert right_images.shape[0] >= grid_size[0] * grid_size[1]

    plt.close("all")

    # Create the left grid image
    left_grid = vutils.make_grid(left_images[:grid_size[0] * grid_size[1]], nrow=grid_size[0], padding=2,
                                 normalize=True)

    # Create the right grid image
    right_grid = vutils.make_grid(right_images[:grid_size[0] * grid_size[1]], nrow=grid_size[0], padding=2,
                                  normalize=True)

    # Set up the plot
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Display the left grid image
    axs[0].axis('off')
    axs[0].imshow(left_grid.permute(1, 2, 0))

    # Display the right grid image
    axs[1].axis('off')
    axs[1].imshow(right_grid.permute(1, 2, 0))

    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()

def create_grid_image(batch_images, grid_size, save_path=None, dpi=300):
    assert batch_images.shape[0] >= grid_size[0]*grid_size[1]
    plt.close("all")
    grid = vutils.make_grid(batch_images[:grid_size[0]*grid_size[1]], nrow=grid_size[0], padding=2, normalize=True)
    plt.axis('off')
    plt.imshow(grid.permute(1, 2, 0))
    if save_path is not None:
        plt.savefig(save_path, dpi=dpi)
    else:
        plt.show()





def write_log(log_str, log_file):
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_str + '\n')

def save_best_model(net, best_loss, current_loss, save_dir, save_pt_name):
    if current_loss < best_loss:
        torch.save(net.state_dict(), os.path.join(save_dir, f'best_{save_pt_name}.pt'))
        return current_loss
    else:
        return best_loss

def timestamp():
    return datetime.now().strftime("%Y-%m-%d_%p_%Ih%Mm%Ss")

def ensure_folder(folder_path, remake=False):
    """
    確保某個資料夾必定存在，因為會重新建立。

    @param folder_path:
        要建立的資料夾名。

    @param remake: (Default False)
        如果為 True，會刪除舊的目錄再重新建立。
    """
    if os.path.isdir(folder_path):
        if not remake:
            return
        else:
            shutil.rmtree(folder_path)
            os.makedirs(folder_path, 0o755)
    else:
        os.makedirs(folder_path, 0o755)


def generate_random_message(qr_version):
    """
    取得符合對應 qrcode version 的 message 長度。
    """

    # Set the maximum length limit for each QR code version
    version_limits = {
        1: 25,
        2: 47,
        3: 77,
        4: 114,
        5: 154,
        6: 195,
        7: 224,
        8: 279,
        9: 335,
        10: 395,
        11: 468,
        12: 535,
        13: 619,
        14: 667,
        15: 758,
        16: 854,
        17: 938,
        18: 1046,
        19: 1153,
        20: 1249,
        21: 1352,
        22: 1460,
        23: 1588,
        24: 1704,
        25: 1853,
        26: 1990,
        27: 2132,
        28: 2223,
        29: 2369,
        30: 2520,
        31: 2677,
        32: 2840,
        33: 3009,
        34: 3183,
        35: 3351,
        36: 3537,
        37: 3729,
        38: 3927,
        39: 4087,
        40: 4296
    }
    # Check if the QR code version is valid
    if qr_version < 1 or qr_version > 40:
        raise ValueError("Invalid QR version. Valid versions are 1 to 40.")

    # Get the maximum length limit for the specified version
    max_length = version_limits[qr_version]
    min_length = version_limits[qr_version-1] if qr_version!= 1 else version_limits[1]
    # Generate a random message
    message_length = random.randint(max_length // 2, max_length)  # Randomly choose the message length
    random_message = ''.join(random.choices(string.ascii_letters + string.digits + string.punctuation, k=message_length))

    return random_message


def generate_std_qr(FOLDER, NUMBER_OF_QR):
    # FOLDER 存在哪邊?
    # NUMBER_OF_QR 生幾張?
    if not os.path.isdir(FOLDER):
        os.mkdir(FOLDER)

    # Adjusted sample size to fit the population
    for data in range(NUMBER_OF_QR):
        version =  random.choice(range(1, 14))
        qr = qrcode.QRCode(
            version=version,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=10,
            border=1
        )
        qr.add_data(data)
        qr.make(fit=True)
        image = qr.make_image(fill_color="black", back_color="white")
        filename = f"{FOLDER}/QR_Version_{version}_Data_{data}.png"
        image.save(filename)