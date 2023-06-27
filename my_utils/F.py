import glob
import os
import shutil
from datetime import datetime
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import random
import qrcode
from PIL import Image, ImageDraw, ImageFont
import os
import cv2
from pyzbar import pyzbar
from PIL import Image
import numpy as np
import re


'''
make_gif(r"C:\cgit\zjpj-new\output\ResAE_HalF_ResGAN_RMSprop\2023-06-19_AM_01h23m06s\loss_png\GD",
             r"C:\cgit\zjpj-new\output\ResAE_HalF_ResGAN_RMSprop\2023-06-19_AM_01h23m06s\loss_GD.gif", 200)
'''
def make_gif(image_folder, output_file, time_per_frame=500, image_filter=".png"):
    # List all PNG files in the folder
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(image_filter)])
    image_files.sort(key=lambda x: int(re.search(r'\d+', x).group()))

    # Open the first image to get size information
    first_image = Image.open(os.path.join(image_folder, image_files[0]))
    width, height = first_image.size

    # Create a new image with the timeline and append the original images
    timeline_height = 30
    timeline_image = Image.new("RGBA", (width, height + timeline_height))
    timeline_image.paste(first_image, (0, timeline_height))

    # Define the start and end markers
    start_marker = 10
    end_marker = width - 10

    # Append the rest of the images
    frames = []
    durations = []

    i_factor = (width-1)//len(image_files)
    progress_axis_width = width//i_factor

    for i, f in enumerate(image_files[1:], start=1):
        image = Image.open(os.path.join(image_folder, f))
        frame_image = timeline_image.copy()
        frame_image.paste(image, (0, timeline_height))

        # Create a new frame with the timeline markers
        draw = ImageDraw.Draw(frame_image)
        draw.line([(start_marker + i*i_factor - 1, 0), (start_marker + i*i_factor - 1, timeline_height-1)], fill="blue", width=progress_axis_width)
        draw.line([(end_marker, 0), (end_marker, timeline_height-1)], fill="red", width=10)

        # Add the frame to the list of frames
        frames.append(frame_image)

        # Specify the duration for the frame (adjust as needed)
        durations.append(time_per_frame)

    #first frame wait more times
    durations[0] = 1300
    # Save the frames as an animated GIF
    frames[0].save("{}.gif".format(output_file), save_all=True, append_images=frames[1:], duration=durations, loop=0)


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


def is_qrcode(image):
    if isinstance(image, str):
        #image = cv2.imread(image)
        image = Image.open(image)
        # pillow to np array
        image = np.asarray(image)
    elif isinstance(image, np.ndarray):
        pass
    elif isinstance(image, Image.Image):
        image = np.asarray(image)
    else:
        raise ValueError("not support type:", type(image))

    if len(pyzbar.decode(image))> 0:
        return True
    else:
        return False


def extract_qr_code_(image_path):
    # 讀取圖片
    image = cv2.imread(image_path)

    # 將圖片轉換為灰階
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 檢測 QR Code
    decoded_objects = pyzbar.decode(gray)

    # 如果有檢測到 QR Code
    if decoded_objects:
        # 取得 QR Code 區塊的邊界框座標
        x, y, w, h = decoded_objects[0].rect

        # 切割 QR Code 區塊
        qr_code = image[y:y+h, x:x+w]

        # 轉換成 PIL Image 物件
        pil_image = Image.fromarray(cv2.cvtColor(qr_code, cv2.COLOR_BGR2RGB))

        return pil_image
    else:
        print('未檢測到 QR Code')
        return None


def extract_qr_codes(image_path):
    # 讀取圖片
    image = cv2.imread(image_path)

    # 將圖片轉換為灰階
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 檢測 QR Code
    decoded_objects = pyzbar.decode(gray)

    qr_code_images = []  # 存放切割後的 QR Code 圖片

    # 如果有檢測到 QR Code
    if decoded_objects:
        # 對每個 QR Code 執行切割
        for obj in decoded_objects:
            # 取得 QR Code 區塊的邊界框座標
            x, y, w, h = obj.rect

            # 切割 QR Code 區塊
            qr_code = image[y:y+h, x:x+w]

            # 轉換成 PIL Image 物件
            pil_image = Image.fromarray(cv2.cvtColor(qr_code, cv2.COLOR_BGR2RGB))

            qr_code_images.append(pil_image)

    return qr_code_images


if __name__ == "__main__":
    print("AAA")

    dir = r"C:\cgit\AutoCrawler\download\all"
    dist = r"C:\cgit\AutoCrawler\download\qrcode"
    images = glob.glob(os.path.join(dir, "*.*"))

    passed = 0

    images_list = [_ for _ in images]

    i = 9445
    images_list = images_list[i:]
    for image in images_list:
        print("process > ", image)
        try:
            if is_qrcode(image):
                #print(image)
                # use pillow reopoen and save to dist
                imagename = image
                image = Image.open(image)
                # save to rgb
                image = image.convert("RGB")

                # 切成 qrcode 再存檔
                ppils_list = extract_qr_codes(imagename)
                if ppils_list is []:
                    print("no qr code inside:", imagename)
                    continue
                _idx=1
                for ppil in ppils_list:
                    ppil.save(os.path.join(dist, "{}_{}.jpg".format(Path(imagename).stem, _idx)))
                    _idx += 1
                #image.save(os.path.join(dist, os.path.basename(imagename)))
        except:
            print("pass:", image)
            passed += 1
            continue
    print("total pass:", passed)