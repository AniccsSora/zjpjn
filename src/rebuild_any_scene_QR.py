import os
import glob
from my_utils.useYolo import detect_qr_field

if __name__ == "__main__":
    # 讀取資料夾內的所有圖片到 list 等待後續處理
    dir = r"../my_utils\data\raw_qr"
    assert os.path.isdir(dir)
    images = glob.glob(os.path.join(dir, "*.*"))

    for image in images:
        res = detect_qr_field(image)
        print(res)