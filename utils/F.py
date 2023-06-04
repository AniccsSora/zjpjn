import os
import shutil
from datetime import datetime

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