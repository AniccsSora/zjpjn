import os
import shutil
from datetime import datetime
import torch


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