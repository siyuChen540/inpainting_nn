#########################
# Early Stopping
#########################

import os

import torch
import numpy as np
from torch import nn


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model_state, epoch, save_path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(val_loss, model_state, epoch, save_path)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save_checkpoint(val_loss, model_state, epoch, save_path)
            self.counter = 0

    def _save_checkpoint(self, val_loss, model_state, epoch, save_path):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model_state, os.path.join(save_path, f"checkpoint_{epoch}_{val_loss:.6f}.pth.tar"))
        self.val_loss_min = val_loss

#########################
# Directory Setup
#########################
def record_dir_setting_create(father_dir: str, mark: str) -> str:
    dir_temp = os.path.join(father_dir, mark)
    if not os.path.exists(dir_temp):
        os.makedirs(dir_temp)
    return dir_temp

#########################
# pruning Settings
#########################
def apply_pruning(model, amount=0.2):
    """
    Apply global unstructured pruning to all Conv2d and ConvTranspose2d layers in the model.
    
    Args:
        model (nn.Module): The model to prune.
        amount (float): The proportion of connections to prune (e.g., 0.2 for 20%).
    """
    parameters_to_prune = []
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
            parameters_to_prune.append((module, 'weight'))
    
    if parameters_to_prune:
        nn.utils.prune.global_unstructured(
            parameters_to_prune,
            pruning_method=nn.utils.prune.L1Unstructured,
            amount=amount,
        )
        print(f"Applied global unstructured pruning: {amount*100}% of connections pruned.")
    else:
        print("No Conv2d or ConvTranspose2d layers found for pruning.")

#########################
# random seeds Setting
#########################
def random_seed_set(random_seed):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        
#########################
# crop image
#########################
from typing import Tuple, List

def crop_npy_files(
    input_dir: str,
    output_dir: str,
    patch_size: Tuple[int, int] = (256, 256),
    step_size: Tuple[int, int] = (200, 200)
):
    """
    裁剪目录中的所有 .npy 文件，并保存裁剪后的文件到输出目录。

    Args:
        input_dir (str): 原始 .npy 文件所在的目录。
        output_dir (str): 裁剪后文件保存的目录。
        patch_size (Tuple[int, int], optional): 裁剪块的大小。默认为 (256, 256)。
        step_size (Tuple[int, int], optional): 步长。默认为 (200, 200)。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.npy'):
            filepath = os.path.join(input_dir, filename)
            data = np.load(filepath)
            height, width = data.shape

            # 计算裁剪起始位置
            patches = []
            patch_rows = []
            patch_cols = []

            for i in range(0, height, step_size[0]):
                if i + patch_size[0] > height:
                    i = height - patch_size[0]
                for j in range(0, width, step_size[1]):
                    if j + patch_size[1] > width:
                        j = width - patch_size[1]
                    patch = data[i:i + patch_size[0], j:j + patch_size[1]]
                    patches.append(patch)
                    patch_rows.append(i)
                    patch_cols.append(j)
            
            # 去重裁剪起始位置
            unique_patches = {}
            for idx, (i, j) in enumerate(zip(patch_rows, patch_cols)):
                key = (i, j)
                if key not in unique_patches:
                    unique_patches[key] = patches[idx]

            # 保存裁剪后的文件
            for (i, j), patch in unique_patches.items():
                base_name = os.path.splitext(filename)[0]
                new_filename = f"{base_name}_patch_r{i}_c{j}.npy"
                save_path = os.path.join(output_dir, new_filename)
                np.save(save_path, patch)

    print(f"裁剪完成，裁剪后的文件保存在 {output_dir}")

# 示例调用
# crop_npy_files('path/to/original_npy', 'path/to/cropped_npy')


#########################
# stitching npy data
#########################
def stitching_npy(
    cropped_dir: str,
    output_dir: str,
    original_shape: Tuple[int, int] = (1440, 2040),
    patch_size: Tuple[int, int] = (256, 256),
    step_size: Tuple[int, int] = (200, 200)
):
    """
    根据裁剪后的文件重建原始的 .npy 矩阵。

    Args:
        cropped_dir (str): 裁剪后的 .npy 文件所在的目录。
        output_dir (str): 重建后文件保存的目录。
        original_shape (Tuple[int, int], optional): 原始矩阵的形状。默认为 (1440, 2040)。
        patch_size (Tuple[int, int], optional): 裁剪块的大小。默认为 (256, 256)。
        step_size (Tuple[int, int], optional): 步长。默认为 (200, 200)。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # supose all files are named as '20020801_20020831_r{i}_c{j}.npy', and can be adjust to your own file name pattern
    files = [f for f in os.listdir(cropped_dir) if f.endswith('.npy')]
    # select all files with the same time, suppose all files are named as '20020801_20020831_r{i}_c{j}.npy'
    time_set = set()
    for f in files:
        parts = f.split('.')
        time_info = parts[1]  # example: '20020801_20020831'
        time_set.add(time_info)

    for time in time_set:
        # select files with the same time
        time_files = [f for f in files if f.split('.')[1] == time]
        stitched = np.zeros(original_shape)
        weight = np.zeros(original_shape)

        for f in time_files:
            parts = f.split('_')
            r_part = parts[-2]      # 'r{i}'
            c_part = parts[-1]      # 'c{j}.npy'
            i = int(r_part[1:])
            j = int(c_part[1:-4])   # remove 'c' and '.npy'

            patch = np.load(os.path.join(cropped_dir, f))
            stitched[i:i + patch_size[0], j:j + patch_size[1]] += patch
            weight[i:i + patch_size[0], j:j + patch_size[1]] += 1

        # avoid divide by zero
        stitched /= np.maximum(weight, 1)

        # 保存重建后的文件
        new_filename = f"reconstructed_{time}.npy"
        save_path = os.path.join(output_dir, new_filename)
        np.save(save_path, stitched)

    print(f"重建完成，重建后的文件保存在 {output_dir}")

#########################
# SSIM & MSSIM Functions
#########################

from math import exp

import torch
import torch.nn.functional as F
from torch import nn


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/(2*sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1:torch.tensor, img2:torch.tensor, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Compute SSIM
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1
        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    if img1.ndimension() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    elif img1.ndimension() == 5:
        (_, _, channel, height, width) = img1.size()
        img1 = img1.reshape(-1, channel, height, width)
        img2 = img2.reshape(-1, channel, height, width)
    (_, channel, height, width) = img1.size()


    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=0, groups=channel)
    mu2 = F.conv2d(img2, window, padding=0, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=0, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=0, groups=channel) - mu2_sq
    sigma12   = F.conv2d(img1*img2, window, padding=0, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = v1 / v2  # contrast sensitivity

    ssim_map = ((2*mu1_mu2 + C1)*v1)/((mu1_sq + mu2_sq + C1)*v2)

    if size_average:
        ret = ssim_map.mean()
        cs = cs.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)
        cs = cs.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret

def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=None):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    ssims = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
        if normalize == "relu":
            ssims.append(torch.relu(sim))
            mcs.append(torch.relu(cs))
        else:
            ssims.append(sim)
            mcs.append(cs)
        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    ssims = torch.stack(ssims)
    mcs = torch.stack(mcs)

    if normalize == "simple" or normalize == True:
        ssims = (ssims + 1)/2
        mcs = (mcs + 1)/2

    pow1 = mcs ** weights
    pow2 = ssims ** weights
    output = torch.prod(pow1[:-1]) * pow2[-1]
    return output

class SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        channel = img1.size()[1]
        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window.to(img1.device)
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel
        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)



if __name__ == "__main__":
    
    crop_npy_files('/root/autodl-tmp/mask', '/root/autodl-tmp/mask_256')
    # stitching_npy('path/to/cropped_npy', 'path/to/reconstructed_npy')

