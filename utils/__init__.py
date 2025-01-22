        
from typing import Tuple

import os
import numpy as np
import torch
from torch import nn

from .early_stopping import EarlyStopping
from .image_processing import crop_npy_files, stitching_npy
from .metrics import SSIM, ssim, msssim
from .misc import record_dir_setting_create, apply_pruning, random_seed_set

__all__ = [
    'EarlyStopping',
    'crop_npy_files',
    'stitching_npy',
    'SSIM',
    'ssim',
    'msssim',
    'record_dir_setting_create',
    'apply_pruning',
    'random_seed_set',
    'crop_npy_files',
    'random_seed_set',
    'apply_pruning',
    'stitching_npy'
]

__version__ = '0.2.1'

__author__ = 'SiyuChen'

__license__ = 'MIT'

__description__ = 'A collection of utility functions for the project.'

__url__ = 'https://github.com/SiyuChen/inpainting_nn/utils'

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

def random_seed_set(random_seed):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)

def crop_npy_files(
    input_dir: str,
    output_dir: str,
    patch_size: Tuple[int, int] = (256, 256),
    step_size: Tuple[int, int] = (200, 200)
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.npy'):
            filepath = os.path.join(input_dir, filename)
            data:np.ndarray = np.load(filepath)
            height, width = data.shape

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
            
            unique_patches = {}
            for idx, (i, j) in enumerate(zip(patch_rows, patch_cols)):
                key = (i, j)
                if key not in unique_patches:
                    unique_patches[key] = patches[idx]

            for (i, j), patch in unique_patches.items():
                base_name = os.path.splitext(filename)[0]
                new_filename = f"{base_name}_patch_r{i}_c{j}.npy"
                save_path = os.path.join(output_dir, new_filename)
                np.save(save_path, patch)

def stitching_npy(
    cropped_dir: str,
    output_dir: str,
    original_shape: Tuple[int, int] = (1440, 2040),
    patch_size: Tuple[int, int] = (256, 256),
    step_size: Tuple[int, int] = (200, 200)
):    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    
    files = [f for f in os.listdir(cropped_dir) if f.endswith('.npy')]
    
    time_set = set()
    for f in files:
        parts = f.split('.')
        time_info = parts[1]  
        time_set.add(time_info)

    for time in time_set:        
        time_files = [f for f in files if f.split('.')[1] == time]
        stitched = np.zeros(original_shape)
        weight = np.zeros(original_shape)

        for f in time_files:
            parts = f.split('_')
            r_part = parts[-2] 
            c_part = parts[-1] 
            i = int(r_part[1:])
            j = int(c_part[1:-4])  

            patch = np.load(os.path.join(cropped_dir, f))
            stitched[i:i + patch_size[0], j:j + patch_size[1]] += patch
            weight[i:i + patch_size[0], j:j + patch_size[1]] += 1

        stitched /= np.maximum(weight, 1)

        new_filename = f"reconstructed_{time}.npy"
        save_path = os.path.join(output_dir, new_filename)
        np.save(save_path, stitched)


