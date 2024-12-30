"""
    @Author: SiyuChen
    @Date: 2024-12-30
    @Description: Miscellaneous functions for the project.
    @License: MIT
    @File: misc.py
"""


import os
import torch
import numpy as np
from torch import nn


def record_dir_setting_create(father_dir: str, mark: str) -> str:
    """
    Create a directory with a specific mark inside the given father directory.
    
    Args:
        father_dir (str): The parent directory where the new directory will be created.
        mark (str): The name of the new directory to be created.
    
    Returns:
        str: The path to the newly created directory.
    
    This function checks if a directory with the specified mark exists within the father directory.
    If it does not exist, it creates the directory. The function then returns the path to the new directory.
    """
    dir_temp = os.path.join(father_dir, mark)
    if not os.path.exists(dir_temp):
        os.makedirs(dir_temp)
    return dir_temp


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
    """
    Set the random seed for reproducibility.
    
    Args:
        random_seed (int): The seed value to set for numpy and torch.
    
    This function sets the random seed for numpy and torch to ensure reproducibility of experiments.
    If a GPU is available, it also sets the random seed for CUDA.
    """
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)