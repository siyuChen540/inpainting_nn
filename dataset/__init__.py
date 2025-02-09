"""
@author: Siyu Chen
@date: 2025.1.28
@file:dataset/__init__.py
@description:
    This file implements PyTorch datasets for loading and processing sequential data.
@classes:
    - InpaintingDataset: Dataset for loading sequential frames with optional masking.
    - DataProcessor: Helper class for organizing data files by spatial indices and time.
    - InpaintingDatasetV2: Dataset for loading cropped sequential frames with matching different patch.
    - InpaintingDatasetV3: Using multi-threading to load data.
    - InpaintingDatasetV4: Load all frames in a single tensor.
"""


import os
import re
import json
import time
from typing import Tuple, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import ToTensor, Lambda, Compose


def log_transform(x: Tensor) -> Tensor:
    """Apply log10 transform with clamping to avoid negative infinity."""
    return torch.log10(torch.clamp(x, min=1e-8))


class InpaintingDataset(Dataset):
    """
    PyTorch Dataset for loading sequential frames of data (e.g., chlorophyll images)
    with optional monthly masking and transformation.

    Args:
        data_root_dir (str): Directory containing .npy data files.
        num_frames (int): Number of frames (temporal dimension) per sample.
        resize_shape (Tuple[int, int]): Target spatial size (height, width) for resizing.
        enable_monthly_mask (bool): If True, use monthly masks; otherwise random mask.
        monthly_mask_dir (str): Directory containing the mask .npy files (if needed).
        is_training_set (bool): True if dataset is for training, False otherwise.
        train_split_ratio (float): Ratio to split data into training and testing sets.
        task_name (str): Optional task identifier.
        file_chunk_list (List[List[str]]): Pre-computed data file groups.
        mask_chunk_list (List[List[str]]): Pre-computed mask file groups.
        global_mean (float): Global mean value for normalization.
        global_std (float): Global standard deviation for normalization.
        apply_log (bool): Whether to apply log10 transform.
        apply_normalize (bool): Whether to apply normalization.
    """
    def __init__(self,
                 data_root_dir: str,
                 num_frames: int,
                 resize_shape: Tuple[int, int] = (48, 48),
                 enable_monthly_mask: bool = False,
                 monthly_mask_dir: str = None,
                 is_training_set: bool = True,
                 train_split_ratio: float = 0.7,
                 task_name: str = None,
                 file_chunk_list: List[List[str]] = None,
                 mask_chunk_list: List[List[str]] = None,
                 global_mean: float = None,
                 global_std: float = None,
                 apply_log: bool = True,
                 apply_normalize: bool = True) -> None:
        super().__init__()
        self.data_root_dir = data_root_dir
        self.num_frames = num_frames
        self.resize_shape = resize_shape
        self.enable_monthly_mask = enable_monthly_mask
        self.is_training_set = is_training_set
        self.train_split_ratio = train_split_ratio
        self.task_name = task_name

        if self.enable_monthly_mask:
            if monthly_mask_dir is None:
                raise ValueError("monthly_mask_dir must be specified if enable_monthly_mask is True")
            self.monthly_mask_dir = monthly_mask_dir

        # Generate file_chunk_list if not provided.
        if file_chunk_list is None:
            full_data_list: List[str] = self._list_files(self.data_root_dir, suffix='.npy')
            file_chunk_list = self._create_chunks(full_data_list, self.num_frames)
            file_chunk_list = self._train_test_split(file_chunk_list)
        self.file_chunk_list = file_chunk_list

        if self.enable_monthly_mask:
            if mask_chunk_list is None:
                full_mask_list: List[str] = self._list_files(self.monthly_mask_dir, suffix='.npy')
                mask_chunk_list = self._create_chunks(full_mask_list, self.num_frames)
                mask_chunk_list = self._train_test_split(mask_chunk_list)
            self.mask_chunk_list = mask_chunk_list
        else:
            self.mask_chunk_list = []

        self.length: int = len(self.file_chunk_list)
        self.global_mean = global_mean
        self.global_std = global_std
        self.apply_log = apply_log
        self.apply_normalize = apply_normalize

        # Build transformation pipeline.
        transform_list = [ToTensor()]
        if self.apply_log:
            transform_list.append(Lambda(log_transform))
        transform_list.append(transforms.Resize(self.resize_shape))
        if self.apply_normalize and (self.global_mean is not None and self.global_std is not None):
            transform_list.append(transforms.Normalize(mean=[self.global_mean], std=[self.global_std]))
        self.transform = Compose(transform_list)

    def _train_test_split(self, chunk_list: List[List[str]]) -> List[List[str]]:
        np.random.shuffle(chunk_list)
        split_idx = int(self.train_split_ratio * len(chunk_list))
        return chunk_list[:split_idx] if self.is_training_set else chunk_list[split_idx:]

    @staticmethod
    def _list_files(base_dir: str, suffix: str = '.npy') -> List[str]:
        files_list = [
            os.path.join(base_dir, f) 
            for f in os.listdir(base_dir) 
            if f.endswith(suffix)
            ]
        files_list.sort()
        return files_list

    @staticmethod
    def _create_chunks(file_list: List[str], frames: int) -> List[List[str]]:
        return [
            file_list[i:i + frames] 
            for i in range(len(file_list) - frames + 1)
            ]

    def __len__(self) -> int:
        return self.length

    @staticmethod
    def _load_npy(file_path: str) -> np.ndarray:
        return np.load(file_path)

    @staticmethod
    def _apply_mask(data: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        if mask.shape != data.shape:
            raise ValueError(f"Mask shape {mask.shape} does not match data shape {data.shape}")
        return data * mask, mask

    @staticmethod
    def _apply_random_mask(data: Tensor) -> Tuple[Tensor, Tensor]:
        shape = data.shape
        rand_mask = torch.from_numpy(np.where(np.random.normal(loc=100, scale=10, size=shape) < 90, 0, 1)).float()
        return data * rand_mask, rand_mask

    def __getitem__(self, index: int) -> Tuple[int, Tensor, Tensor]:
        chunk = self.file_chunk_list[index]
        mask_paths: List[Any]
        if self.enable_monthly_mask:
            mask_idx = min(index, len(self.mask_chunk_list) - 1)
            mask_paths = self.mask_chunk_list[mask_idx]
        else:
            mask_paths = [None] * len(chunk)

        inputs_list: List[Tensor] = []
        target_list: List[Tensor] = []

        for data_path, mask_path in zip(chunk, mask_paths):
            array = self._load_npy(data_path)
            array[np.isnan(array)] = 1.0
            array[array == 0.0] = 1.0
            data_tensor = self.transform(array)

            if self.enable_monthly_mask and mask_path is not None:
                mask_array = np.load(mask_path)
                mask_array[np.isnan(mask_array)] = 0.0
                # Resize mask tensor to match data_tensor shape.
                mask_tensor = torch.resize_as_(torch.from_numpy(mask_array).unsqueeze(0).float(), data_tensor)
                in_data, _ = self._apply_mask(data_tensor, mask_tensor)
            else:
                in_data, _ = self._apply_random_mask(data_tensor)
            if torch.any(torch.isnan(in_data)):
                raise ValueError("NaNs detected in the processed data")
            inputs_list.append(in_data)
            target_list.append(data_tensor)
        inputs = torch.stack(inputs_list, dim=0)
        targets = torch.stack(target_list, dim=0)
        return index, inputs.float(), targets.float()


class DataProcessor:
    """
    Processes and organizes data files by spatial indices and temporal information.

    This class uses regular expressions for robust extraction of indices and time info,
    making it adaptable to variations in file naming.

    Args:
        root_dir (str): Directory containing data files.
        json_file_path (str): Optional path for a JSON file to cache the processed dictionary.
    """
    def __init__(self, root_dir: str, json_file_path: str = None) -> None:
        self.root_dir = root_dir
        self.json_file_path = json_file_path

    def _key_to_str(self, key: Tuple[int, int]) -> str:
        """Convert tuple key to a comma-separated string."""
        return f"{key[0]},{key[1]}"

    def _parse_filename(self, filename: str) -> Tuple[str, Tuple[int, int]]:
        """
        Extract time information and spatial indices (row, col) from filename using regex.

        Returns:
            Tuple[str, Tuple[int, int]]: Extracted time string and spatial indices.
        """
        # Regex to extract time info: 8-digit or 8-digit_8-digit pattern.
        time_match = re.search(r'(\d{8}(?:_\d{8})?)', filename)
        time_info = time_match.group(1) if time_match else "unknown_time"

        # Regex to extract spatial indices: look for pattern r{row}_c{col}
        indices_match = re.search(r"r(?P<row>\d+)[^0-9a-zA-Z]*c(?P<col>\d+)", filename, re.IGNORECASE)
        if not indices_match:
            raise ValueError(f"Filename {filename} does not contain valid spatial indices.")
        row = int(indices_match.group("row"))
        col = int(indices_match.group("col"))
        return time_info, (row, col)

    def process_data(self) -> Dict[Tuple[int, int], List[Tuple[str, str]]]:
        """
        Process data files and organize them by spatial indices and time.

        Returns:
            Dict[Tuple[int, int], List[Tuple[str, str]]]: Dictionary with keys as (row, col)
            and values as lists of (time_info, filename) tuples.
        """
        files = [f for f in os.listdir(self.root_dir) if f.endswith('.npy')]
        data_dict: Dict[Tuple[int, int], List[Tuple[str, str]]] = {}

        if self.json_file_path and os.path.exists(self.json_file_path):
            with open(self.json_file_path, 'r') as f:
                loaded_dict = json.load(f)
            data_dict = {tuple(map(int, k.split(','))): v for k, v in loaded_dict.items()}
            return data_dict

        for f in files:
            try:
                time_info, key = self._parse_filename(f)
            except ValueError:
                continue
            if key not in data_dict:
                data_dict[key] = []
            data_dict[key].append((time_info, f))

        for key in data_dict:
            data_dict[key].sort(key=lambda x: x[0])
        if self.json_file_path and not os.path.exists(self.json_file_path):
            serializable_dict = {self._key_to_str(k): v for k, v in data_dict.items()}
            with open(self.json_file_path, 'w') as f:
                json.dump(serializable_dict, f)
        return data_dict


class InpaintingDatasetV2(Dataset):
    """
    PyTorch Dataset for loading cropped sequential frames from the same spatial
    location. It uses a robust file organization and applies transformations such
    as resizing or masking.

    Args:
        data_root_dir (str): Directory containing cropped .npy files.
        num_frames (int): Number of frames per sample.
        json_file_path (str): Optional path for caching organized file info.
        resize_shape (Tuple[int, int]): Target resize dimensions (height, width).
        enable_mask (bool): Whether or not to use mask files.
        mask_root_dir (str): Directory of mask files if enable_mask is True.
        is_training_set (bool): True if dataset is for training, False otherwise.
        train_split_ratio (float): Ratio to split training and testing data.
        global_mean (float): Global mean for normalization.
        global_std (float): Global std for normalization.
        apply_log (bool): Whether to apply log10 transformation.
        apply_normalize (bool): Whether to apply normalization.
    """
    def __init__(self,
                 data_root_dir: str,
                 num_frames: int,
                 json_file_path: str = None,
                 resize_shape: Tuple[int, int] | bool = (256, 256),
                 enable_mask: bool = False,
                 mask_root_dir: str = None,
                 is_training_set: bool = True,
                 train_split_ratio: float = 0.7,
                 global_mean: float = None,
                 global_std: float = None,
                 apply_log: bool = True,
                 apply_normalize: bool = True) -> None:
        super().__init__()
        self.data_root_dir = data_root_dir
        self.num_frames = num_frames
        self.json_file_path = json_file_path
        self.resize_shape = resize_shape
        self.enable_mask = enable_mask
        self.is_training_set = is_training_set
        self.train_split_ratio = train_split_ratio
        self.global_mean = global_mean
        self.global_std = global_std
        self.apply_log = apply_log
        self.apply_normalize = apply_normalize

        if self.enable_mask:
            if mask_root_dir is None:
                raise ValueError("mask_root_dir must be specified if enable_mask is True")
            self.mask_root_dir = mask_root_dir

        # Organize files by spatial and temporal info.
        self.data_dict: Dict[Tuple[int, int], List[Tuple[str, str]]] = self._organize_files(self.data_root_dir, self.json_file_path)
        self.sorted_keys = sorted(self.data_dict.keys())
        self.chunk_list = self._create_chunks(self.data_root_dir, self.data_dict)
        self.length: int = len(self.chunk_list)

        if self.enable_mask:
            self.mask_dict = self._organize_files(self.mask_root_dir)
            self.mask_chunk_list = self._create_chunks(self.mask_root_dir, self.mask_dict)
        else:
            self.mask_dict = {}

        # Build transformation pipeline.
        transform_ops = [ToTensor()]
        if self.apply_log:
            transform_ops.append(Lambda(log_transform))
        if self.resize_shape:
            transform_ops.append(transforms.Resize(self.resize_shape))
        if self.apply_normalize and (self.global_mean is not None and self.global_std is not None):
            transform_ops.append(transforms.Normalize(mean=[self.global_mean], std=[self.global_std]))
        self.transform = Compose(transform_ops)

    def _organize_files(self, root: str, json_file_path: str = None) -> Dict[Tuple[int, int], List[Tuple[str, str]]]:
        dp = DataProcessor(root, json_file_path)
        return dp.process_data()

    def _create_chunks(self, root: str, data_dict: Dict[Tuple[int, int], List[Tuple[str, str]]]) -> List[List[str]]:
        chunks: List[List[str]] = []
        for key, files in data_dict.items():
            if len(files) < self.num_frames:
                continue
            for i in range(len(files) - self.num_frames + 1):
                chunk = [os.path.join(root, files[i + j][1]) for j in range(self.num_frames)]
                chunks.append(chunk)
        np.random.shuffle(chunks)
        split_idx = int(self.train_split_ratio * len(chunks))
        return chunks[:split_idx] if self.is_training_set else chunks[split_idx:]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Tuple[int, Tensor, Tensor]:
        chunk = self.chunk_list[index]
        inputs_list: List[Tensor] = []
        target_list: List[Tensor] = []
        mask_paths: List[Any] = (self.mask_chunk_list[min(index, len(self.mask_chunk_list)-1)]
                                   if self.enable_mask else [None] * len(chunk))
        for data_path, mask_path in zip(chunk, mask_paths):
            array = np.load(data_path)
            array = np.where((array == 0.0) | np.isnan(array), 1.0, array)
            data_tensor = self.transform(array)
            if self.enable_mask and mask_path is not None:
                mask_array = np.load(mask_path)
                mask_array = np.where(np.isnan(mask_array), 0.0, 1.0)
                mask_tensor = torch.from_numpy(mask_array).unsqueeze(0).float()
                if mask_tensor.shape != data_tensor.shape:
                    mask_tensor = F.interpolate(
                        mask_tensor.unsqueeze(0),  
                        size=data_tensor.shape[-2:],
                        mode='nearest'
                    ).squeeze(0)
                in_data, _ = self._apply_mask(data_tensor, mask_tensor)
            else:
                in_data, _ = self._apply_random_mask(data_tensor)
            inputs_list.append(in_data)
            target_list.append(data_tensor)
        inputs = torch.stack(inputs_list, dim=0)
        targets = torch.stack(target_list, dim=0)
        return index, inputs.float(), targets.float()

    @staticmethod
    def _apply_mask(data: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        if mask.shape != data.shape:
            raise ValueError(f"Mask shape {mask.shape} does not match data shape {data.shape}")
        mask[torch.isnan(mask)] = 0.0
        return data * mask, mask

    @staticmethod
    def _apply_random_mask(data: Tensor, p: int = 90) -> Tuple[Tensor, Tensor]:
        mask = (torch.rand_like(data) < p).float()
        return data * mask, mask

class InpaintingDatasetV3(InpaintingDatasetV2):
    def __init__(self, 
                 data_root_dir, 
                 num_frames, 
                 json_file_path = None, 
                 resize_shape = (256, 256), 
                 enable_mask = False, 
                 mask_root_dir = None, 
                 is_training_set = True, 
                 train_split_ratio = 0.7, 
                 global_mean = None, 
                 global_std = None, 
                 apply_log = True, 
                 apply_normalize = True):
        super().__init__(data_root_dir, num_frames, json_file_path, resize_shape, enable_mask, mask_root_dir, is_training_set, train_split_ratio, global_mean, global_std, apply_log, apply_normalize)

    def __getitem__(self, index: int) -> Tuple[int, Tensor, Tensor]:
        chunk = self.chunk_list[index]
        mask_paths: List[Any] = (self.mask_chunk_list[min(index, len(self.mask_chunk_list)-1)]
                                if self.enable_mask else [None] * len(chunk))

        # 使用多线程并行加载数据
        with ThreadPoolExecutor() as executor:
            data_futures = [executor.submit(self._load_and_preprocess, data_path, mask_path) for data_path, mask_path in zip(chunk, mask_paths)]
            results = [future.result() for future in data_futures]

        inputs_list, target_list = zip(*results)
        inputs = torch.stack(inputs_list, dim=0)
        targets = torch.stack(target_list, dim=0)
        return index, inputs.float(), targets.float()

    def _load_and_preprocess(self, data_path: str, mask_path: str) -> Tuple[Tensor, Tensor]:
        array = np.load(data_path)
        array = np.where((array == 0.0) | np.isnan(array), 1.0, array)
        data_tensor = self.transform(array)
        if self.enable_mask and mask_path is not None:
            mask_array = np.load(mask_path)
            mask_array = np.where(np.isnan(mask_array), 0.0, 1.0)
            mask_tensor = torch.from_numpy(mask_array).unsqueeze(0).float()
            if mask_tensor.shape != data_tensor.shape:
                mask_tensor = F.interpolate(
                    mask_tensor.unsqueeze(0),  
                    size=data_tensor.shape[-2:],
                    mode='nearest'
                ).squeeze(0)
            in_data, _ = self._apply_mask(data_tensor, mask_tensor)
        else:
            in_data, _ = self._apply_random_mask(data_tensor)
        return in_data, data_tensor

class InpaintingDatasetV4(InpaintingDatasetV2):
    def __init__(self, data_root_dir, num_frames, json_file_path = None, resize_shape = (256, 256), enable_mask = False, mask_root_dir = None, is_training_set = True, train_split_ratio = 0.7, global_mean = None, global_std = None, apply_log = True, apply_normalize = True):
        resize_shape = False
        super().__init__(data_root_dir, num_frames, json_file_path, resize_shape, enable_mask, mask_root_dir, is_training_set, train_split_ratio, global_mean, global_std, apply_log, apply_normalize)
    
    def __getitem__(self, index):
        chunk = self.chunk_list[index]
        arrays: List[np.ndarray] = [np.load(data_path) for data_path in chunk]
        array_3d = np.stack(arrays, axis=0)
        array_3d = np.where((array_3d == 0.0) | np.isnan(array_3d), 1.0, array_3d)
        array_3d = np.transpose(array_3d, (1, 2, 0))
        data_tensor:Tensor = self.transform(array_3d)
        data_tensor = data_tensor.unsqueeze(1)
        mask_paths: List[Any] = (self.mask_chunk_list[min(index, len(self.mask_chunk_list)-1)]
                                if self.enable_mask else [None] * len(chunk))
        if self.enable_mask:
            mask_arrays = [np.load(mask_path) for mask_path in mask_paths]
            mask_3d = np.stack(mask_arrays, axis=0)
            mask_3d = np.where(np.isnan(mask_3d), 0.0, 1.0)
            mask_tensor = torch.from_numpy(mask_3d).unsqueeze(1).float()
            if mask_tensor.shape != data_tensor.shape:
                raise ValueError(f"Mask shape {mask_tensor.shape} does not match data shape {data_tensor.shape}")
            in_data, _ = self._apply_mask(data_tensor, mask_tensor)
        else:
            in_data, _ = self._apply_random_mask(data_tensor)
        
        return index, in_data.float(), data_tensor.float()

import time

class InpaintingDatasetV5(InpaintingDatasetV2):
    def __init__(self, data_root_dir, num_frames, json_file_path=None, resize_shape=(256, 256), enable_mask=False, mask_root_dir=None, is_training_set=True, train_split_ratio=0.7, global_mean=None, global_std=None, apply_log=True, apply_normalize=True):
        resize_shape = False
        super().__init__(data_root_dir, num_frames, json_file_path, resize_shape, enable_mask, mask_root_dir, is_training_set, train_split_ratio, global_mean, global_std, apply_log, apply_normalize)

    def __getitem__(self, index):
        start_time = time.time()
        
        chunk = self.chunk_list[index]
        arrays: List[np.ndarray] = [np.load(data_path) for data_path in chunk]
        print(f"Data paths: {chunk}")
        load_time = time.time()
        array_sizes:list[int] = [os.path.getsize(data_path) for data_path in chunk]
        print(f"Data file sizes: {array_sizes}")
        print(f"Number of data files: {len(chunk)}")
        
        
        array_3d = np.stack(arrays, axis=0)
        stack_time = time.time()
        
        array_3d = np.where((array_3d == 0.0) | np.isnan(array_3d), 1.0, array_3d)
        where_time = time.time()
        
        array_3d = np.transpose(array_3d, (1, 2, 0))
        transpose_time = time.time()
        
        data_tensor: Tensor = self.transform(array_3d)
        transform_time = time.time()
        
        data_tensor = data_tensor.unsqueeze(1)
        unsqueeze_time = time.time()
        
        mask_paths: List[Any] = (self.mask_chunk_list[min(index, len(self.mask_chunk_list)-1)]
                                 if self.enable_mask else [None] * len(chunk))
        mask_paths_time = time.time()
        
        if self.enable_mask:
            # 打印掩码文件大小和数量
            mask_sizes = [os.path.getsize(mask_path) for mask_path in mask_paths]
            print(f"Mask file sizes: {mask_sizes}")
            print(f"Number of mask files: {len(mask_paths)}")
            
            mask_check_time = time.time()
            # # 使用多线程并行加载掩码文件
            with ThreadPoolExecutor() as executor:
                mask_arrays = list(executor.map(np.load, chunk))
            mask_load_time = time.time()
            
            mask_3d = np.stack(mask_arrays, axis=0)
            mask_stack_time = time.time()
            
            mask_3d = np.where(np.isnan(mask_3d), 0.0, 1.0)
            mask_where_time = time.time()
            
            mask_tensor = torch.from_numpy(mask_3d).unsqueeze(1).float()
            mask_tensor_time = time.time()
            
            if mask_tensor.shape != data_tensor.shape:
                raise ValueError(f"Mask shape {mask_tensor.shape} does not match data shape {data_tensor.shape}")
            
            in_data, _ = self._apply_mask(data_tensor, mask_tensor)
            apply_mask_time = time.time()
        else:
            in_data, _ = self._apply_random_mask(data_tensor)
            apply_random_mask_time = time.time()
        
        end_time = time.time()
        
        total_time = end_time - start_time
        print(f"Load time: {load_time - start_time:.4f}s ({(load_time - start_time) / total_time * 100:.2f}%)")
        print(f"Stack time: {stack_time - load_time:.4f}s ({(stack_time - load_time) / total_time * 100:.2f}%)")
        print(f"Where time: {where_time - stack_time:.4f}s ({(where_time - stack_time) / total_time * 100:.2f}%)")
        print(f"Transpose time: {transpose_time - where_time:.4f}s ({(transpose_time - where_time) / total_time * 100:.2f}%)")
        print(f"Transform time: {transform_time - transpose_time:.4f}s ({(transform_time - transpose_time) / total_time * 100:.2f}%)")
        print(f"Unsqueeze time: {unsqueeze_time - transform_time:.4f}s ({(unsqueeze_time - transform_time) / total_time * 100:.2f}%)")
        print(f"Mask paths time: {mask_paths_time - unsqueeze_time:.4f}s ({(mask_paths_time - unsqueeze_time) / total_time * 100:.2f}%)")
        
        if self.enable_mask:
            print(f"Mask check time: {mask_check_time - mask_paths_time:.4f}s ({(mask_check_time - mask_paths_time) / total_time * 100:.2f}%)")
            print(f"Mask load time: {mask_load_time - mask_check_time:.4f}s ({(mask_load_time - mask_check_time) / total_time * 100:.2f}%)")
            print(f"Mask stack time: {mask_stack_time - mask_load_time:.4f}s ({(mask_stack_time - mask_load_time) / total_time * 100:.2f}%)")
            print(f"Mask where time: {mask_where_time - mask_stack_time:.4f}s ({(mask_where_time - mask_stack_time) / total_time * 100:.2f}%)")
            print(f"Mask tensor time: {mask_tensor_time - mask_where_time:.4f}s ({(mask_tensor_time - mask_where_time) / total_time * 100:.2f}%)")
            print(f"Apply mask time: {apply_mask_time - mask_tensor_time:.4f}s ({(apply_mask_time - mask_tensor_time) / total_time * 100:.2f}%)")
        else:
            print(f"Apply random mask time: {apply_random_mask_time - mask_paths_time:.4f}s ({(apply_random_mask_time - mask_paths_time) / total_time * 100:.2f}%)")
        
        print(f"Total time: {total_time:.4f}s")
        
        return index, in_data.float(), data_tensor.float()

# Detailed test functions to ensure dataset stability and error handling.
def test_dataset(ds_cls:Dataset) -> None:
    """
    Test function for InpaintingDatasetV2.
    Validates loading, transformation and error handling.
    """
    test_root = 'E:/04_DevelopReleas/02_test_MArineSIR/dataset/train/input_256/'
    test_mask_dir = 'E:/04_DevelopReleas/02_test_MArineSIR/dataset/train/mask_256/'
    try:
        dataset = ds_cls(test_root, num_frames=10, json_file_path=None,
                        enable_mask=True, mask_root_dir=test_mask_dir, 
                        is_training_set=True, train_split_ratio=0.7,
                        apply_log=True, apply_normalize=False, resize_shape=False
                        )
    except Exception as e:
        print(f"Error initializing dataset: {e}")
        return
    print(f"Dataset length: {len(dataset)}")
    start = time.time()
    for i in range(len(dataset)):
        try:
            # Load and process sample. 
            idx, inputs, targets = dataset[i]
            assert inputs.shape == targets.shape, "Input and target shapes mismatch"
            # print(f"Sample {idx}: inputs shape {inputs.shape}, targets shape {targets.shape}")
        except Exception as err:
            print(f"Error processing sample {i}: {err}")
    end = time.time()
    mean_time = (end - start) / len(dataset)
    print(f"Mean time per sample: {mean_time} seconds")

def compare_3_datasets():
    test_root = '/input_256/'
    test_mask_dir = '/mask_256/'
    dataset_v2 = InpaintingDatasetV2(test_root, num_frames=10, json_file_path=None,
                        enable_mask=True, mask_root_dir=test_mask_dir, 
                        is_training_set=True, train_split_ratio=0.7,
                        apply_log=True, apply_normalize=False, resize_shape=False
                        )
    dataset_v3 = InpaintingDatasetV3(test_root, num_frames=10, json_file_path=None,
                        enable_mask=True, mask_root_dir=test_mask_dir, 
                        is_training_set=True, train_split_ratio=0.7,
                        apply_log=True, apply_normalize=False
                        )
    dataset_v4 = InpaintingDatasetV4(test_root, num_frames=10, json_file_path=None,
                        enable_mask=True, mask_root_dir=test_mask_dir, 
                        is_training_set=True, train_split_ratio=0.7,
                        apply_log=True, apply_normalize=False
                        )
    for i in range(len(dataset_v2)):
        idx_v2, inputs_v2, targets_v2 = dataset_v2[i]
        idx_v3, inputs_v3, targets_v3 = dataset_v3[i]
        idx_v4, inputs_v4, targets_v4 = dataset_v4[i]
        assert torch.allclose(inputs_v2, inputs_v3),   f"DatasetV2 and DatasetV3 inputs  mismatch at index {i}"
        assert torch.allclose(targets_v2, targets_v3), f"DatasetV2 and DatasetV3 targets mismatch at index {i}"
        assert torch.allclose(inputs_v2, inputs_v4),   f"DatasetV2 and DatasetV4 inputs  mismatch at index {i}"
        assert torch.allclose(targets_v2, targets_v4), f"DatasetV2 and DatasetV4 targets mismatch at index {i}"
    print("All datasets are consistent.")

if __name__ == '__main__':
    # print("Testing InpaintingDatasetV2...")
    # test_dataset(InpaintingDatasetV2)
    # print("------------------------------")
    # print("Testing InpaintingDatasetV3...")
    # test_dataset(InpaintingDatasetV3)
    # print("------------------------------")
    # print("Testing InpaintingDatasetV4...")
    # test_dataset(InpaintingDatasetV4)
    # print("------------------------------")
    # print("Comparing datasets...")
    # compare_3_datasets()
    # print("All tests passed successfully.")
    print("------------------------------")
    print("Testing InpaintingDatasetV5...")
    test_dataset(InpaintingDatasetV5)
    print("------------------------------")