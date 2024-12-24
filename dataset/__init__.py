#########################
# Dataset Class
#########################
import os
from typing import Tuple, List
import json

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import ToTensor, Lambda, Compose


def log_transform(x):
    return torch.log10(torch.clamp(x, min=1e-8))

class inpainting_DS(Dataset):
    """
    A PyTorch Dataset for loading sequential frames of data (e.g., chlorophyll data),
    optionally applying masks and transformations.
    """
    def __init__(self, 
                 root: str, 
                 frames: int, 
                 shape_scale=(48, 48),
                 is_month:bool=False, 
                 mask_dir:str=None, 
                 is_train:bool=True, 
                 train_test_ratio:float=0.7, 
                 task_name:str=None,
                 chunk_list:list=None,
                 mask_chunk_list:list=None,
                 global_mean:float=None,
                 global_std:float=None,
                 apply_log:bool=True,
                 apply_normalize:bool=True):
        """
        Args:
            root (str): Directory containing .npy files for data.
            frames (int): Number of frames (temporal dimension) per sample.
            shape_scale (tuple): If resizing is needed, (height, width).
            is_month (bool): Indicates if monthly mask is used.
            mask_dir (str): Directory containing corresponding mask npy files.
            is_train (bool): If True, dataset is for training; else for validation/testing.
            train_test_ratio (float): Ratio for splitting train/validation sets.
            task_name (str): Not fully utilized, can be used to specify a certain task (like 'dineof').
            chunk_list (list): Pre-computed list of data file paths grouped by frames.
            mask_chunk_list (list): Pre-computed list of mask file paths grouped by frames.
            global_mean (float): Global mean for normalization.
            global_std (float): Global std for normalization.
            apply_log (bool): Whether to apply log10 transform.
            apply_normalize (bool): Whether to apply normalization with provided mean/std.
        """
        super().__init__()
        self.root = root
        self.frames = frames
        self.shape_scale = shape_scale
        self.is_month = is_month
        self.is_train = is_train
        self.train_test_ratio = train_test_ratio
        self.task_name = task_name

        if self.is_month:
            assert mask_dir is not None, "mask_dir must be specified if is_month=True"
            self.mask_dir = mask_dir

        # If no external chunk list is provided, generate it
        if chunk_list is None:
            full_list = self._dir_list_select_combine(self.root, suffix='.npy')
            chunk_list = self._create_chunks(full_list, self.frames)
            chunk_list = self._train_test_split(chunk_list)
        self.chunk_list = chunk_list

        if self.is_month:
            if mask_chunk_list is None:
                full_mask_list = self._dir_list_select_combine(self.mask_dir, suffix='.npy')
                mask_chunk_list = self._create_chunks(full_mask_list, self.frames)
                mask_chunk_list = self._train_test_split(mask_chunk_list)
            self.mask_chunk = mask_chunk_list
        else:
            self.mask_chunk = []

        self.length = len(self.chunk_list)
        
        self.global_mean = global_mean
        self.global_std = global_std
        self.apply_log = apply_log
        self.apply_normalize = apply_normalize

        # Build transformation pipeline
        transform_list = [ToTensor()]
        if self.apply_log:
            # log10 requires positive values, handled by replacing zeros/nans before this step
            transform_list.append(Lambda(log_transform))
        
        # If resizing is required:
        transform_list.append(transforms.Resize(self.shape_scale))
        
        if self.apply_normalize and (self.global_mean is not None and self.global_std is not None):
            transform_list.append(transforms.Normalize(mean=[self.global_mean], std=[self.global_std]))
        
        self.transform = Compose(transform_list)

    def _train_test_split(self, chunk_list: List[List[str]]) -> List[List[str]]:
        np.random.shuffle(chunk_list)
        split_idx = int(self.train_test_ratio * len(chunk_list))
        return chunk_list[:split_idx] if self.is_train else chunk_list[split_idx:]

    @staticmethod
    def _dir_list_select_combine(base_dir: str, suffix: str='.npy') -> List[str]:
        files_list = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.endswith(suffix)]
        files_list.sort()
        return files_list

    @staticmethod
    def _create_chunks(file_list: List[str], frames: int) -> List[List[str]]:
        return [file_list[i:i+frames] for i in range(len(file_list)-frames+1)]

    def __len__(self):
        return self.length

    @staticmethod
    def _load_npy(file_path: str) -> np.ndarray:
        arr = np.load(file_path)
        return arr

    @staticmethod
    def _apply_mask_logic(data: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # If mask is given (cloud mask), mask should align with data shape
        assert mask.shape == data.shape, f"Mask shape {mask.shape} doesn't match data {data.shape}"
        return data * mask, mask

    @staticmethod
    def _apply_random_mask(data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shape = data.shape
        rand_mask = torch.from_numpy(np.where(np.random.normal(loc=100, scale=10, size=shape) < 90, 0, 1)).float()
        return data * rand_mask, rand_mask

    def __getitem__(self, index: int):
        chunk = self.chunk_list[index]
        if self.is_month:
            mask_index = min(index, len(self.mask_chunk)-1)
            mask_paths = self.mask_chunk[mask_index]
        else:
            mask_paths = [None]*len(chunk)

        inputs_list = []
        target_list = []
        # masks_list = []

        for data_path, mask_path in zip(chunk, mask_paths):
            array = self._load_npy(data_path)
            array[np.isnan(array)] = 1.0
            array[array == 0.0] = 1.0
            
            data_tensor = self.transform(array)
            
            
            if self.is_month and mask_path is not None:
                mask_array = np.load(mask_path)
                mask_array[np.isnan(mask_array)] = 0.0
                mask_tensor = torch.resize_as_(torch.from_numpy(mask_array).unsqueeze(0).float(), data_tensor)
                in_data, mask_data = self._apply_mask_logic(data_tensor, mask_tensor)
            else:
                in_data, mask_data = self._apply_random_mask(data_tensor)

            assert ~torch.any(torch.isnan(in_data)), "there are nan generate from here"
            inputs_list.append(in_data)
            target_list.append(data_tensor)
            # masks_list.append(mask_data)

        inputs = torch.stack(inputs_list, dim=0)
        targets = torch.stack(target_list, dim=0)
        # masks = torch.stack(masks_list, dim=0)
        
        return index, inputs.float(), targets.float()
    
    
class DataProcessor:
    def __init__(self, cropped_root, cropped_ds_dir=None):
        self.cropped_root = cropped_root
        self.cropped_ds_dir = cropped_ds_dir

    def _key_to_str(self, key: Tuple[int, int]) -> str:
        """将元组键转换为字符串"""
        return f"{key[0]},{key[1]}"

    def _key_to_tuple(self, key_str: str) -> Tuple[int, int]:
        """将字符串键转换为元组"""
        return tuple(map(int, key_str.split(',')))

    def process_data(self):
        files = [f for f in os.listdir(self.cropped_root) if f.endswith('.npy')]
        data_dict = {}
        json_file_path = self.cropped_ds_dir

        if json_file_path is not None and os.path.exists(json_file_path):
            # 如果存在数据字典文件，直接读取
            with open(json_file_path, 'r') as f:
                data_dict = json.load(f)
            # 将字符串键解析回元组形式
            data_dict = {self._key_to_tuple(k): v for k, v in data_dict.items()}
            return data_dict

        for f in files:
            parts = f.split('_')
            base_time = parts[1]                 # 提取时间信息
            time_info = base_time.split('.')[1]  # 例如 '20020801_20020831'

            # 提取空间索引
            r_part = parts[-2]                   # 'r{i}'
            c_part = parts[-1]                   # 'c{j}.npy'
            i = int(r_part[1:])
            j = int(c_part[1:-4])                # 去除 'c' 和 '.npy'

            key = (i, j)                         # 保留元组形式
            if key not in data_dict:
                data_dict[key] = []
            data_dict[key].append((time_info, f))

        # 按时间排序每个空间位置的文件
        for key in data_dict:
            data_dict[key].sort(key=lambda x: x[0])  # 根据时间信息排序

        # 保存 data_dict 到 JSON 文件
        if json_file_path is not None and not os.path.exists(json_file_path):
            # 在保存时将元组键转换为字符串
            serializable_dict = {self._key_to_str(k): v for k, v in data_dict.items()}
            with open(json_file_path, 'w') as f:
                json.dump(serializable_dict, f)

        return data_dict


class inpainting_DS_v2(Dataset):
    """
    一个用于加载裁剪后的序列帧数据的 PyTorch Dataset 类。
    确保同一空间位置的不同时间帧被正确加载，并保留所有原始功能。
    """
    def __init__(self, 
                 root: str, 
                 frames: int, 
                 inputChunkJSON_dir: str=None,
                 shape_scale: Tuple[int, int] = (256, 256),
                 is_mask: bool = False, 
                 mask_root: str = None, 
                 is_train: bool = True, 
                 train_test_ratio: float = 0.7, 
                 global_mean: float = None,
                 global_std: float = None,
                 apply_log: bool = True,
                 apply_normalize: bool = True):
        """
        Args:
            cropped_root (str)      : 裁剪后的 .npy 文件所在的目录。
            frames (int)            : 每个样本包含的时间帧数。
            cropped_ds_dir (str)    : 包含裁剪后的 .npy 文件的jsonwebtoken所在的目录。
            shape_scale (tuple)     : 图像的缩放尺寸。
            is_month (bool)         : 是否使用月掩码。
            mask_dir (str)          : 掩码文件所在的目录。
            is_train (bool)         : 是否为训练集。
            train_test_ratio (float): 训练集和测试集的划分比例。
            global_mean (float)     : 用于归一化的全局均值。
            global_std (float)      : 用于归一化的全局标准差。
            apply_log (bool)        : 是否应用对数变换。
            apply_normalize (bool)  : 是否应用标准化。
        """
        super().__init__()
        self.root = root
        self.frames = frames
        self.inputChunkJSON_dir = inputChunkJSON_dir
        self.shape_scale = shape_scale
        self.is_month = is_mask
        self.is_train = is_train
        self.train_test_ratio = train_test_ratio
        self.global_mean = global_mean
        self.global_std = global_std
        self.apply_log = apply_log
        self.apply_normalize = apply_normalize

        if self.is_month:
            assert mask_root is not None, "mask_dir must be specified if is_month=True"
            self.mask_root = mask_root

        # Organize files and sort them by location and time
        self.data_dict = self._organize_files(self.root, self.inputChunkJSON_dir)
        self.sorted_keys = sorted(self.data_dict.keys())

        # create chunk_list and make sure each chunk contains frames from the same location
        self.chunk_list = self._create_chunks(self.root, self.data_dict)

        # dataset length
        self.length = len(self.chunk_list)

        # process masks
        if self.is_month:
            self.mask_dict = self._organize_files(self.mask_root)
            mask_chunk_list = self._create_chunks(self.mask_root, self.mask_dict)
            self.mask_chunk = mask_chunk_list
        else:
            self.mask_dict = {}

        # Build transformation pipeline
        transform_list = [ToTensor()]
        if self.apply_log:
            transform_list.append(Lambda(log_transform))
        if self.shape_scale:
            transform_list.append(transforms.Resize(self.shape_scale))
        if self.apply_normalize and (self.global_mean is not None and self.global_std is not None):
            transform_list.append(transforms.Normalize(mean=[self.global_mean], std=[self.global_std]))
        self.transform = Compose(transform_list)

    def _organize_files(self, root:str, json_file_path:str=None) -> dict:
        """
        organize files by location and time.

        Returns:
            dict: keys are (i, j), values are a list of tuples (time_info, file_path).
        """
        dataprocess = DataProcessor(root, json_file_path)
        data_dict = dataprocess.process_data()
        return data_dict

    def _create_chunks(self, root:str, data_dict:dict) -> List[List[str]]:
        """
        创建 chunk_list，每个 chunk 包含连续的时间帧，并且来自同一空间位置。

        Returns:
            List[List[str]]: 每个元素是一个 chunk，包含多个文件路径。
        """
        chunks = []
        for key, files in data_dict.items():
            # 确保有足够的帧
            if len(files) < self.frames:
                continue
            for i in range(len(files) - self.frames + 1):
                chunk = [os.path.join(root, files[i + j][1]) for j in range(self.frames)]
                chunks.append(chunk)
        # 随机打乱并划分训练集和测试集
        np.random.shuffle(chunks)
        split_idx = int(self.train_test_ratio * len(chunks))
        return chunks[:split_idx] if self.is_train else chunks[split_idx:]

    def __len__(self):
        return self.length

    def __getitem__(self, index: int):
        chunk = self.chunk_list[index]
        inputs_list = []
        target_list = []
        masks_path = self.mask_chunk[min(index, len(self.mask_chunk)-1)] if self.is_month else None

        for data_path, mask_path in zip(chunk, masks_path):
            array = np.load(data_path)
            array[np.isnan(array)] = 1.0
            array[array == 0.0] = 1.0

            data_tensor = self.transform(array)

            if self.is_month and mask_path is not None:
                mask_array = np.load(mask_path)
                if np.any(np.isnan(mask_array)):
                    print(f"mask_array: {mask_array.shape}, {np.any(np.isnan(mask_array))}")
                mask_array[~np.isnan(mask_array)] = 1.0
                mask_array[np.isnan(mask_array)] = 0.0
                mask_tensor = torch.resize_as_(torch.from_numpy(mask_array).unsqueeze(0).float(), data_tensor)
                in_data, _ = self._apply_mask_logic(data_tensor, mask_tensor)
            else:
                in_data, mask_data = self._apply_random_mask(data_tensor)
                print(f"mask_data: {mask_data.shape}")
                
            inputs_list.append(in_data)
            target_list.append(data_tensor)

        inputs = torch.stack(inputs_list, dim=0)
        targets = torch.stack(target_list, dim=0)
        # if torch.any(torch.isnan(inputs)) or torch.any(torch.isnan(targets)):
        #     print(f"there are nan generate from here, index: {index}")
        return index, inputs.float(), targets.float()

    @staticmethod
    def _apply_mask_logic(data: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        应用掩码逻辑。

        Args:
            data (torch.Tensor): 数据张量。
            mask (torch.Tensor): 掩码张量。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 处理后的数据和掩码。
        """
        assert mask.shape == data.shape, f"Mask shape {mask.shape} doesn't match data {data.shape}"
        mask[torch.isnan(mask)] = 0.0
        return data * mask, mask

    @staticmethod
    def _apply_random_mask(data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        随机应用掩码。

        Args:
            data (torch.Tensor): 数据张量。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 处理后的数据和掩码。
        """
        shape = data.shape
        rand_mask = torch.from_numpy(np.where(np.random.normal(loc=100, scale=10, size=shape) < 90, 0, 1)).float()
        return data * rand_mask, rand_mask


# test inpainting_DS_v2
def test_inpainting_DS_v2():
    cropped_root = '/root/autodl-tmp/input_256'
    mask_dir = '/root/autodl-tmp/mask_256'
    croped_ds_dir = '/root/dataset/cropped_ds_256.json'
    ds = inpainting_DS_v2(cropped_root, 10, croped_ds_dir, is_mask=True, mask_root=mask_dir, is_train=True, train_test_ratio=0.7)
    print(len(ds))
    for i in range(len(ds)):
        index, inputs, targets = ds[i]
        print(index, inputs.shape, targets.shape)

if __name__ == '__main__':
    test_inpainting_DS_v2()