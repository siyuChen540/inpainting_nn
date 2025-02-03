import os
import re
import json
from typing import Tuple, List, Dict, Any

import numpy as np
import torch
from torch import Tensor
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
                 resize_shape: Tuple[int, int] = (256, 256),
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
            array[np.isnan(array)] = 1.0
            array[array == 0.0] = 1.0
            data_tensor = self.transform(array)
            if self.enable_mask and mask_path is not None:
                mask_array = np.load(mask_path)
                mask_array[~np.isnan(mask_array)] = 1.0
                mask_array[np.isnan(mask_array)] = 0.0
                mask_tensor = torch.resize_as_(torch.from_numpy(mask_array).unsqueeze(0).float(), data_tensor)
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
    def _apply_random_mask(data: Tensor) -> Tuple[Tensor, Tensor]:
        shape = data.shape
        rand_mask = torch.from_numpy(np.where(np.random.normal(loc=100, scale=10, size=shape) < 90, 0, 1)).float()
        return data * rand_mask, rand_mask


# Detailed test functions to ensure dataset stability and error handling.
def test_inpainting_dataset_v2() -> None:
    """
    Test function for InpaintingDatasetV2.
    Validates loading, transformation and error handling.
    """
    test_root = 'E:/04_DevelopReleas/02_test_MArineSIR/dataset/train/mask_256/'
    test_mask_dir = 'E:/04_DevelopReleas/02_test_MArineSIR/dataset/train/mask_256/'
    try:
        dataset = InpaintingDatasetV2(test_root, num_frames=10, json_file_path=None,
                                      enable_mask=True, mask_root_dir=test_mask_dir, is_training_set=True, train_split_ratio=0.7)
    except Exception as e:
        print(f"Error initializing dataset: {e}")
        return
    from time import time
    print(f"Dataset length: {len(dataset)}")
    for i in range(20):
        try:
            # Load and process sample. 
            # record time cost
            start = time()            
            idx, inputs, targets = dataset[i]
            print(f"Time cost for sample {i}: {time()-start}")
            assert inputs.shape == targets.shape, "Input and target shapes mismatch"
            print(f"Sample {idx}: inputs shape {inputs.shape}, targets shape {targets.shape}")
        except Exception as err:
            print(f"Error processing sample {i}: {err}")

if __name__ == '__main__':
    # Run tests for both datasets.
    print("Testing InpaintingDatasetV2...")
    test_inpainting_dataset_v2()
    # ...existing test code for InpaintingDataset if needed...