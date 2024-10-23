import numpy as np
import os
import torch
from torch import float32, from_numpy
from torch import mean, std
import torch.utils.data as data
import os
from torchvision.transforms import Lambda, Resize, Compose, ToPILImage, ToTensor

class ChloraData(data.Dataset):
    def __init__(self, root, frames, shape_scale:set=(48,48),transoform=None, is_month=False, mask_dir=None, is_train=True, train_test_ratio = 0.7,task_name=None,is_shuffle=True):
        super(ChloraData, self).__init__()
        # define img_dir and label_dir
        self.is_train = is_train
        self.dataset = None
        self.root = root
        self.frames = frames
        self.shape_scale = shape_scale
        self.transform = transoform
        self.is_month = is_month
        self.is_dineof = task_name
        self.is_shuffle = is_shuffle
        self.train_test_ratio = train_test_ratio
        self.chunk_list = self.get_chunk_list('.npy')
        self.length = len(self.chunk_list)
        
        if self.is_month:
            assert mask_dir != None, "not setting mask folder directory"
            self.mask_dir = mask_dir
            self.mask_chunk = self.get_chunk_list('.npy', True)

    def train_test_split(self,chunk_list:list):
        if self.is_shuffle:
            np.random.shuffle(chunk_list)
        else:
            pass
        
        if self.is_train:
            return chunk_list[0:int(self.train_test_ratio*len(chunk_list)+1)]
        else:
            return chunk_list[int(self.train_test_ratio*len(chunk_list)-1):]
    
    def __getitem__(self, index):
        chunk_len = len(self.chunk_list[index])
        mask_index = np.random.randint(index, len(self.mask_chunk))
        inputs:np.ndarray = np.empty(
            shape=(
                self.frames, 
                1, 
                self.shape_scale[0], 
                self.shape_scale[1]))
        target:np.ndarray = np.empty(
            shape=(
                self.frames, 
                1, 
                self.shape_scale[0], 
                self.shape_scale[1]))
        mask:np.ndarray = np.empty(
            shape=(
                self.frames, 
                1, 
                self.shape_scale[0], 
                self.shape_scale[1]))
        
        for i in range(chunk_len):
            var_dir = self.chunk_list[index][i]
            with open(var_dir, 'rb') as frame_ds:
                buffer_array = np.load(frame_ds)
                target[i,0] = self.custom_transform(dir, buffer_array)
            if self.is_month:
                mask_dir = self.mask_chunk[mask_index][i]
                with open(mask_dir, 'rb') as frame_ds:
                    mask_array = np.load(frame_ds)
                inputs[i,0], mask[i] = self.custom_transform(dir, buffer_array, "cloud mask", mask_array)
            else:
                inputs[i,0], mask[i] = self.custom_transform(dir, buffer_array, "random noise mask")
                
        inputs = from_numpy(inputs).to(float32)
        target = from_numpy(target).to(float32)

        return [index, inputs, target, mask]

    def dir_list_select_combine(self, base_dir:str, suffix_condition: str='.npy') -> list:
        files_dir = os.listdir(base_dir)
        
        assert len(files_dir) > 0
        
        suffix_files_list:list = [
            os.path.join(base_dir, file_name) for file_name in files_dir if os.path.splitext(file_name)[1] == suffix_condition]
        
        return suffix_files_list

    def get_chunk_list(self, suffix_condition='.npy', mask=False) -> list:
        frame_list:list = []
        contain_list:list = []
        
        dir = self.root
        if mask:
            dir = self.mask_dir
        
        dir_list = self.dir_list_select_combine(dir, suffix_condition)

        for i in range(len(dir_list) - self.frames):
            start:int = i
            end:int = i + self.frames

            frame_list = dir_list[start:end]
            contain_list.append(frame_list)
        chunk_list = self.train_test_split(contain_list)
        return chunk_list
    
    def custom_transform(self,dir, array:np.ndarray, state:str=None, mask_array=None):
        array[np.isnan(array)] = 1
        
        assert ~np.any(np.isnan(array)), f"{dir} nan did not remove clearly"
        
        array[array==0.0] = 1
        
        log_lambda = Lambda(lambda x:np.log10(x))
        
        resize = Resize(self.shape_scale)
        normalize_lambda = Lambda(lambda x: (x-mean(x))/std(x))

        transform = Compose([
            log_lambda,
            ToTensor(),
            # ToPILImage(),
            # resize,
            # ToTensor(),
            # normalize_lambda,
        ])
        
        result_array = transform(array)
        
        assert ~torch.any(torch.isnan(result_array)), "there are nan generate from here"
        
        if not state:
            return result_array
        elif state == "cloud mask":
            mask = torch.unsqueeze(from_numpy(mask_array), 0)
            assert mask.shape == result_array.shape, "mask shape do not match with result array"            
            return result_array * mask, mask
        else:
            mask:np.ndarray = np.random.normal(loc=100, scale=10, size=result_array.shape)
            mask[mask<90] = 0
            mask[mask>=90] = 1

            return result_array * mask, mask
    
    def __len__(self):
        return self.length

if __name__ == "__main__":
    
    root = "/home/chensiyu/workspace/01_main_dataset/02_reconstruct_dataset/02_chl_mon_scs_npy/chlora"
    mask = "/home/chensiyu/workspace/01_main_dataset/02_reconstruct_dataset/02_chl_8d_scs_npy/chl_mask"
    
    frames = 10
    shape_scale=(480,480)
    
    trainFolder = ChloraData(root, frames, shape_scale, is_month = True, mask_dir = mask)
    
    from torch.utils.data import DataLoader
    trainloder = DataLoader(trainFolder, batch_size=2, shuffle=False)
    len(trainFolder)
    index, inputs, target, mask = trainFolder.__getitem__(index=240)
    for _, (index, inputs, target, mask) in enumerate(trainloder):
        print(inputs.shape)