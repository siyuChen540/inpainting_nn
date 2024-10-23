"""
    @author: SiyuChen540
    @date: 2024.10.23
    @file: test_0413.py
    @version: 2.0.0
    @description: This script is used to test the trained model on the test dataset.
    @email: chensy57@mail2.sysu.edu.cn
"""

import os
import torch
import numpy as np
from model.encoder import Encoder
from model.decoder import Decoder
from model.ed import ED
from model.net_structor import convlstm_scs_encoder_params, convlstm_scs_decoder_params
from tqdm import tqdm
from torch import from_numpy
from torchvision.transforms import Lambda, Resize, Compose, ToPILImage, ToTensor

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def dir_list_select_combine(base_dir:str, suffix_condition: str='.npy') -> list:
    files_dir = os.listdir(base_dir)
    
    assert len(files_dir) > 0
    
    suffix_files_list:list = [
        os.path.join(base_dir, file_name) for file_name in files_dir if os.path.splitext(file_name)[1] == suffix_condition]
    
    return suffix_files_list

def get_chunk_list(root, suffix_condition='.npy', mask=False, mask_dir=None,frames=10) -> list:
    frame_list:list = []
    contain_list:list = []
    
    dir = root
    if mask:
        dir = mask_dir
    
    dir_list = dir_list_select_combine(dir, suffix_condition)

    for i in range(len(dir_list) - frames):
        start:int = i
        end:int = i + frames

        frame_list = dir_list[start:end]
        contain_list.append(frame_list)
    return contain_list

def custom_transform(array, mask_array=None, state=None):
    array[np.isnan(array)] = 1
        
    array[array==0.0] = 1
    
    log_lambda = Lambda(lambda x:np.log10(x))
    
    transform = Compose([
        log_lambda,
        ToTensor(),
    ])
    
    result_array = transform(array)
    
    if not state:
        return result_array
    elif state == "cloud mask":
        mask = torch.unsqueeze(from_numpy(mask_array), 0)
        return result_array * mask, mask
    

def load(data_dir:list, mask_dir:list,frame:int=10):
    label_contian = np.empty(shape=(frame,1,480,480))
    mask_contian = np.empty(shape=(frame,1,480,480))
    input_contian = np.empty(shape=(frame,1,480,480))
    

    # 使用日期部分作为排序键
    data_dir = sorted(data_dir, key=lambda x: int(x.split('/')[-1].split('.')[0]))

    for i in range(frame):
        with open(data_dir[i], 'rb') as frame_ds:
            label = np.load(frame_ds)
            label_contian[i,0,:,:] = custom_transform(label)
        with open(mask_dir[i], 'rb') as mask_ds:
            mask = np.load(mask_ds)
            input_contian[i,0,:,:], mask_contian[i,:,:] = custom_transform(label, mask, state="cloud mask")
    
    input_contian = torch.unsqueeze(from_numpy(input_contian).to(torch.float32),0)
    label_contian = torch.unsqueeze(from_numpy(label_contian).to(torch.float32),0)
    
    return input_contian, label_contian, mask_contian, data_dir[0].split('/')[-1].split('.')[0][:6]

def convlstm_test():
    # Load the trained model
    model = ED(
                Encoder(
                    convlstm_scs_encoder_params[0], 
                    convlstm_scs_encoder_params[1]), 
                Decoder(
                    convlstm_scs_decoder_params[0], 
                    convlstm_scs_decoder_params[1])
            )

    ckpts_dir = "/home/chensiyu/workspace/02_scientist_program/02_img_recovery/record/ckpts/20230106-0141/checkpoint_149_0.201368.pth.tar"
    code_dir  = "/home/chensiyu/workspace/02_scientist_program/02_img_recovery"
    ds_base  = "/home/chensiyu/workspace/01_main_dataset/02_reconstruct_dataset/"
    # Define data loader
    root_dir = os.path.join(ds_base, "02_chl_mon_scs_npy/chlora")
    mask_dir = os.path.join(ds_base, "02_chl_8d_scs_npy/chl_mask")

    model_info = torch.load(ckpts_dir)
    model.load_state_dict(model_info['state_dict'])
    model.eval()



    results   = os.path.join(code_dir, "test/20230411-ED/")
    os.makedirs(results,exist_ok=True)

    frames      = 10  
    shape_scale = [480,480]

    # Forward input data and save the results
    mask_contain_list = get_chunk_list(root=root_dir, mask=True,mask_dir=mask_dir)
    contian_list = get_chunk_list(root=root_dir)

    contain_save = np.empty(shape=(3,480,480))

    with torch.no_grad():
        for i in tqdm(range(len(contian_list)), desc='Processing', unit='items', unit_scale=True):
            inputs, targets, mask, file_dir = load(contian_list[i], mask_contain_list[i])
            
            # Forward input data
            outputs = model(inputs)
            
            # Save the results
            contain_save[0] = inputs.numpy()[0,0,0]        
            contain_save[1] = targets.numpy()[0,0,0]        
            contain_save[2] = outputs.numpy()[0,0,0]        
            
            np.save(f"{results}{file_dir}.npy",contain_save)
            logging.info(f'Processed {file_dir} items', )
def fconvlstm():
    from model.net_structor import fconvlstm_scs_encoder_params
    from model.net_structor import fconvlstm_scs_decoder_params
    
    from model.net_structor import fconvlstm_scs_min_cache_encoder_params
    from model.net_structor import fconvlstm_scs_min_cache_decoder_params

    model = ED(
                Encoder(
                    fconvlstm_scs_min_cache_encoder_params[0], 
                    fconvlstm_scs_min_cache_encoder_params[1]), 
                Decoder(
                    fconvlstm_scs_min_cache_decoder_params[0], 
                    fconvlstm_scs_min_cache_decoder_params[1])
            )

    ckpts_dir = "/root/autodl-fs/CKPTS/checkpoint_480_0.095276.pth.tar"

    # Define data loader
    root_dir = "/autodl-fs/data/inpaint_scs_data"
    mask_dir = "/autodl-fs/data/inpaint_scs_data"

    model_info = torch.load(ckpts_dir)
    # print(model_info['state_dict'])
    model.load_state_dict(model_info['state_dict'])
    model.eval()

    model = model.cuda()

    results   = os.path.join("/root/autodl-fs/monthResult_10/", "20230411-FED/")
    os.makedirs(results,exist_ok=True)

    frames      = 10  
    shape_scale = [480,480]

    # Forward input data and save the results
    mask_contain_list = get_chunk_list(root=root_dir, mask=True,mask_dir=mask_dir)
    contian_list = get_chunk_list(root=root_dir)

    contain_save = np.empty(shape=(3, 10, 480,480))

    with torch.no_grad():
        for i in tqdm(range(len(contian_list)), desc='Processing', unit='items', unit_scale=True):
            inputs, targets, mask, file_dir = load(contian_list[i], mask_contain_list[i])
            #targets send into cuda
            targets = targets.cuda()
            print(next(model.parameters()).device)

            # Forward input data
            outputs:torch.Tensor = model(targets)
            
            # Save the results
            contain_save[0] = inputs.numpy()[0,:,0]        
            contain_save[1] = targets.cpu().numpy()[0,:,0]        
            contain_save[2] = outputs.cpu().numpy()[0,:,0]        
            
            np.save(f"{results}{file_dir}.npy",contain_save)
            logging.info(f'Processed {file_dir} items', )
if __name__ == "__main__":
    # convlstm_test()
    fconvlstm()