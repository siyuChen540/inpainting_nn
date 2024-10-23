CKPT_FILE = "D:/ceeres/03_Program/01_SYSUM/00_science/04_image_recovery/01_program/git_repo/convlstm_en_de/record/ckpts/20230323-1116/checkpoint_422_0.109316.pth.tar"

FRAME_NUM = 4

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
logging.basicConfig(filename='test_model.log', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def dir_list_select_combine(base_dir:str, suffix_condition: str='.npy',ismask=False) -> list:
    files_dir = os.listdir(base_dir)
    files_dir = [file_name for file_name in files_dir if os.path.splitext(file_name)[1] == suffix_condition]
    assert len(files_dir) > 0
    # 根据12到20的字符位置的数字来排序
    if ismask:
        pass
    else:
        # use for 8 days
        try:
            files_dir.sort(key=lambda x:int(x[11:19]))
        except:
            files_dir.sort(key=lambda x:int(x.split('.')[0][0:8]))
        # files_dir.sort(key=lambda x:int(x[11:19]))
        
    suffix_files_list:list = [
        os.path.join(base_dir, file_name) for file_name in files_dir if os.path.splitext(file_name)[1] == suffix_condition]
    
    return suffix_files_list

def get_chunk_list(root, suffix_condition='.npy', mask=False, mask_dir=None,frames=10) -> list:
    frame_list:list = []
    contain_list:list = []
    
    dir = root
    if mask:
        dir = mask_dir
        ismask=True
    else:
        ismask=False
    dir_list = dir_list_select_combine(dir, suffix_condition, ismask=ismask)

    for i in range(len(dir_list) - frames+1):
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
    
def load(data_dir:list, mask_dir:list,results:str,frame:int=10):
    label_contian = np.empty(shape=(frame,1,480,480))
    mask_contian = np.empty(shape=(frame,1,480,480))
    input_contian = np.empty(shape=(frame,1,480,480))
    
    for i in range(frame):
        with open(data_dir[i], 'rb') as frame_ds:
            label = np.load(frame_ds)
            label_contian[i,0,:,:] = custom_transform(label)
        with open(mask_dir[i], 'rb') as mask_ds:
            mask = np.load(mask_ds)
            input_contian[i,0,:,:], mask_contian[i,:,:] = custom_transform(label, mask, state="cloud mask")
    
    input_contian = torch.unsqueeze(from_numpy(input_contian).to(torch.float32),0)
    label_contian = torch.unsqueeze(from_numpy(label_contian).to(torch.float32),0)
    # month
    # return input_contian, label_contian, mask_contian, data_dir[-1].split('/')[-1].split('.')[1][:6]
    # 8days
    # return input_contian, label_contian, mask_contian, data_dir[-1].split('/')[-1].split('.')[0][:8]
    # torch gpu 4-bidirection lstm windows system
    file_name = data_dir[-1].split('\\')[-1]
    return input_contian, label_contian, mask_contian, file_name

def fconvlstm_gpu(root_dir):
    torch.cuda.manual_seed(540)
    # auto select best algorithm for speeding up
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    from model.net_structor import fconvlstm_scs_min_cache_encoder_params
    from model.net_structor import fconvlstm_scs_min_cache_decoder_params
    encoder = Encoder(fconvlstm_scs_min_cache_encoder_params[0], fconvlstm_scs_min_cache_encoder_params[1]).cuda()
    decoder = Decoder(fconvlstm_scs_min_cache_decoder_params[0], fconvlstm_scs_min_cache_decoder_params[1]).cuda()
    model = ED(encoder, decoder)
    device = torch.device("cuda:0")
    model.to(device)

    ckpts_dir = CKPT_FILE    
    mask_dir = 'D:/ceeres/03_Program/01_SYSUM/00_science/04_image_recovery/01_program/00_dataset/chl_mask'

    model_info = torch.load(ckpts_dir)
    model.load_state_dict(model_info['state_dict'])
    model.eval()

    results   = os.path.join(root_dir, "20231106_bidirection/")
    os.makedirs(results,exist_ok=True)

    # Forward input data and save the results
    mask_contain_list = get_chunk_list(root=root_dir, mask=True,mask_dir=mask_dir,frames=FRAME_NUM)
    contian_list = get_chunk_list(root=root_dir,frames=FRAME_NUM)

    contain_save = np.empty(shape=(3,480,480))

    with torch.no_grad():
        for i in tqdm(range(len(contian_list)), desc='Processing', unit='items', unit_scale=True):
        # for i in tqdm(range(180,len(contian_list)), desc='Processing', unit='items', unit_scale=True):
            inputs, targets, mask, file_dir = load(contian_list[i], mask_contain_list[i], frame=FRAME_NUM, results=results)
            X = inputs.to(device)
            Y = targets.to(device)
            outputs = model(X)
            
            # Save the results
            contain_save[0] = inputs.numpy()[0,-1,0]        
            contain_save[1] = targets.numpy()[0,-1,0]        
            contain_save[2] = outputs.cpu().numpy()[0,-1,0]        
            
            np.save(f"{results}\{file_dir}",contain_save)
            logging.info(f'Processed {file_dir} items', )

if __name__ == "__main__":
    root_dir = 'D:/ceeres/03_Program/01_SYSUM/00_science/04_image_recovery/01_program/00_dataset/chlora_month/rename_input'
    fconvlstm_gpu(root_dir)
    logging.info(f'Processed {root_dir} items', )