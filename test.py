import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset.chlora import ChloraData
from model.encoder import Encoder
from model.decoder import Decoder
from model.ed import ED
from model.net_structor import convlstm_scs_encoder_params, convlstm_scs_decoder_params
from tqdm import tqdm

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
model_info = torch.load(ckpts_dir)
model.load_state_dict(model_info['state_dict'])
model.eval()

# Define data loader
ds_base  = "/home/chensiyu/workspace/01_main_dataset/02_reconstruct_dataset/"
root_dir = os.path.join(ds_base, "02_chl_mon_scs_npy/chlora")
mask_dir = os.path.join(ds_base, "02_chl_8d_scs_npy/chl_mask")

code_dir  = "/home/chensiyu/workspace/02_scientist_program/02_img_recovery"
stats_dir = os.path.join(code_dir, "record/stats/")
img_dir   = os.path.join(code_dir, "record/img/")
log_dir   = os.path.join(code_dir, "record/log/")

results   = os.path.join(code_dir, "test/20230315-ED/")
os.makedirs(results,exist_ok=True)

frames      = 10  
shape_scale = [480,480]

train_dataset = ChloraData(root_dir, frames, shape_scale, is_month=True, mask_dir=mask_dir, is_train=True, is_shuffle=False)
train_loader  = DataLoader(train_dataset, batch_size=1, shuffle=False)
test_dataset = ChloraData(root_dir, frames, shape_scale, is_month=True, mask_dir=mask_dir, is_train=False, is_shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Forward input data and save the results

contain_save = np.empty(shape=(3,10,480,480))
step = 0
with torch.no_grad():
    for i, (index, inputs, targets, mask) in enumerate(tqdm(train_loader, total=len(train_loader))):
        
        # Forward input data
        outputs = model(inputs)
        
        # Save the results
        contain_save[0] = inputs.numpy()[0,:,0]        
        contain_save[1] = targets.numpy()[0,:,0]        
        contain_save[2] = outputs.numpy()[0,:,0]        
        
        np.save(f"{results}{step}-{index}.npy",contain_save)
        step+=1
    
    for i, (index, inputs, targets, mask) in enumerate(tqdm(test_loader, total=len(test_loader))):
        
        # Forward input data
        outputs = model(inputs)
        
        # Save the results
        contain_save[0] = inputs.numpy()[0,:,0]        
        contain_save[1] = targets.numpy()[0,:,0]        
        contain_save[2] = outputs.numpy()[0,:,0]        
        np.save(f"{results}{step}-{index}.npy",contain_save)
        step+=1
        