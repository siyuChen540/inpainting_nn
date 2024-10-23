from tensorboardX import SummaryWriter

import os
import yaml
import numpy as np
from tqdm import tqdm

# datasets
from dataset.chlora import ChloraData

# model consrtuct
from model.encoder import Encoder
from model.decoder import Decoder
from model.ed import ED

# model parameters
from model.net_structor import convlstm_scs_encoder_params
from model.net_structor import convlstm_scs_decoder_params
from model.net_structor import fconvlstm_scs_encoder_params
from model.net_structor import fconvlstm_scs_decoder_params
from model.net_structor import fconvlstm_scs_min_cache_encoder_params
from model.net_structor import fconvlstm_scs_min_cache_decoder_params

# local earlyStoper
from earlystopping import EarlyStopping

# custom record param
from utils import draw_train_array
from utils import record_dir_setting_create

# custom loss function
import pytorch_ssmi
import torch
from torch import nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_value_


yaml_dir = r"/root/workspace/src/record/param/20231108.yml"
with open(yaml_dir, "r") as aml:
    yml_params = yaml.safe_load(aml)

EPOCH       = yml_params["EPOCH"]
BATCH       = yml_params["BATCH"]
LR          = yml_params["LR"]
frames      = yml_params["frames"]
shape_scale = yml_params["shape_scale"]

random_seed = yml_params["random_seed"]
root_dir    = yml_params["root_dir"]
mask_dir    = yml_params["mask_dir"]
is_month    = yml_params["is_month"]

device      = yml_params["device"]
ckpt_files  = yml_params["ckpt_files"]

try:
    fconv       = yml_params["fconv"]
    min_cache   = yml_params["min_cache"]
    beta        = yml_params["beta"]
except:
    fconv       = False
    min_cache   = False
    beta        = 0.5

save_dir    = record_dir_setting_create(yml_params["ckpts_dir"], yml_params["mark"])
stat_dir    = record_dir_setting_create(yml_params["stats_dir"], yml_params["mark"])
img_dir     = record_dir_setting_create(yml_params["img_dir"], yml_params["mark"])
log_dir     = record_dir_setting_create(yml_params["log_dir"], yml_params["mark"])

trainFolder = ChloraData(root_dir, frames, shape_scale, is_month = is_month, mask_dir = mask_dir)
validFolder = ChloraData(root_dir, frames, shape_scale, is_month = is_month, mask_dir = mask_dir)

trainLoader = DataLoader(trainFolder, batch_size=BATCH, shuffle=False)
validLoader = DataLoader(validFolder, batch_size=BATCH, shuffle=False)

def main_gpu(ckpt_files=None,fconv=False, min_cache=False):
    torch.cuda.manual_seed(random_seed)
    # auto select best algorithm for speeding up
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    encoder = Encoder(convlstm_scs_encoder_params[0], convlstm_scs_encoder_params[1]).cuda()
    decoder = Decoder(convlstm_scs_decoder_params[0], convlstm_scs_decoder_params[1]).cuda()
    
    if fconv:
        encoder = Encoder(fconvlstm_scs_encoder_params[0], fconvlstm_scs_encoder_params[1]).cuda()
        decoder = Decoder(fconvlstm_scs_decoder_params[0], fconvlstm_scs_decoder_params[1]).cuda()
        if min_cache:
            from model.net_structor import generate_conv_lstm_encoder_decoder_params
            encoder_params, decoder_params = generate_conv_lstm_encoder_decoder_params(frames_len=frames, is_cuda=True)
            encoder = Encoder(encoder_params[0], encoder_params[1]).cuda()
            decoder = Decoder(decoder_params[0], decoder_params[1]).cuda()
    
    model = ED(encoder, decoder)
    
    tb = SummaryWriter(log_dir)
    
    early_stopping = EarlyStopping(
        patience=20, 
        verbose=True)
    
    device = torch.device("cuda:0")
    model.to(device)
    
    
    if ckpt_files != 'None':        
        print('==> loading existing model')
        model_info = torch.load(ckpt_files)
        model.load_state_dict(model_info['state_dict'])
        optimizer = torch.optim.Adam(model.parameters())
        optimizer.load_state_dict(model_info['optimizer'])
        cur_epoch = model_info['epoch'] + 1
    else:
        cur_epoch = 0
        
    ssmi_loss_func = pytorch_ssmi.SSIM().cuda()
    mae_loss_func = nn.SmoothL1Loss().cuda()
    mse_loss_func = nn.MSELoss().cuda()
    
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    pla_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=14, verbose=True)
    
    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []
    
    mae_losses = []
    mse_losses = []
    ssmi_losses = []
    
    train_global_times = 0
    valid_glabal_times = 0
    
    for epoch in range(cur_epoch, EPOCH + 1):
        # train #
        model.train()
        img_index = np.random.randint(0, len(trainLoader))
        t = tqdm(trainLoader, leave=False, total=len(trainLoader))
        for i, (index, inputs, targets,mask) in enumerate(t):
            
            X = inputs.to(device)
            Y = targets.to(device)
            # Y = targets[0,-1,0,:,:].to(device)
            
            assert ~torch.any(torch.isnan(X)), "there is nan in X"
            assert ~torch.any(torch.isnan(Y)), "there is nan in Y"
            
            optimizer.zero_grad()
            pred:torch.tensor = model(X)
            
            assert ~torch.any(torch.isnan(pred)), "there is nan in Y"

            # temp_pred_for_pixel_wise_loss = torch.where(Y==0,torch.tensor(0).cuda(),pred)
            # mae_loss = mae_loss_func(temp_pred_for_pixel_wise_loss, Y)
            # mse_loss = mse_loss_func(temp_pred_for_pixel_wise_loss, Y)
            # ssmi_loss = ssmi_loss_func(pred.unsqueeze(0), Y.unsqueeze(0))
            mae_loss = mae_loss_func(pred, Y)
            mse_loss = mse_loss_func(pred, Y)
            ssmi_loss = ssmi_loss_func(pred[:,:,0,:,:], Y[:,:,0,:,:])

            mse_losses.append(mse_loss.item() / BATCH)
            mae_losses.append(mae_loss.item() / BATCH)
            ssmi_losses.append(ssmi_loss.item() / BATCH)

            assert ((1-ssmi_loss) >= 0) & ((1-ssmi_loss)/2 <= 1) 
            
            loss = beta * (mse_loss) + (1 - beta) * ((1-ssmi_loss)/2) 

            loss_aver = loss.item() / BATCH
            loss.backward()
            
            train_losses.append(loss_aver)
            
            clip_grad_value_(model.parameters(), clip_value=10.0)
            
            optimizer.step()
            
            t.set_postfix({
                'trainloss': '{:.6f}'.format(loss_aver),
                'epoch': '{:02d}'.format(epoch)
            })
            
            # draw_train_array(tb, inputs, pred, targets, train_global_times, "train")
            train_global_times += 1
            
            if index == img_index and epoch % 10 == 0:
                np.save(f"{img_dir}/train_{epoch}_{img_index}.npy", inputs.numpy())
                np.save(f"{img_dir}/label_{epoch}_{img_index}.npy", targets.numpy())
                if pred.is_cuda:
                    y_hat = pred.cpu().detach().numpy()
                else:
                    y_hat = pred.detach().numpy()
                np.save(f"{img_dir}/pred_{epoch}_{img_index}.npy", y_hat)
                torch.cuda.empty_cache()
    
        tb.add_scalar('TrainLoss', loss_aver, epoch)
        
        with open(f"{stat_dir}/mse_losses.txt", 'wt') as f:
            for i in mse_losses:
                print(i, file=f)
        with open(f"{stat_dir}/mae_losses.txt", 'wt') as f:
            for i in mae_losses:
                print(i, file=f)
        with open(f"{stat_dir}/ssmi_losses.txt", 'wt') as f:
            for i in ssmi_losses:
                print(i, file=f)
        if epoch % 10 == 0 and epoch != 0:
            with torch.no_grad():
                model.eval()
                t = tqdm(validLoader, leave=False, total=len(validLoader))
                for i, (index, inputs, targets, mask) in enumerate(t):

                    X = inputs.to(device)
                    Y = targets.to(device)
                    
                    optimizer.zero_grad()
                    pred:torch.tensor = model(X)

                    # temp_pred_for_pixel_wise_loss = torch.where(Y==0,torch.tensor(0).cuda(),pred)
                    # mae_loss = mae_loss_func(temp_pred_for_pixel_wise_loss, Y)
                    # mse_loss = mse_loss_func(temp_pred_for_pixel_wise_loss, Y)
                    # ssmi_loss = ssmi_loss_func(pred.unsqueeze(0), Y.unsqueeze(0))

                    mae_loss = mae_loss_func(pred, Y)
                    mse_loss = mse_loss_func(pred, Y)
                    ssmi_loss = ssmi_loss_func(pred[:,:,0,:,:], Y[:,:,0,:,:])

                    mse_losses.append(mse_loss.item() / BATCH)
                    mae_losses.append(mae_loss.item() / BATCH)
                    ssmi_losses.append(ssmi_loss.item() / BATCH)

                    assert ((1-ssmi_loss) >= 0) & ((1-ssmi_loss) <= 2) 
                    
                    loss = beta * (mse_loss) + (1 - beta) * (1-ssmi_loss) 
                    loss_aver = loss.item() / BATCH
                    
                    valid_losses.append(loss_aver)
                    
                    t.set_postfix({
                        'validloss': '{:.6f}'.format(loss_aver),
                        'epoch': '{:02d}'.format(epoch)
                    })
                    
                    # draw_train_array(tb, inputs, pred, targets, valid_glabal_times, "valid")
                    valid_glabal_times += 1
                    torch.cuda.empty_cache()
            
                tb.add_scalar('ValidLoss', loss_aver, epoch)
                
                train_loss = np.average(train_losses)
                valid_loss = np.average(valid_losses)
                avg_train_losses.append(train_loss)
                avg_valid_losses.append(valid_loss)

                epoch_len = len(str(EPOCH))

                print_msg = (f'[{epoch:>{epoch_len}}/{EPOCH:>{epoch_len}}] ' +
                            f'train_loss: {train_loss:.6f} ' +
                            f'valid_loss: {valid_loss:.6f}')

                print(print_msg)
                train_losses = []
                valid_losses = []
                pla_lr_scheduler.step(valid_loss)
                model_dict = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                early_stopping(valid_loss.item(), model_dict, epoch, save_dir)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
    with open("avg_train_losses.txt", 'wt') as f:
        for i in avg_train_losses:
            print(i, file=f)

    with open("avg_valid_losses.txt", 'wt') as f:
        for i in avg_valid_losses:
            print(i, file=f)
    
def main_cpu(ckpt_files=None,fconv=False):
    
    encoder = Encoder(convlstm_scs_encoder_params[0], convlstm_scs_encoder_params[1])
    decoder = Decoder(convlstm_scs_decoder_params[0], convlstm_scs_decoder_params[1])
    
    if fconv:
        encoder = Encoder(fconvlstm_scs_encoder_params[0], fconvlstm_scs_encoder_params[1])
        decoder = Decoder(fconvlstm_scs_decoder_params[0], fconvlstm_scs_decoder_params[1])
        if min_cache:
            encoder = Encoder(fconvlstm_scs_min_cache_encoder_params[0], fconvlstm_scs_min_cache_encoder_params[1])
            decoder = Decoder(fconvlstm_scs_min_cache_decoder_params[0], fconvlstm_scs_min_cache_decoder_params[1])
            
        
    
    model = ED(encoder, decoder)
    
    tb = SummaryWriter(log_dir)
    early_stopping = EarlyStopping(
        patience=20, verbose=True)
    
    device = torch.device("cpu")
    model.to(device)
    
    if os.path.exists(ckpt_files):        
        print('==> loading existing model')
        model_info = torch.load(ckpt_files)
        model.load_state_dict(model_info['state_dict'])
        optimizer = torch.optim.Adam(model.parameters())
        optimizer.load_state_dict(model_info['optimizer'])
        cur_epoch = model_info['epoch'] + 1
    else:
        cur_epoch = 0   
    
    ssmi_loss_func = pytorch_ssmi.SSIM()
    mae_loss_func = nn.SmoothL1Loss()
    mse_loss_func = nn.MSELoss()
    
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    pla_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4, verbose=True)
    
    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []
    
    mse_losses = []
    mae_losses = []
    ssmi_losses = []
    
    train_global_times = 0
    valid_glabal_times = 0
    
    for epoch in range(cur_epoch, EPOCH + 1):
        # train the model #
        img_index = np.random.randint(0, len(trainLoader))
        t = tqdm(trainLoader, leave=False, total=len(trainLoader))
        for i, (index, inputs, targets, mask) in enumerate(t):
            
            X = inputs.to(device)
            Y = targets.to(device)
            
            assert ~torch.any(torch.isnan(X)), "there is nan in X"
            assert ~torch.any(torch.isnan(Y)), "there is nan in Y"

            optimizer.zero_grad()
            model.train()
            pred = model(X)
            
            assert ~torch.any(torch.isnan(pred)), "there is nan in Y"
            
            mae_loss = mae_loss_func(pred, Y)
            ssmi_loss = ssmi_loss_func(pred[:,:,0,:,:], Y[:,:,0,:,:])
            mse_loss = mse_loss_func(pred, Y)
            
            mse_losses.append(mse_loss.item() / BATCH)
            mae_losses.append(mae_loss.item() / BATCH)
            ssmi_losses.append(ssmi_loss.item() / BATCH)
            
            assert ((1-ssmi_loss) >= 0) & ((1-ssmi_loss) <= 2) 
            loss = mae_loss - ssmi_loss + 1
            
            loss_aver = loss.item() / BATCH
            loss.backward()
            
            train_losses.append(loss_aver)
            
            clip_grad_value_(model.parameters(), clip_value=10.0)
            
            optimizer.step()
            
            t.set_postfix({
                'trainloss': '{:.6f}'.format(loss_aver),
                'epoch': '{:02d}'.format(epoch)
            })
            
            draw_train_array(tb, inputs, pred, targets, train_global_times, "train")
            train_global_times += 1
            
            if index == img_index:
                np.save(f"{img_dir}/train__{epoch}_{img_index}.npy", X.numpy())
                np.save(f"{img_dir}/label__{epoch}_{img_index}.npy", Y.numpy())
                np.save(f"{img_dir}/pred__{epoch}_{img_index}.npy", pred.detach().numpy())
    
        tb.add_scalar('TrainLoss', loss_aver, epoch)
        
        with open(f"{stat_dir}/mse_losses.txt", 'wt') as f:
            for i in mse_losses:
                print(i, file=f)
        with open(f"{stat_dir}/mae_losses.txt", 'wt') as f:
            for i in mae_losses:
                print(i, file=f)
        with open(f"{stat_dir}/ssmi_losses.txt", 'wt') as f:
            for i in ssmi_losses:
                print(i, file=f)
        
        with torch.no_grad():
            model.eval()
            t = tqdm(validLoader, leave=False, total=len(validLoader))
            for i, (index, inputs, targets, mask) in enumerate(t):
                X = inputs.to(device)
                Y = targets.to(device)
                
                optimizer.zero_grad()
                pred = model(X)
                
                mae_loss = mae_loss_func(pred, Y)
                ssmi_loss = ssmi_loss_func(pred[:,:,0,:,:], Y[:,:,0,:,:])
                
                assert ((1-ssmi_loss) >= 0) & ((1-ssmi_loss) <= 2) 
                
                loss = mae_loss - ssmi_loss + 1
                loss_aver = loss.item() / BATCH
                
                valid_losses.append(loss_aver)
                
                t.set_postfix({
                    'validloss': '{:.6f}'.format(loss_aver),
                    'epoch': '{:02d}'.format(epoch)
                })
                
                draw_train_array(tb, inputs, pred, targets, valid_glabal_times, "valid")
                valid_glabal_times += 1
        
            tb.add_scalar('ValidLoss', loss_aver, epoch)
            torch.cuda.empty_cache()
            
            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)

            epoch_len = len(str(EPOCH))

            print_msg = (f'[{epoch:>{epoch_len}}/{EPOCH:>{epoch_len}}] ' +
                        f'train_loss: {train_loss:.6f} ' +
                        f'valid_loss: {valid_loss:.6f}')

            print(print_msg)
            train_losses = []
            valid_losses = []
            pla_lr_scheduler.step(valid_loss)
            model_dict = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            early_stopping(valid_loss.item(), model_dict, epoch, save_dir)
            if early_stopping.early_stop:
                print("Early stopping")
                break
    with open("avg_train_losses.txt", 'wt') as f:
        for i in avg_train_losses:
            print(i, file=f)

    with open("avg_valid_losses.txt", 'wt') as f:
        for i in avg_valid_losses:
            print(i, file=f)

if __name__ == "__main__":
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    if device == 'cpu':
        main_cpu(ckpt_files,fconv)
    elif device == 'gpu':
        if torch.cuda.is_available():
            main_gpu(ckpt_files,fconv,min_cache)
        else:
            main_cpu(ckpt_files,fconv,min_cache)