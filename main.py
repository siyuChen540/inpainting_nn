import os
import yaml
import time
from collections import OrderedDict

import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import neptune
from neptune_pytorch import NeptuneLogger
from neptune.utils import stringify_unsupported
from neptune.types import File

from dataset import inpainting_DS, inpainting_DS_v2
from module import ConvGRU_cell, ConvGRU_cell_v2,ED, Encoder, Decoder
from utils import SSIM,record_dir_setting_create, EarlyStopping, random_seed_set

load_dotenv()

run = neptune.init_run(
    api_token=os.getenv("NEPTUNE_API_TOKEN"),
    project=os.getenv("NEPTUNE_PROJECT"),
    tags= ["test", "ftc-lstm"],
    capture_stdout=True,             # Enable capturing stdout
    capture_stderr=True,             # Enable capturing stderr
    capture_traceback=True,          # Enable capturing traceback
    capture_hardware_metrics=True    # Enable capturing hardware metrics  
)
run["notes"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
run["algorithm"] = "Fourier Transform Convolutional Long Short-Term Memory (FTC-LSTM)"

def trans_log_to_0_255(img):
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min())
    img = (img * 255).astype(np.uint8)
    return img


def main(yaml_dir):
    
    with open(yaml_dir, "r") as aml:
        yml_params:dict = yaml.safe_load(aml)

    # Extract parameters
    EPOCH       = yml_params["EPOCH"]
    BATCH       = yml_params["BATCH"]
    LR          = yml_params["LR"]
    frames      = yml_params["frames"]
    shape_scale = yml_params["shape_scale"]
    channels    = yml_params["channel_scale"]

    random_seed = yml_params["random_seed"]
    root_dir    = yml_params["root_dir"]
    mask_dir    = yml_params.get("mask_dir", None)
    is_month    = yml_params.get("is_month", False)
    device      = yml_params["device"]
    ckpt_files  = yml_params.get("ckpt_files",None)
    inputChunkJSON = yml_params.get("inputChunkJSON", None)

    beta        = yml_params.get("beta", 0.5)
    fconv       = yml_params.get("fconv", False)

    save_dir    = record_dir_setting_create(yml_params["ckpts_dir"], yml_params["mark"])

    random_seed_set(random_seed)

    # Prepare datasets and loaders
    train_dataset = inpainting_DS_v2(root=root_dir, inputChunkJSON_dir=inputChunkJSON, frames=frames, shape_scale=shape_scale, is_mask=is_month, mask_root=mask_dir, is_train=True)
    valid_dataset = inpainting_DS_v2(root=root_dir, inputChunkJSON_dir=inputChunkJSON, frames=frames, shape_scale=shape_scale, is_mask=is_month, mask_root=mask_dir, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)

    # Define model parameters with reduced channels and features_num    
    c_1, c_2, c_3 = channels            # [4, 8, 16]
    shapes_w, shapes_h = shape_scale    # [480, 480]
    
    # Optimized Encoder and Decoder Parameters
    encoder_params = [
        [
            OrderedDict({'conv1_leaky_1': [1,   c_1, 3, 1, 1]}),
            OrderedDict({'conv2_leaky_1': [c_2, c_2, 3, 2, 1]}),
            OrderedDict({'conv3_leaky_1': [c_3, c_3, 3, 2, 1]}),
        ],
        [   
            ConvGRU_cell(
                shape=(shapes_w,   shapes_h),    channels=c_1, 
                kernel_size=3, features_num=c_2, fconv=fconv, 
                frames_len=frames, device=device),
            ConvGRU_cell(
                shape=(shapes_w//2,shapes_h//2), channels=c_2, 
                kernel_size=3, features_num=c_3, fconv=fconv, 
                frames_len=frames, device=device),
            ConvGRU_cell(
                shape=(shapes_w//4,shapes_h//4), channels=c_3, 
                kernel_size=3, features_num=c_3, fconv=fconv, 
                frames_len=frames, device=device)
        ]
    ]

    decoder_params = [
        [
            OrderedDict({'deconv1_leaky_1': [c_3, c_3, 4, 2, 1]}),
            OrderedDict({'deconv2_leaky_1': [c_3, c_3, 4, 2, 1]}),
            OrderedDict({                                       
                'conv3_leaky_1': [c_2, c_1, 3, 1, 1],
                'conv4_leaky_1': [c_1,   1, 1, 1, 0]
            }),
        ],
        [
            ConvGRU_cell(
                shape=(shapes_w//4,shapes_h//4), channels=c_3, 
                kernel_size=3, features_num=c_3, fconv=fconv, 
                frames_len=frames, device=device),
            ConvGRU_cell(
                shape=(shapes_w//2,shapes_h//2), channels=c_3, 
                kernel_size=3, features_num=c_3, fconv=fconv, 
                frames_len=frames, device=device),
            ConvGRU_cell(
                shape=(shapes_w,   shapes_h),    channels=c_3, 
                kernel_size=3, features_num=c_2, fconv=fconv, 
                frames_len=frames, device=device),
        ]
    ]

    # Initialize Encoder and Decoder
    encoder = Encoder(encoder_params[0], encoder_params[1]).to(device)
    decoder = Decoder(decoder_params[0], decoder_params[1]).to(device)

    # Initialize the ED model
    model = ED(encoder, decoder).to(device)
    
    # Initialize neptune logger
    npt_logger = NeptuneLogger(
        run,
        model=model,  # Model will be set after initialization
        log_model_diagram=True,
        log_gradients=True,
        log_parameters=True,
        log_freq=30,
    )
    run[npt_logger.base_namespace]["hyperparameters"] = stringify_unsupported(yml_params)

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Load model checkpoint if provided
    current_epoch = 0
    if ckpt_files not in [None, 'None', '']:
        print('==> Loading existing model from checkpoint')
        checkpoint = torch.load(ckpt_files, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        current_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {current_epoch}")

    # Define loss functions
    ssmi_loss_func = SSIM().to(device)
    mae_loss_func  = nn.SmoothL1Loss().to(device)
    mse_loss_func  = nn.MSELoss().to(device)

    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=14, verbose=True)

    # Initialize Early Stopping
    early_stopping = EarlyStopping(patience=20, verbose=True)

    # Initialize mixed precision scaler
    scaler = torch.amp.GradScaler()

    # Training Loop
    for epoch in range(current_epoch, EPOCH+1):
        model.train()
        t = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch}")
        # select a random index for each batch to append predictions to neptune
        idx_random = np.random.randint(0, len(train_loader), 3)
        for idx, inputs, targets in t:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if torch.all(torch.isnan(inputs)) or torch.all(torch.isnan(targets)):
                continue
            optimizer.zero_grad()
            # Determine device_type for autocast
            device_type = 'cuda' if device == 'cuda' else 'cpu'
            with torch.amp.autocast(device_type):
                pred = model(inputs)
                
                # Compute losses
                mae_loss = mae_loss_func(pred, targets)
                mse_loss = mse_loss_func(pred, targets)
                ssmi_loss = ssmi_loss_func(pred[:, :, 0:1, :, :], targets[:, :, 0:1, :, :])  # Assuming first channel is relevant

                # loss = beta * mse_loss + (1 - beta) * ((1 - ssmi_loss)/2)
                loss = mae_loss
                
                if torch.isnan(loss):
                    print("Loss is NaN")
                    print("Inputs is NaN:", torch.any(torch.isnan(inputs)))
                    print("Targets is NaN:", torch.any(torch.isnan(targets)))
                    print("Pred is NaN:", torch.any(torch.isnan(pred)))
                    for dm_i in range(frames):
                        # save the input, target, and pred as npy files for debugging
                        np.save(os.path.join(save_dir, f"input_{idx.item()}_{dm_i}.npy"), inputs[0,dm_i,0].detach().cpu().numpy())
                        np.save(os.path.join(save_dir, f"target_{idx.item()}_{dm_i}.npy"), targets[0,dm_i,0].detach().cpu().numpy())
                        np.save(os.path.join(save_dir, f"pred_{idx.item()}_{dm_i}.npy"), pred[0,dm_i,0].detach().cpu().numpy())
                        run[npt_logger.base_namespace]["nan/pred"].append(File.as_image(trans_log_to_0_255(pred[0,dm_i,0].detach().cpu().numpy())))
                        run[npt_logger.base_namespace]["nan/target"].append(File.as_image(trans_log_to_0_255(targets[0,dm_i,0].detach().cpu().numpy())))
                        run[npt_logger.base_namespace]["nan/input"].append(File.as_image(trans_log_to_0_255(inputs[0,dm_i,0].detach().cpu().numpy())))
                    continue
            # Backward pass with mixed precision
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()

            t.set_postfix(loss=loss.item())
        run[npt_logger.base_namespace]["each_EPOCH/loss"].append(loss.item())
        run[npt_logger.base_namespace]["each_EPOCH/mae_loss"].append(mae_loss.item())
        run[npt_logger.base_namespace]["each_EPOCH/mse_loss"].append(mse_loss.item())
        run[npt_logger.base_namespace]["each_EPOCH/ssmi_loss"].append(ssmi_loss.item())
        # Validation every few epochs (e.g., every 10 epochs)
        if epoch % 10 == 0 and epoch != 0:
            model.eval()
            valid_losses = []
            with torch.no_grad():
                for idx, inputs, targets in tqdm(valid_loader, desc=f"Validation Epoch {epoch}", leave=False):
                    inputs = inputs.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    
                    device_type = 'cuda' if device == 'cuda' else 'cpu'
                    with torch.amp.autocast(device_type):
                        pred = model(inputs)

                        mae_loss = mae_loss_func(pred, targets)
                        mse_loss = mse_loss_func(pred, targets)
                        ssmi_loss = ssmi_loss_func(pred[:, :, 0:1, :, :], targets[:, :, 0:1, :, :])

                        loss = beta * mse_loss + (1 - beta) * ((1 - ssmi_loss)/2)
                        valid_losses.append(loss.item())

            mean_valid_loss = np.mean(valid_losses)
            run[npt_logger.base_namespace]["epoch/train_loss"].log(mean_valid_loss)
            # Step the scheduler
            scheduler.step(mean_valid_loss)

            # Prepare model state for checkpointing
            model_state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }

            # Early Stopping Check
            early_stopping(mean_valid_loss, model_state, epoch, save_dir)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

            # Log to neptune the model checkpoint as an artifact
            # run[npt_logger.base_namespace]["model/checkpoint"].log(File.as_image(os.path.join(save_dir, f"checkpoint_{epoch}_{mean_valid_loss:.6f}.pth.tar")))

    # Save final model
    torch.save(model.state_dict(), os.path.join(save_dir, "final_model.pth"))
    # run[npt_logger.base_namespace]["model/final_model"].append(os.path.join(save_dir, "final_model.pth"))
    run.stop()

if __name__ == "__main__":
    yaml_dir = r"/root/params/1214.yml"
    main(yaml_dir)
