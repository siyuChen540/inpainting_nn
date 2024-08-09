import os
import yaml

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.chlora import ChloraData
from model.encoder import Encoder
from model.decoder import Decoder
from model.ed import ED
from model.net_structor import convlstm_scs_encoder_params, convlstm_scs_decoder_params
from model.net_structor import fconvlstm_scs_encoder_params, fconvlstm_scs_decoder_params
from model.net_structor import fconvlstm_scs_min_cache_encoder_params, fconvlstm_scs_min_cache_decoder_params
from utils import record_dir_setting_create

yaml_dir = r"/root/workspace/src/record/param/20240717.yml"
with open(yaml_dir, "r") as aml:
    yml_params = yaml.safe_load(aml)

BATCH = yml_params["BATCH"]
frames = yml_params["frames"]
shape_scale = yml_params["shape_scale"]
random_seed = yml_params["random_seed"]
root_dir = yml_params["root_dir"]
mask_dir = yml_params["mask_dir"]
is_month = yml_params["is_month"]
device = yml_params["device"]
ckpt_files = yml_params["ckpt_files"]

try:
    fconv = yml_params["fconv"]
    min_cache = yml_params["min_cache"]
except:
    fconv = False
    min_cache = False

output_dir = record_dir_setting_create(yml_params["output_dir"], yml_params["mark"])

predictFolder = ChloraData(root_dir, frames, shape_scale, is_month=is_month, mask_dir=mask_dir)
predictLoader = DataLoader(predictFolder, batch_size=BATCH, shuffle=False)

def apply_model(ckpt_files=None, fconv=False, min_cache=False):
    torch.cuda.manual_seed(random_seed)
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
    device = torch.device("cuda:0")
    model.to(device)

    if ckpt_files != 'None':
        print('==> loading existing model')
        model_info = torch.load(ckpt_files)
        model.load_state_dict(model_info['state_dict'])

    model.eval()
    t = tqdm(predictLoader, leave=False, total=len(predictLoader))
    with torch.no_grad():
        for i, (index, inputs, targets, mask) in enumerate(t):
            X = targets.to(device)
            pred = model(X)
            
            for j in range(targets.size(1)):
                file_path = predictFolder.chunk_list[index[0]][j].split('/')[-1]
                current_root = predictFolder.chunk_list[index[0]][0].split('/')[-1].replace('.npy', '')
                output_file_path = os.path.join(output_dir, file_path).replace(".npy", f"_{current_root}_pred.npy")
                if os.path.exists(output_file_path):
                    continue
                if pred.is_cuda:
                    y_hat = pred[0,j,0].cpu().numpy()
                else:
                    y_hat = pred[0,j,0].numpy()
                np.save(output_file_path, y_hat)

            t.set_postfix({
                'Processed': '{}/{}'.format(i + 1, len(predictLoader))
            })

if __name__ == "__main__":
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    if device == 'gpu' and torch.cuda.is_available():
        apply_model(ckpt_files, fconv, min_cache)
    else:
        print("GPU not available. Please run the application on a machine with GPU.")
