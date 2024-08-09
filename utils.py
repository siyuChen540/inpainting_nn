# -*- encoding: utf-8 -*-
'''
@File        :  utils.py
@Time        :  2022/12/17 10:24:00
@Author      :  chen siyu
@Mail        :  chensiyu57@mail2.sysu.edu.cn
@Version     :  1.0
@Description :  utils
'''

import os
import numpy as np
import torch

from torch import nn
from collections import OrderedDict
from tensorboardX import SummaryWriter
from model.pconv import PartialConv2d


def mkLayer(block):
    layers = []
    for layer_name, params in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(
                kernel_size = params[0], 
                stride = params[1], 
                padding = params[2])
            layers.append((layer_name, layer))
        elif 'deconv' in layer_name:
            transposeConv2d = nn.ConvTranspose2d(
                in_channels  = params[0],
                out_channels = params[1],
                kernel_size = params[2],
                stride = params[3],
                padding = params[4])
            layers.append((layer_name, transposeConv2d))
            
            if 'relu' in layer_name:
                layers.append((
                    'relu_' + layer_name, 
                    nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append((
                    'leaky_' + layer_name,
                    nn.LeakyReLU(
                       negative_slope=0.2, 
                       inplace=True)))

        elif 'conv' in layer_name:
            if 'pconv' in layer_name:
                conv2d = PartialConv2d(
                    in_channels = params[0],
                    out_channels = params[1],
                    kernel_size = params[2],
                    stride = params[3],
                    padding = params[4])
            else:
                conv2d = nn.Conv2d(
                    in_channels = params[0],
                    out_channels = params[1],
                    kernel_size = params[2],
                    stride = params[3],
                    padding = params[4])
                
            layers.append((layer_name, conv2d))
            
            if 'relu' in layer_name:
                layers.append((
                    'relu_' + layer_name, 
                    nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name,
                               nn.LeakyReLU(
                                   negative_slope=0.2, 
                                   inplace=True)))
        else:
            raise NotImplementedError
        
    return nn.Sequential(OrderedDict(layers))

def draw_train_array(tb:SummaryWriter,inputs,pred,labels,global_times, state):
    batch,frame,channel,h,w = inputs.size()
    x = inputs.numpy()
    y = labels.numpy()
    
    if pred.is_cuda:
        y_hat = pred.cpu().detach().numpy()
    else:
        y_hat = pred.detach().numpy()
    
    for b in range(batch):
        for f in range(frame):
            tb.add_image(f"{state}_input/{b}/{f}", x[b,f], global_times)
            tb.add_image(f"{state}_label/{b}/{f}", y[b,f], global_times)
            tb.add_image(f"{state}_predict/{b}/{f}", y_hat[b,f], global_times)
            
def record_dir_setting_create(father_dir:str, mark:str) -> str:
    dir_temp:str = os.path.join(father_dir,mark)
    assert os.path.split(dir_temp)[-1] == mark, "fail join folder"
    
    if not os.path.exists(dir_temp):
        os.makedirs(dir_temp)
    
    return dir_temp

def modelsize(model, input, type_size=4):
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))

    input_ = input.clone()
    input_.requires_grad_(requires_grad=False)

    mods = list(model.modules())
    out_sizes = []

    for i in range(1, len(mods)):
        m = mods[i]
        if isinstance(m, nn.ReLU):
            if m.inplace:
                continue
        out = m(input_)
        for o in out:
            for os in o:
                out_sizes.append(np.array(os.size()))
        input_ = out

    total_nums = 0
    for i in range(len(out_sizes)):
        s = out_sizes[i]
        nums = np.prod(np.array(s))
        total_nums += nums


    print('Model {} : intermedite variables: {:3f} M (without backward)'
          .format(model._get_name(), total_nums * type_size / 1000 / 1000))
    print('Model {} : intermedite variables: {:3f} M (with backward)'
          .format(model._get_name(), total_nums * type_size*2 / 1000 / 1000))

def cal_stat(array:np.ndarray,name:str):
    stat_list = []
    b_name = ["input","raw","predict"]
    if array.ndim ==5:
        b, t, c, h, w = array.shape
        for b_ in range(b):
            for i in range(t):
                mean = np.mean(array[b_,i,0,:,:])
                std = np.std(array[b_,i,0,:,:])
                stat_list.append({
                    "file_name":name,
                    "mean":mean,
                    "std":std
                })
    elif array.ndim == 4:
        b, t, h, w = array.shape
        for b_ in range(b):
            for i in range(t):
                mean = np.mean(array[b_,i,:,:])
                std = np.std(array[b_,i,:,:])
                stat_list.append({
                    "file_name":name,
                    "mean":mean,
                    "std":std
                })
        for i in range(b):
            mean = np.mean(array[i,0,:,:])
            std = np.std(array[i,0,:,:])
            stat_list.append({
                "file_name":name,
                "mean":mean,
                "std":std
            })
    elif array.ndim == 3:
        t, h, w = array.shape
        for i in range(t):
            mean = np.mean(array[i,:,:])
            std = np.std(array[i,:,:])
            stat_list.append({
                "mean":mean,
                "std":std
            })
    
    return stat_list

def cal_region_stat(array:np.ndarray, ll_bbox:list, ll_extent:list=[105,125,5,25]):
    extent_lon_min, extent_lon_max, extent_lat_min, extent_lat_max = ll_extent
    lon_min, lat_min, lon_max, lat_max = ll_bbox
    
    lon = np.linspace(extent_lon_min, extent_lon_max, array.shape[-2])
    lat = np.linspace(extent_lat_min, extent_lat_max, array.shape[-1])
    
    lon_index = np.where((lon>=lon_min) & (lon<=lon_max))[0]
    lat_index = np.where((lat>=lat_min) & (lat<=lat_max))[0]
    
    array_index = np.ix_(lon_index, lat_index)

    region_array = array[array_index]
    
    stat_list = cal_stat(region_array)
    return stat_list

if __name__ == "__main__":
    
    father_dir = "/home/chensiyu/workspace/02_scientist_program/02_img_recovery/record/ckpts"
    mark = "1"
    dir = record_dir_setting_create(father_dir, mark)
    print(dir)
