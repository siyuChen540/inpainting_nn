# -*- encoding: utf-8 -*-
'''
@File        :  nc2_scs_npy.py
@Time        :  2023/01/06 10:24:00
@Author      :  chen siyu
@Mail        :  chensiyu57@mail2.sysu.edu.cn
@Version     :  1.0
@Description :  netCDF4 satellite datasets scale
     to southern china sea region in npy dataset format
'''


import os
import numpy as np
import xarray as xr
from typing import Tuple

from tqdm import tqdm

def init_info() -> Tuple[str, str, str]:
    datasets_dir:str = '/home/chensiyu/workspace/01_main_dataset/02_reconstruct_dataset/00_chl_mon_gloabel_nc'

    result_npy_dir:str = "/home/chensiyu/workspace/01_main_dataset/02_reconstruct_dataset/02_chl_mon_scs_npy/chlora"
    
    return datasets_dir, result_npy_dir

def get_nc_files_list(datasets_dir:str) -> list:
    all_files_list:str = os.listdir(datasets_dir)
    files_list:list = [nc_file for nc_file in all_files_list if os.path.splitext(nc_file)[1] == '.nc']
    return files_list


def get_lon_lat(datasets_dir:str, files_list:list) -> np.ndarray:
    temp_nc_files = xr.open_dataset(os.path.join(datasets_dir, files_list[0]))
    lon = temp_nc_files['lon'].data
    lat = temp_nc_files['lat'].data
    return lon, lat

def generate_land_mask(lon, lat):
    pass

def logi_cut(lon_array:np.ndarray, lat_array:np.ndarray, cut_array:np.ndarray) -> np.ndarray:
    lat_boolean = (lat_array > 5.0) & (lat_array < 25.0)
    lon_boolean = (lon_array > 105.0) & (lon_array < 125.0)
    
    row_cut:np.ndarray = cut_array[lat_boolean]
    col_row_cut:np.ndarray = row_cut[:, lon_boolean]
    return col_row_cut

def generate_cloud_mask(cut_var:np.ndarray, land_mask:np.ndarray) -> np.ndarray:
    nan_mask = np.zeros(shape=cut_var.shape)
    nan_mask[~np.isnan(cut_var)] = 1
    nan_mask[land_mask==1] = 0
    
    return nan_mask
    
def main():
    datasets_dir, result_npy_dir = init_info()
    
    nc_files_list = get_nc_files_list(datasets_dir)

    lon, lat = get_lon_lat(datasets_dir, nc_files_list)
    t = tqdm(nc_files_list, total=len(nc_files_list))
    
    for file_name in t:
        npy_dir = os.path.join(result_npy_dir, file_name.replace("nc", "npy"))

        file_dir:str = os.path.join(datasets_dir, file_name)
        file = xr.open_dataset(file_dir)
        var = file["chlor_a"]

        var_cut = logi_cut(lon, lat, var)
        
        
        np.save(npy_dir, var_cut)        

if __name__ == '__main__':
    main()