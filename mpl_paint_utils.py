import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from typing import Tuple
import cartopy.crs as ccrs
import cartopy.feature as cfeat
import latexify
from scipy.signal import convolve2d
@latexify.with_latex
def RMSE(pred:np.ndarray,label:np.ndarray) -> np.ndarray:
    pred_copy = np.copy(pred)
    label_copy = np.copy(label)
    
    pred_copy[np.isnan(pred_copy)] = 0
    label_copy[np.isnan(label_copy)] = 0
    return np.sqrt(((pred_copy-label_copy)**2).mean())

@latexify.with_latex
def R_2(pred:np.ndarray,label:np.ndarray)-> float:
    pred_copy = np.copy(pred)
    label_copy = np.copy(label)
    
    pred_copy[np.isnan(pred_copy)] = 0
    label_copy[np.isnan(label_copy)] = 0
    
    res = ((pred_copy-label_copy)**2).sum()
    tot = ((label_copy-label_copy.mean())**2).sum()
    return 1- (res/tot) 


def gaussian(window_size, sigma):
    gauss = np.array([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).reshape(window_size, 1)
    _2D_window = np.outer(_1D_window, _1D_window.transpose())
    window = np.tile(_2D_window, (channel, 1, 1, 1))
    return window

def ssim(img_1, img_2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    img1 = np.copy(img_1)
    img2 = np.copy(img_2)
    img1[np.isnan(img1)] = 0
    img2[np.isnan(img2)] = 0
    
    if val_range is None:
        if np.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if np.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 'same'
    (channel, height, width) = img1.shape
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel)
    img1 = img1[0]
    img2 = img2[0]
    window=window[0,0]
    
    mu1 = convolve2d(img1, window, mode='same', boundary='symm')
    mu2 = convolve2d(img2, window, mode='same', boundary='symm')

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = convolve2d(img1**2, window, mode='same', boundary='symm') - mu1_sq
    sigma2_sq = convolve2d(img2**2, window, mode='same', boundary='symm') - mu2_sq
    sigma12 = convolve2d(img1*img2, window, mode='same', boundary='symm') - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = v1 / v2

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        cs = np.mean(cs)
        ret = np.mean(ssim_map)
    else:
        cs = np.mean(cs, axis=(1, 2, 3))
        ret = np.mean(ssim_map, axis=(1, 2, 3))

    if full:
        return ret, cs
    return ret

def log_prob_density(x, y, sigma=None):
    if sigma is None:
        sigma = np.std(x - y)
        if sigma == 0:
            sigma = np.finfo(float).eps

    return - 0.5 * np.log(2 * np.pi * sigma**2) - 0.5 * ((x - y)**2) / (sigma**2)
def probability_density(x, y, sigma=1.0):
    if sigma == 0:
        sigma = np.finfo(float).eps
    sigma = np.std(x-y)
    diff = x - y
    norm = 1.0 / np.sqrt(2 * np.pi * sigma**2)
    exp = np.exp(-0.5 * (diff / sigma)**2)
    
    return exp / (norm * len(x))
def cal_cloud_mask(array_with_cloud, land_mask):
    cloud_ratio = np.sum(np.isnan(array_with_cloud[land_mask == 0])) / (np.sum(np.isnan(array_with_cloud[land_mask == 0])) + np.sum(~np.isnan(array_with_cloud[land_mask == 0])))
    
    return np.around(100 * cloud_ratio,2)

def creat_land_mask(lon_start=105,lon_end=125,lat_start=5,lat_stop=25,resolution=480):
    from global_land_mask import globe
    lon_ = np.linspace(
        start=lon_start,
        stop=lon_end,
        num=resolution
    )
    lat_ = np.linspace(
        start=lat_stop,
        stop=lat_start,
        num=resolution
    )
    lons, lats = np.meshgrid(lon_, lat_)
    land_mask = globe.is_land(lats, lons)
    return land_mask

def generate_path(folder_date,folder_time,epoch,iters,basic_path):
    label_file = f"{folder_date}-0{folder_time}/label__{epoch}_{iters}.npy"
    label_path = basic_path + label_file
    train_path = label_path.replace("label","train")
    pred_path = label_path.replace("label","pred")
    return label_path,train_path,pred_path
def read_npy_ds(folder_date,folder_time,epoch,iters,basic_path):
    # generate landmask
    land_mask = creat_land_mask()
    # generate dataset directory
    label_path,train_path,pred_path = generate_path(folder_date,folder_time,epoch,iters,basic_path)
    # load dataset
    label:np.ndarray = np.load(label_path)
    pred:np.ndarray = np.load(pred_path)
    train:np.ndarray= np.load(train_path)
    # cloud mask removal
    train = np.where(train == 0, np.nan, train)
    label = np.where(label == 0, np.nan, label)
    # process 3-D and 5-D land mask
    if train.ndim == 3:
        train[0,land_mask] = np.nan
        label[0,land_mask] = np.nan
        pred[0,land_mask] = np.nan
    elif train.ndim > 3:
        train[0,:,0,land_mask] = np.nan
        label[0,:,0,land_mask] = np.nan
        pred[0,:,0,land_mask] = np.nan
            
    return train,label,pred,land_mask
def get_iters(path:str):
    file_name_list:list = os.listdir(path)
    contain_list:list = [[file_name.split("_")[2], file_name.split("_")[3].split(".")[0]] for file_name in file_name_list]
    unique_list:list = []
    for epoch_iter_set in contain_list:
        if epoch_iter_set in unique_list:
            pass
        else:
            unique_list.append(epoch_iter_set)
    return unique_list

def calculate_ticks(train, label, pred,default_min=0.05,default_max=8):
    # Calculate minimum and maximum values for ticks
    min_val = np.nanmax([np.nanmin(train[0]), np.nanmin(label[0]), np.nanmin(pred[0])]) + 0.5
    max_val = np.nanmin([np.nanmax(train[0]), np.nanmax(label[0]), np.nanmax(pred[0])]) - 0.5

    # Calculate step size and generate array of ticks
    step_size = (max_val - min_val) / 100
    ticks = np.array([
        np.log10(default_min),
        min_val + 15 * step_size,
        min_val + 30 * step_size,
        min_val + 45 * step_size,
        min_val + 60 * step_size,
        min_val + 75 * step_size,
        np.log10(default_max)
    ], dtype=float)
    
    return min_val,max_val,ticks

def generate_ticks_labels(train:np.ndarray,label:np.ndarray,pred:np.ndarray,default_min=0.05,default_max=8) -> Tuple[float,float,float,np.ndarray,np.ndarray]:
    min_val,max_val,ticks = calculate_ticks(train, label, pred,default_min,default_max)
    labels = np.around(np.power(10, ticks), 2)
    labels[0] = default_min
    labels[-1] = default_max
    return ticks[0], ticks[-1], ticks, labels

def create_figure() -> Tuple[matplotlib.figure.Figure, np.ndarray]:
    plt.figure()
    fig = plt.figure(dpi=600)
    ax_list = []
    for i in range(1,4):
        axes = fig.add_subplot(1,4,i,projection=ccrs.PlateCarree())
        axes.add_feature(
            cfeat.LAND,
            facecolor='grey',
            edgecolor='black',
            linewidth=0.5
        )
        ax_list.append(axes)
    ax_list.append(fig.add_subplot(1,4,4))
    return fig, ax_list

def configure_subplots(axes,lon,lat,cloud_mask_ratio,axes_index):
    axes.set_xticks(np.linspace(105,125,5),crs=ccrs.PlateCarree())
    axes.set_yticks(np.linspace(5,25,5),crs=ccrs.PlateCarree())
    if axes_index == 0:
        pass
    else:
        axes.set_yticklabels(['']*5)
    axes.tick_params(axis='both',direction='in',length=2,labelsize=5)
    axes.text(106,23,f"cloud: {cloud_mask_ratio}%", size=5)
    axes.gridlines(linestyle='--',color='gray',linewidth=0.5,alpha=0.5)
    return axes

def random_select(label_reshape:np.ndarray,pred_reshape:np.ndarray,ratio:float=0.3):
    idx = np.where(~np.isnan(label_reshape))[0]
    sampler_size = int(len(idx)*ratio)
    random_index = np.random.choice(idx,size=sampler_size,replace=False)
    
    return label_reshape[random_index],pred_reshape[random_index]

    
def show_diff_subplots(ax,im_list,label:np.ndarray,pred:np.ndarray):
    rmse = str(RMSE(label[0],pred[0]))[0:4]
    r_2 = np.round(R_2(pred[0],label[0]),2)
    ssim_ = np.round(ssim(label,pred),2)
    
    label_reshape,pred_reshape = random_select(label.reshape(-1),pred.reshape(-1),ratio=0.3)
    points_nums = int(label_reshape.shape[0])
    # diff = np.absolute(label_reshape - pred_reshape)
    
    # kde = gaussian_kde(label_reshape.T)
    # diff = kde.logpdf(pred_reshape.T)
    
    diff = points_nums * probability_density(label_reshape, pred_reshape)
    print(np.nanmin(diff))
    print(np.nanmax(diff))
    im_list.append(ax.scatter(
        label_reshape,
        pred_reshape,
        s=0.7,
        marker='o',edgecolor='none',
        facecolor=diff,c=diff,
        cmap='rainbow',
        vmin=0,
        vmax=0.12,
        zorder=10))

    value_range = np.linspace(np.nanmin(label_reshape),np.nanmax(label_reshape),6,dtype=float)
    
    ax.plot(value_range,value_range,linestyle='--',color='black',linewidth=0.7,zorder=11)
    
    ax.text(
        x = (value_range[0]+value_range[1])/2,
        y = (value_range[-1]+value_range[-2])/2,
        s = f'RMSE = {rmse}\nSSIM={ssim_}\nN={points_nums}\n'+'$\mathregular{R^2}$'+f"= {r_2}",
        ha='left',
        va='top',
        size=5
    )
    
    value_range_label = np.round(np.power(10,value_range),2)
    ax.set_xlim(value_range[0],value_range[-1])
    ax.set_ylim(value_range[0],value_range[-1])
    ax.set_xticks(value_range)
    ax.set_yticks(value_range)
    ax.set_xticklabels(value_range_label,fontsize=5)
    ax.set_yticklabels(value_range_label,fontsize=5)
    ax.tick_params(axis='both',direction='in',length=2)
    ax.grid(linewidth=0.5,alpha=0.5)
    
    return ax,im_list

def show_subplots(train, label, pred, land_mask, vmin, vmax, ax):
    cm = plt.get_cmap('rainbow')
    lon = np.linspace(105,125,5,dtype=np.uint8)
    lat = np.linspace(5,25,5,dtype=np.uint8)
    
    im_list:list = []
    
    for i, arr in enumerate([train[0], label[0], pred[0]]):
        im_list.append(ax[i].imshow(
            arr,
            origin='upper',
            transform=ccrs.PlateCarree(),
            extent=[105,125,5,25], 
            vmin=vmin, vmax=vmax, 
            cmap=cm))
        cloud_mask_ratio = cal_cloud_mask(arr,land_mask)
        ax[i] = configure_subplots(ax[i],lon,lat,cloud_mask_ratio,i)
    ax[-1],im_list = show_diff_subplots(ax[-1],im_list,label,pred)

    return im_list,ax

def set_subplot_titles(ax):
    unit = "($\mathregular{mg \ m^{-3}}$)"
    ax[0].set_title("Model Input", fontsize=6)
    ax[1].set_title("Raw Chl-a", fontsize=6)
    ax[2].set_title("Reconstructed Chl-a", fontsize=6)
    ax[0].set_xlabel("Longitude", fontsize=6)
    ax[0].set_ylabel("Latitude", fontsize=6)
    ax[1].set_xlabel("Longitude", fontsize=6)
    ax[2].set_xlabel("Longitude", fontsize=6)
    ax[3].set_ylabel("Reconstructed Chl-a"+unit, fontsize=6)
    ax[3].set_xlabel("Raw Chl-a"+unit, fontsize=6)
    return ax

def set_subplots_spines(ax):
    for h in range(len(ax)):
        for axis in ["left","right","top","bottom"]:
                ax[h].spines[axis].set_linewidth(0.45)
    return ax

def get_position(ax):
    position_list = []
    for axes in ax:
        position_list.append(axes.get_position().get_points().flatten())
    return position_list

def generate_colorbar(fig,ax,im,position_list,ticks,labels,cb_label):
    ax_cbar = fig.add_axes(position_list)
    colorbar = fig.colorbar(im, 
                            cax=ax_cbar, orientation='horizontal', 
                            ticks=ticks
                            )
    colorbar.ax.set_xticklabels(labels)
    colorbar.ax.tick_params(labelsize=5,length=2)
    if cb_label is not None:
        colorbar.set_label(
            label = cb_label,
            family = 'times new roman',
            fontsize = 5
        )
    
    return fig,ax

plt.rcParams["font.family"] = "times New Roman"
def dincae_plt(train:np.ndarray,label:np.ndarray,pred:np.ndarray,land_mask:np.ndarray,epoch:int,title:str):
    # 计算颜色条的ticks和ticklabels
    min, max, ticks, ticklabels = generate_ticks_labels(train, label, pred)
    # 创建一个带有4个子图的图形
    fig, ax = create_figure()
    # 设置子图的标题
    ax = set_subplot_titles(ax)
    # 可视化子图
    im_list,ax = show_subplots(train, label, pred, land_mask, min, max, ax)
    # 设置子图的边框
    ax = set_subplots_spines(ax)
    # plt.tight_layout()
    # 定位子图
    position_list = get_position(ax)
    ax_3_position = [
        position_list[2][2]+position_list[2][0]-position_list[1][2]+0.05,position_list[2][1],
        position_list[0][2]-position_list[0][0],
        position_list[0][3]-position_list[0][1]]
    ax[3].set_position(ax_3_position)
    position_list[3] = ax[3].get_position().get_points().flatten()
    # 创建colorbar
    cb_postion_1 = [position_list[0][0],position_list[0][1]-0.08, position_list[2][2]-position_list[0][0], 0.01]
    cb_postion_2 = [position_list[3][0],position_list[0][1]-0.08, position_list[3][2]-position_list[3][0],0.01]
    fig,ax = generate_colorbar(
        fig,ax,im_list[0],
        cb_postion_1,
        ticks,ticklabels,'Chlorophyll ($\mathregular{mg \ m^{-3}}$)')
    cb_2_ticks = np.linspace(0,0.12,5)
    cb_2_ticklabels = [0,0.05,0.1,0.15,0.2]
    fig,ax = generate_colorbar(
        fig,ax,im_list[3],
        cb_postion_2,
        ticks=cb_2_ticks,labels=cb_2_ticklabels,cb_label="Probability Density")
    fig.suptitle(title, y=0.71,fontsize=16, 
                #  horizontalalignment='center'
                x = 0.525
                )
    plt.show()
def single_plot_func():
    # from mpl_paint_utils import read_npy_ds,dincae_plt
    folder_date = 20230228
    folder_time = 943
    basic_path = "/home/chensiyu/workspace/02_scientist_program/02_img_recovery/record/img/"
    epoch_iters = get_iters("/home/chensiyu/workspace/02_scientist_program/02_img_recovery/record/img/20230228-0943")
    epoch_target = [0,5,10,30,82,110,115]
    epoch_target = [115]
    title = "DinCAE"
    for e,i in epoch_iters:
        if int(e) in epoch_target:
            try:
                train,label,pred,land_mask = read_npy_ds(folder_date,folder_time,int(e),int(i),basic_path)
                dincae_plt(train,label,pred,land_mask,e,title)
            except FileNotFoundError as error:
                print("e: ",e)
                print("i: ",i)
        else:   
            pass
    folder_date = 20230106
    folder_time = 141
    basic_path = "/home/chensiyu/workspace/02_scientist_program/02_img_recovery/record/img/"
    title = "Fourier Conv-LSTM"
    epoch = 200
    iters = 61
    train,label,pred,land_mask = read_npy_ds(folder_date,folder_time,int(epoch),int(iters),basic_path)

    for i in range(10):
        train_plot_arr = train[:,i,0,:,:]
        label_plot_arr = label[:,i,0,:,:]
        pred_plot_arr = pred[:,i,0,:,:]
        dincae_plt(train_plot_arr,label_plot_arr,pred_plot_arr,land_mask,epoch,title)
if __name__ == '__main__':
    folder_date = 20230319
    folder_time = "0754"
    epoch = 18
    iters = 34
    basic_path = "D:/ceeres/03_Program/01_SYSUM/00_science/04_image_recovery/01_program/git_repo/convlstm_en_de/record/img/20230319-0754/label_18_34.npy"
    train,label,pred,land_mask = read_npy_ds(folder_date,folder_time,epoch,iters, basic_path=basic_path)
    # 时序图像一张图
    plt.rcParams["font.family"] = "times New Roman"
    fig, ax = plt.subplots(3, 4, dpi=600, figsize=(4,3))
    for i in range(4):
        min = np.nanmax([np.nanmin(train[0,i,0]), np.nanmin(label[0,i,0]), np.nanmin(pred[0,i,0])])\
            + 0.5
        max = np.nanmin([np.nanmax(train[0,i,0]), np.nanmax(label[0,i,0]), np.nanmax(pred[0,i,0])])\
            - 0.5
        step_scale = (max - min)/100 
        cm = plt.get_cmap('rainbow')
        ticks = np.array([min, min+15*step_scale,0,min+30*step_scale,min+60*step_scale,min+85*step_scale,max], dtype=float)
        ticklabels = np.around(np.power(10, ticks), 2)
        
        ticks[0] = np.log10(0.05)
        ticks[-1] = np.log10(8)
        
        ticklabels[0] = 0.05
        ticklabels[-1] = 8
        
        min = np.log10(0.05)
        max = np.log10(8)
        
        im = ax[0,i].imshow(train[0,i,0],
                        vmin=min, vmax=max, 
                        cmap=cm)
        ax[0,i].set_xticks([])
        ax[0,i].set_yticks([])
        ax[0,i].text(10,45,f"cloud: {cal_cloud_mask(train[0,i,0],land_mask)}%", size=3.5)
        
        iL = ax[1,i].imshow(label[0,i,0],
                        vmin=min, vmax=max, 
                        cmap=cm)
        ax[1,i].set_xticks([])
        ax[1,i].set_yticks([])
        ax[1,i].text(10,45,f"cloud: {cal_cloud_mask(label[0,i,0],land_mask)}%", size=3.5)
        
        ip = ax[2,i].imshow(pred[0,i,0],
                            vmin=min, vmax=max, 
                            cmap=cm)
        ax[2,i].set_xticks([])
        ax[2,i].set_yticks([])
        ax[2,i].text(10,45,f"cloud: {cal_cloud_mask(pred[0,i,0],land_mask)}%", size=3.5)

    for w in range(10):
        for h in range(3):
            for axis in ["left","right","top","bottom"]:
                ax[h,w].spines[axis].set_linewidth(0.45)

    ax[0,0].set_ylabel("Input", fontsize=6)
    ax[1,0].set_ylabel("G T", fontsize=6)
    ax[2,0].set_ylabel("Pred", fontsize=6)

    p0 = ax[2,0].get_position().get_points().flatten()
    p2 = ax[2,9].get_position().get_points().flatten()

    ax_cbar1 = fig.add_axes([p0[0],p0[1]-0.05, p2[2]-p0[0], 0.01])
    colorbar = plt.colorbar(im, cax=ax_cbar1, orientation='horizontal', 
                            ticks=ticks
                            )
    colorbar.ax.set_xticklabels(ticklabels)
    plt.suptitle(f'Fourier Conv-LSTM performence-{epoch}', y=1.01,fontsize=16)
    # plt.suptitle(f'Conv-LSTM performence-{epoch}', y=1.01,fontsize=16)
    plt.show()