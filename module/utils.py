import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any
from .layers.recurrent import *
from .models.encoder import Encoder
from .models.decoder import Decoder
from .models.ed import ED


def build_conv2d(cfg: Dict[str, Any]) -> nn.Module:
    """
    cfg example:
        {
            "type": "conv",
            "in_channels": 3,
            "out_channels": 64,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "bias": True,
            "activation": "relu"
        }
    """
    return nn.Conv2d(
        in_channels=cfg["in_channels"],
        out_channels=cfg["out_channels"],
        kernel_size=cfg.get("kernel_size", 3),
        stride=cfg.get("stride", 1),
        padding=cfg.get("padding", 0),
        bias=cfg.get("bias", True)
    )

def build_deconv2d(cfg: Dict[str, Any]) -> nn.Module:
    """
    cfg example:
        {
            "type": "deconv",
            "in_channels": 3,
            "out_channels": 64,
            "kernel_size": 4,
            "stride": 2,
            "padding": 1,
            "bias": True
        }
    """
    return nn.ConvTranspose2d(
        in_channels=cfg["in_channels"],
        out_channels=cfg["out_channels"],
        kernel_size=cfg.get("kernel_size", 4),
        stride=cfg.get("stride", 2),
        padding=cfg.get("padding", 1),
        bias=cfg.get("bias", True)
    )

def build_pool2d(cfg: Dict[str, Any]) -> nn.Module:
    """
    cfg example:
        {
            "type": "pool",
            "kernel_size": 2,
            "stride": 2,
            "padding": 0
        }
    """
    return nn.MaxPool2d(
        kernel_size=cfg.get("kernel_size", 2),
        stride=cfg.get("stride", 2),
        padding=cfg.get("padding", 0)
    )

def build_convlstm_cell(cfg: Dict[str, Any]) -> nn.Module:
    return ConvLSTM_cell(**cfg)

def build_convgru_cell(cfg: Dict[str, Any]) -> nn.Module:
    return ConvGRU_cell(**cfg)

def build_convgru_cell_v2(cfg: Dict[str, Any]) -> nn.Module:
    return ConvGRU_cell_v2(**cfg)

def build_ftcgrru_cell(cfg: Dict[str, Any]) -> nn.Module:
    return FTCGRUCell(**cfg)

def build_single_frame_ftcgrru_cell(cfg: Dict[str, Any]) -> nn.Module:
    return singleFrameFTCGRUCell(**cfg)

###############################################################################
# Registry: Map string to specific build functions or classes
###############################################################################

ACTIVATION_REGISTRY = {
    "relu":       (nn.ReLU,       {"inplace": True}),
    "leaky_relu": (nn.LeakyReLU,  {"negative_slope": 0.2, "inplace": True}),
}


LAYER_BUILDERS = {
    "conv":   build_conv2d,
    "deconv": build_deconv2d,
    "pool":   build_pool2d,
}


RNN_BUILDERS = {
    "convgru":     build_convgru_cell,
    "convgru_v2":  build_convgru_cell_v2,
}

def build_layer(cfg: Dict[str, Any]) -> nn.Module:
    """
    giving a layer config, build a nn.Module.
    also add activation function if it is in the config.
    """
    layer_type = cfg["type"].lower()
    if layer_type not in LAYER_BUILDERS:
        raise ValueError(f"Unknown layer type '{layer_type}' in config.")
    # 1) build key layers
    core_layer = LAYER_BUILDERS[layer_type](cfg)

    # 2) 若带激活函数
    act_name = cfg.get("activation", None)
    if act_name is None:
        return core_layer
    if act_name not in ACTIVATION_REGISTRY:
        raise ValueError(f"Unknown activation '{act_name}'.")
    act_class, act_kwargs = ACTIVATION_REGISTRY[act_name]
    activation_layer = act_class(**act_kwargs)

    # 3) 封装到 Sequential
    return nn.Sequential(core_layer, activation_layer)


def build_rnn_cell(cfg: Dict[str, Any]) -> nn.Module:
    """
    根据RNN配置构建一个 RNN Cell (比如 ConvGRUCellV2)
    """
    cell_type = cfg["type"].lower()
    if cell_type not in RNN_BUILDERS:
        raise ValueError(f"Unknown RNN cell type '{cell_type}'.")
    return RNN_BUILDERS[cell_type](cfg)


def build_layers(cfg_list: List[Dict[str, Any]]) -> nn.Sequential:
    """
    将多层的配置打包成一个 nn.Sequential
    """
    modules = []
    for idx, layer_cfg in enumerate(cfg_list):
        layer = build_layer(layer_cfg)
        modules.append((f"{layer_cfg['type']}_{idx}", layer))
    return nn.Sequential(nn.ModuleDict(modules))


def build_rnn_cells(cfg_list: List[Dict[str, Any]]) -> nn.ModuleList:
    """
    将多个 RNN Cell 配置打包成一个 nn.ModuleList
    """
    cells = [build_rnn_cell(cfg) for cfg in cfg_list]
    return nn.ModuleList(cells)

BUILDERS_REGISTRY: Dict[str, Any] = {
        "relu":       (nn.ReLU,       {"inplace": True}),
        "leaky_relu": (nn.LeakyReLU,  {"negative_slope": 0.2, "inplace": True}),
        "tanh":       (nn.Tanh,       {}),
        "sigmoid":    (nn.Sigmoid,    {}),
}
def build_layer(cfg: Dict[str, Any]) -> nn.Module:
    """
    giving a layer config, build a nn.Module.
    also add activation function if it is in the config.
    
    Inputs:
        cfg (dict): layer config
    
    Outputs:
        layer (nn.Module) 

    """
    


    pass