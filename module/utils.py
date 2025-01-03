import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any
from .layers.recurrent import *
from .models.encoder import Encoder
from .models.decoder import Decoder
from .models.ed import ED

###############################################################################
# Func: layer builder
###############################################################################

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
    # 1) 先构建核心层
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

###############################################################################
# 3. Encoder / Decoder / ED 示例
###############################################################################


class Decoder(nn.Module):
    """
    Decoder: 每个stage包括:
      1) 一个 RNN Cell
      2) 若干层 UpConv/Conv
    """
    def __init__(
        self,
        stage_conv_cfgs: List[List[Dict[str, Any]]],
        stage_rnn_cfgs:  List[Dict[str, Any]]
    ):
        super().__init__()
        assert len(stage_conv_cfgs) == len(stage_rnn_cfgs)

        self.num_stages = len(stage_conv_cfgs)
        self.rnns  = build_rnn_cells(stage_rnn_cfgs)
        self.convs = nn.ModuleList([
            build_layers(conv_cfgs) for conv_cfgs in stage_conv_cfgs
        ])

    def forward(self, hidden_states: List[torch.Tensor]):
        """
        hidden_states是 encoder 的输出(列表), 这里简单演示:
          - 解码时逆序使用 hidden_state
          - 构造一个固定长度 T 来进行解码
        """
        hidden_states = hidden_states[::-1]  # 反转
        T = 5  # 示例: 假设想解码出 T 帧
        x = None

        for i, (rnn_cell, conv_block) in enumerate(zip(self.rnns, self.convs)):
            hidden = hidden_states[i]
            outs = []
            for t in range(T):
                if x is None:
                    # 说明是第一个 stage 的第一个时间步
                    # hidden.shape = (B, C, H, W)
                    dummy_in = torch.zeros_like(hidden)
                    out, hidden = rnn_cell(dummy_in, hidden)
                else:
                    out, hidden = rnn_cell(x[t], hidden)
                outs.append(out.unsqueeze(0))

            x = torch.cat(outs, dim=0)  # (T, B, C', H', W')
            # UpConv / Conv
            T_, B, C_, H_, W_ = x.shape
            x_reshaped = x.reshape(T_ * B, C_, H_, W_)
            feat = conv_block(x_reshaped)
            _, C2, H2, W2 = feat.shape
            x = feat.view(T_, B, C2, H2, W2)

        return x.transpose(0, 1)  # -> (B, T, C, H, W)



###############################################################################
# 4. 一个简单的测试用例
###############################################################################
def test_model():
    # 假设我们想构建2个stage的Encoder:
    encoder_stage_conv_cfgs = [
        # stage0: conv + pool
        [
            {"type": "conv",  "in_channels":1,"out_channels":4, "kernel_size":3,"padding":1,"activation":"leaky_relu"},
            {"type": "pool",  "kernel_size":2,"stride":2}  # 不带 activation
        ],
        # stage1: conv
        [
            {"type": "conv",  "in_channels":4,"out_channels":8, "kernel_size":3,"padding":1,"activation":"leaky_relu"}
        ]
    ]
    # 对应的 RNN Cell:
    encoder_stage_rnn_cfgs = [
        {"type": "convgru", "shape": [32,32], "in_channels":4, "out_channels":4},
        {"type": "convgru_v2", "shape": [32,32], "in_channels":8, "out_channels":8}
    ]

    encoder = Encoder(encoder_stage_conv_cfgs, encoder_stage_rnn_cfgs)

    # Decoder 也2个stage: rnn + deconv
    decoder_stage_conv_cfgs = [
        # stage0
        [
            {"type":"deconv", "in_channels":8, "out_channels":4, "kernel_size":4,"stride":2,"padding":1,"activation":"leaky_relu"}
        ],
        # stage1
        [
            {"type":"conv",   "in_channels":4, "out_channels":1, "kernel_size":3,"padding":1}
        ]
    ]
    # RNN cells
    decoder_stage_rnn_cfgs = [
        {"type": "convgru_v2", "shape": [64,64], "in_channels":8, "out_channels":8},
        {"type": "convgru",    "shape": [64,64], "in_channels":4, "out_channels":4},
    ]
    decoder = Decoder(decoder_stage_conv_cfgs, decoder_stage_rnn_cfgs)

    # 整合
    model = ED(encoder, decoder)

    # 试跑
    x = torch.randn(2, 5, 1, 64, 64)  # (B=2, T=5, C=1, H=64, W=64)
    y = model(x)
    print("Output shape:", y.shape)  # (2, T, C, H, W), 其中 T=5(示例)


if __name__ == "__main__":
    test_model()
