"""
    @Author: Siyu Chen
    @Date: 2025-1-4
    @Description: Decoder for ED model.
    @File: module/models/decoder.py
    @email: chensy57@mail2.sysu.edu.cn
"""

from typing import List, Dict, Any

import torch
from torch import nn
from ..utils import build_layers, build_rnn_cells

class Decoder(nn.Module):
    """
    Decoder composed of multiple ConvLSTM cells and 
    UpConv (DeConv) layers to reconstruct frames.
    Optimized for reduced memory consumption.

    Args:
        stage_conv_cfgs (dict): each stage's convolutional layers config
        stage_rnn_cfgs  (dict): each stage's recurrent neural network config

    Inputs:
        hidden_states   (list): output of encoder
    
    output:
        output          (list): reconstruct dataset matching with input of encoder

    """
    def __init__(
        self, 
        stage_conv_cfgs, 
        stage_rnn_cfgs
    ):
        super().__init__()
        assert len(stage_conv_cfgs) == len(stage_rnn_cfgs)
        self.block_num = len(stage_conv_cfgs)
        self.convs = nn.ModuleList([
            build_layers(conv_cfgs) 
            for conv_cfgs in stage_conv_cfgs
        ])
        self.convlstm_cells = nn.ModuleList(stage_rnn_cfgs)

    def forward(self, hidden_states):
        # hidden_states are reversed for decoding
        hidden_states = hidden_states[::-1]
        inputs = None
        for i in range(self.block_num):
            # ConvLSTM cell (decoder)
            outputs, _ = self.convlstm_cells[i](inputs, hidden_states[i])
            seq_num, bsz, ch, h, w = outputs.size()
            reshaped = outputs.reshape(-1, ch, h, w)
            processed = self.convs[i](reshaped)
            _, nch, nh, nw = processed.size()
            inputs = processed.view(seq_num, bsz, nch, nh, nw)

        # final output shape: 
        #   (frames, batch, C, H, W) -> (batch, frames, C, H, W)
        return inputs.transpose(0, 1)
    


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