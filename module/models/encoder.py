"""
    @author: Siyu Chen
    @date: 2025-1-4
    @description: Encoder for ED model.
    @file: module/models/encoder.py
    @email: chensy57@mail2.sysu.edu.cn
"""

from typing import List, Dict, Any

import torch
from torch import nn
from ..utils import build_layers, build_rnn_cells


class Encoder(nn.Module):
    """
    Encoder is optimized for reduced memory consumption.

    Encoder: each stage includes:
      1) several conv/pool layers (nn.Sequential)
      2) one rnn cell (e.g. ConvGRUCell)
    
    Args:
        stage_conv_cfgs (List[List[Dict[str, Any]]]): each stage's conv layers config
        stage_rnn_cfgs  (List[Dict[str, Any]]): each stage's rnn cell config
    
    Inputs:
        x (torch.Tensor): input frames (B, T, C, H, W)
    
    Outputs:
        hidden_states (List[torch.Tensor]): hidden states of each stage

    """
    def __init__(
        self,
        stage_conv_cfgs: List[List[Dict[str, Any]]],  # each stage's conv layers config
        stage_rnn_cfgs:  List[Dict[str, Any]]         # each stage's rnn cell config
    ):
        super().__init__()
        assert len(stage_conv_cfgs) == len(stage_rnn_cfgs)

        self.convs = nn.ModuleList([
            build_layers(conv_cfgs) 
            for conv_cfgs in stage_conv_cfgs
        ])
        self.rnns  = build_rnn_cells(stage_rnn_cfgs)

    def forward(self, x: torch.Tensor):
        # x: (B, T, C, H, W)
        x = x.transpose(0, 1)  # -> (T, B, C, H, W)
        hidden_states = []
        
        for conv_block, rnn_cell in zip(self.convs, self.rnns):
            T, B, C, H, W = x.shape
            # 1) conv processing
            x_reshaped = x.reshape(T * B, C, H, W)
            feat = conv_block(x_reshaped)  # -> (T*B, C', H', W')
            _, C2, H2, W2 = feat.shape
            feat = feat.view(T, B, C2, H2, W2)

            # 2) rnn processing by frame
            outs, hidden = [], None
            for t in range(T):
                out, hidden = rnn_cell(feat[t], hidden)
                outs.append(out.unsqueeze(0))
            x = torch.cat(outs, dim=0)  # (T, B, C2, H2, W2)

            hidden_states.append(hidden)

        return hidden_states  