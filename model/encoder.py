# -*- encoding: utf-8 -*-
'''
@File        :  encoder.py
@Time        :  2022/12/17 10:24:00
@Author      :  chen siyu
@Mail        :  chensiyu57@mail2.sysu.edu.cn
@Version     :  1.0
@Description :  encoder
'''

import torch
from torch import nn
import logging

import sys
sys.path.append('..')

from utils import mkLayer

class Encoder(nn.Module):
    def __init__(self, child_nets_param, convlstms):
        super().__init__()
        assert len(child_nets_param) == len(convlstms)
        
        self.block_num = len(child_nets_param)
        
        for i, (child_net_cell_param, convlstm_cell) in enumerate(zip(child_nets_param, convlstms)):
            setattr(self, 'child_cell' + str(i), mkLayer(child_net_cell_param))
            setattr(self, 'convlstm_cell' + str(i), convlstm_cell)
    
    
    def forward(self, inputs:torch.Tensor):
        inputs = inputs.transpose(0, 1)
        hidden_state = []
        logging.debug(inputs.size())
        
        for i in range(self.block_num):
            # Bidirectional
            if i == 0:
                pass
            else:
                inputs = torch.flip(inputs, [0])
            inputs, state_stage = self.forward_by_step(
                inputs,
                getattr(self, 'child_cell' + str(i)),
                getattr(self, 'convlstm_cell' + str(i))
            )
            hidden_state.append(state_stage)
        
        return tuple(hidden_state)
    
    def forward_by_step(self, inputs, child_cell, convlstm_cell):
        frame_num, batch_size, channel, height, width = inputs.size()
        
        inputs = torch.reshape(inputs, (-1, channel, height, width))
        inputs = child_cell(inputs)
        inputs = torch.reshape(inputs, (frame_num, batch_size, inputs.size(1), inputs.size(2), inputs.size(3)))
        
        outputs, state = convlstm_cell(inputs, None)
        
        return outputs, state
