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

import sys
sys.path.append('..')

from utils import mkLayer

class Decoder(nn.Module):
    def __init__(self, child_nets_param, convlstms):
        super().__init__()
        
        assert len(child_nets_param) == len(convlstms)
        # confim child net block
        self.block_num = len(child_nets_param)
        
        for i, (child_net_cell_param, convlstm_cell) in enumerate(zip(child_nets_param, convlstms)):
            setattr(self, 'child_cell' + str(i), mkLayer(child_net_cell_param))
            setattr(self, 'convlstm_cell' + str(i), convlstm_cell)
        # self.conv_layer = nn.Conv2d(in_channels=convlstms[0].frames_len,out_channels=1, kernel_size=3,padding=(3-1)//2)

    def forward(self, hidden_states):
        # intialize inputs
        inputs = None
        # reverse hidden state list
        hidden_states = hidden_states[::-1]
        
        for i in range(self.block_num):
            inputs = self.forward_by_step(
                inputs,
                hidden_states[i],
                getattr(self, "child_cell" + str(i)),
                getattr(self, "convlstm_cell" + str(i)),
            )
        
        inputs = inputs.transpose(0,1)
        # inputs = self.conv_layer(inputs)
        return inputs
    
    def forward_by_step(self, inputs, hidden_state, child_cell, convlstm_cell, frames_len=10):
        inputs, state = convlstm_cell(inputs, hidden_state)
        
        seq_number, batch_size, input_channel, height, width = inputs.size()
        inputs = torch.reshape(inputs, (-1, input_channel, height, width))
        
        inputs = child_cell(inputs)

        _, input_channel, height, width = inputs.size()

        inputs = torch.reshape(inputs, (seq_number, batch_size, input_channel, height, width))
        
        return inputs