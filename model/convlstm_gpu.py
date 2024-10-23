import torch
import torch.nn as nn
class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        self.conv_i = nn.Conv2d(in_channels=input_channels, out_channels=4*hidden_channels, kernel_size=kernel_size, padding=self.padding)
        self.conv_h = nn.Conv2d(in_channels=hidden_channels, out_channels=4*hidden_channels, kernel_size=kernel_size, padding=self.padding)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        
        # concatenate along channel axis
        combined = torch.cat([input_tensor, h_cur], dim=1)
        
        # input gate, forget gate, cell, and output gate
        gates = self.conv_i(combined) + self.conv_h(h_cur)
        
        i = torch.sigmoid(gates[:, :self.hidden_channels, :, :])
        f = torch.sigmoid(gates[:, self.hidden_channels:2*self.hidden_channels, :, :])
        g = torch.tanh(gates[:, 2*self.hidden_channels:3*self.hidden_channels, :, :])
        o = torch.sigmoid(gates[:, 3*self.hidden_channels:, :, :])
        
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, batch_first=True):
        super(ConvLSTM, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.batch_first = batch_first
        self.cell = ConvLSTMCell(input_channels, hidden_channels, kernel_size)

    def forward(self, input_tensor, init_states=None):
        if self.batch_first:
            # swap batch and sequence length axis
            input_tensor = input_tensor.permute(0, 4, 1, 2, 3)

        # get batch size, sequence length, height, and width from input tensor
        batch_size, seq_len, _, height, width = input_tensor.size()
        
        # if initial states are not given, set them to zero tensors with correct size
        if init_states is None:
            h = torch.zeros((batch_size, self.hidden_channels, height, width), dtype=input_tensor.dtype, device=input_tensor.device)
            c = torch.zeros((batch_size, self.hidden_channels, height, width), dtype=input_tensor.dtype, device=input_tensor.device)
        else:
            h, c = init_states
        
        # list to store output hidden states
        hidden_seq = []
        
        # iterate over time steps
        for t in range(seq_len):
            h, c = self.cell(input_tensor[:, t, :, :, :], (h, c))
            hidden_seq.append(h)
            
        # stack hidden states into tensor and swap batch and sequence length axis back
        hidden_seq = torch.stack(hidden_seq, dim=1).permute(0, 2, 3, 4, 1)

        return hidden_seq
