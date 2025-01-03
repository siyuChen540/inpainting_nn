import torch
from torch import nn
from ..utils import mkLayer

class Decoder(nn.Module):
    """
    Decoder composed of multiple ConvLSTM cells and UpConv (DeConv) layers to reconstruct frames.
    Optimized for reduced memory consumption.
    """
    def __init__(self, child_nets_params, convlstm_cells):
        super().__init__()
        assert len(child_nets_params) == len(convlstm_cells)
        self.block_num = len(child_nets_params)
        self.child_cells = nn.ModuleList([mkLayer(params) for params in child_nets_params])
        self.convlstm_cells = nn.ModuleList(convlstm_cells)

    def forward(self, hidden_states):
        # hidden_states are reversed for decoding
        hidden_states = hidden_states[::-1]
        inputs = None
        for i in range(self.block_num):
            # ConvLSTM cell (decoder)
            outputs, _ = self.convlstm_cells[i](inputs, hidden_states[i])
            seq_num, bsz, ch, h, w = outputs.size()
            reshaped = outputs.reshape(-1, ch, h, w)
            processed = self.child_cells[i](reshaped)
            _, nch, nh, nw = processed.size()
            inputs = processed.view(seq_num, bsz, nch, nh, nw)

        # final output shape: (frames, batch, C, H, W) -> (batch, frames, C, H, W)
        return inputs.transpose(0, 1)