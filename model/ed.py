import torch
from torch import nn
import torch.nn.functional as F

class ED(nn.Module):
    def __init__(self, encoder, decoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, inputs):
        state_list = self.encoder(inputs)
        output = self.decoder(state_list)
        return output
