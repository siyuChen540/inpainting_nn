import torch
from torch import nn
from .encoder import Encoder
from .decoder import Decoder

class ED(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(ED, self).__init__()
        self.encoder = Encoder(in_channels)
        self.decoder = Decoder(out_channels)

    def forward(self, x):
        x, features = self.encoder(x)
        x = self.decoder(x, features)
        return x 