import torch
import torch.nn as nn

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()

        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu_activation = nn.LeakyReLU(inplace=True)
        self.maxpool_layer = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.relu_activation(x)
        x = self.maxpool_layer(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()

        self.upsample_layer = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu_activation = nn.LeakyReLU(inplace=True)

    def forward(self, x, x_i):
        x = x.unsqueeze(0)
        x = self.upsample_layer(x)
        x = x.squeeze(0)
        x = torch.cat((x,x_i),0)
        x = self.conv_layer(x)
        x = self.relu_activation(x)
        return x


class ConvolutionalAutoencoder(nn.Module):
    def __init__(self,img_shape):
        super(ConvolutionalAutoencoder, self).__init__()
        self.fc_in = int(img_shape / 16)
        # Encoder
        self.encoder_block1 = EncoderBlock(in_channels=1,  out_channels=16)
        self.encoder_block2 = EncoderBlock(in_channels=16, out_channels=24)
        self.encoder_block3 = EncoderBlock(in_channels=24, out_channels=36)
        self.encoder_block4 = EncoderBlock(in_channels=36, out_channels=54)
        
        self.fc_layer1 = nn.Linear(in_features=54 * self.fc_in * self.fc_in, out_features=9720)
        self.dropout_layer1 = nn.Dropout(p=0.3)
        self.fc_layer2 = nn.Linear(in_features=9720, out_features=48600)
        self.dropout_layer2 = nn.Dropout(p=0.3)

        # Decoder
        self.decoder_block1 = DecoderBlock(in_channels=90, out_channels=36)
        self.decoder_block2 = DecoderBlock(in_channels=60, out_channels=24)
        self.decoder_block3 = DecoderBlock(in_channels=40, out_channels=16)
        self.decoder_block4 = DecoderBlock(in_channels=17, out_channels=1)

    def forward(self, x):
        # Encoder
        # 1 480 480
        x_0 = x 
        x_1 = self.encoder_block1(x)            # -> 16 240 240
        x_2 = self.encoder_block2(x_1)          # -> 24 120 120
        x_3 = self.encoder_block3(x_2)          # -> 36 60  60 
        x_4 = self.encoder_block4(x_3)          # -> 54 30  30 
        
        # return a 1-D tensor
        assert self.fc_in == x_4.shape[1], "img shape do not match"
        x_4 = x_4.reshape(-1)                   # -> 48600
        
        x = self.fc_layer1(x_4)                 # -> 529
        x = self.dropout_layer1(x)          
        x = self.fc_layer2(x)                   # -> 2646
        x = self.dropout_layer2(x)
        x = x.view(54, self.fc_in, self.fc_in)

        # Decoder
        x = self.decoder_block1(x,  x_3)
        x = self.decoder_block2(x,  x_2)
        x = self.decoder_block3(x,  x_1)
        x = self.decoder_block4(x,  x_0)

        return x
