import torch
from torch import nn
import sys
sys.path.append('..')
from torch.nn import functional as F
from model.pconv import PartialConv2d
from model.ffconv import FlourierConv3d
class ConvLSTM_cell(nn.Module):
    def __init__(self, shape, channels, kernel_size, features_num, fconv=False, frames_len=10,is_cuda=False) -> None:
        super().__init__()
        
        self.shape = shape
        self.channels = channels
        self.features_num = features_num
        self.kernel_size = kernel_size
        self.fconv = fconv
        self.padding = (kernel_size - 1) // 2
        self.frames_len = frames_len
        self.is_cuda = is_cuda
        assert self.features_num % 4 == 0 
        
        groups_num = 4 * self.features_num // 4
        channel_num = 4 * self.features_num
        
        self.conv = nn.Sequential(
            nn.Conv2d(
                self.channels + self.features_num,
                4 * self.features_num,
                self.kernel_size,
                padding=self.padding
            ),
            nn.GroupNorm(groups_num, channel_num)
        )
        
        if fconv:
            self.semi_conv = nn.Sequential(
                nn.Conv2d(
                    2 * (self.channels + self.features_num),
                    4 * self.features_num,
                    self.kernel_size,
                    padding=self.padding
                ),
                nn.GroupNorm(groups_num, channel_num),
                nn.LeakyReLU(inplace=True)
            )
            self.global_conv = nn.Sequential(
                nn.Conv2d(
                    8 * self.features_num,
                    4 * self.features_num,
                    self.kernel_size,
                    padding=self.padding
                ),
                nn.GroupNorm(groups_num, channel_num)
            )
            
        
    def forward(self, inputs=None, hidden_state=None,):
        hx, cx = self.get_hx_cx(inputs,hidden_state)
        
        output_contain = []
        
        for index in range(self.frames_len):
            if inputs is None:
                x = torch.zeros(
                    hx.size(0),
                    self.channels,
                    self.shape[0],                
                    self.shape[1]
                )
                if self.is_cuda:
                    x = x.cuda()
            else:
                x = inputs[index]

            hy, cy = self.gates_cal(x, hx, cx)
            
            output_contain .append(hy)
            
            hx , cx = hy, cy
        return torch.stack(output_contain), (hy, cy)
        
        
    def get_hx_cx(self, inputs, hidden_state):
        if hidden_state is None:
            hx = torch.zeros(
                inputs.size(1),      
                self.features_num,  
                self.shape[0],      
                self.shape[1]   
            )
            cx = torch.zeros(
                inputs.size(1),      
                self.features_num, 
                self.shape[0],     
                self.shape[1]   
            )
        else:
            hx, cx = hidden_state
        
        return hx, cx
    
    def gates_cal(self, x:torch.Tensor, hx, cx):
        current_device = x.device
        hx = hx.to(current_device)
        cx = cx.to(current_device)
        concat = torch.cat((x, hx), 1)
        # local convolution
        gates_out = self.conv(concat)
        
        if self.fconv:
            fft_dim = (-2, -1)
            # real fast flourier transform
            thw2freq = torch.fft.rfftn(concat,dim=fft_dim,norm='ortho')
            
            # reshape array
            thw2freq = torch.stack((thw2freq.real, thw2freq.imag),dim=-1)
            # N, C, H, W/2+1, 2 -> N, C, 2, H, W/2+1
            thw2freq = thw2freq.permute(0,1,4,2,3).contiguous()
            # N, C, 2, H, W/2+1 -> N, 2*C,  H, W/2+1
            thw2freq = thw2freq.view((concat.size()[0], -1,) + thw2freq.size()[3:])         
            
            # frequency convolution (semi convolution)
            ffc_conv = self.semi_conv(thw2freq)
            
            # inverse real faset flourier transform
            ifft_shape = ffc_conv.shape[-2:]
            ffc_out = torch.fft.irfftn(ffc_conv,s=ifft_shape,dim=fft_dim,norm='ortho')
            
            # reshape
            # N, C,  H, W/2+1 -> N, C,  H, W 
            ffc_out_reshape = F.interpolate(ffc_out, size=gates_out.size()[-2:], mode='bilinear',align_corners=False)
            ffc_out_concat_gates_out = torch.cat((ffc_out_reshape, gates_out), 1)

            # global convolution
            gates_out = self.global_conv(ffc_out_concat_gates_out)
        
        
        in_gate, forget_gate, hat_cell_gate, out_gate = torch.split(
            gates_out,
            self.features_num,
            dim = 1
        )
        
        in_gate = torch.sigmoid(in_gate) 
        forget_gate = torch.sigmoid(forget_gate)
        hat_cell_gate = torch.tanh(hat_cell_gate)
        out_gate = torch.sigmoid(out_gate)
        
        cell_state = (forget_gate * cx) + (in_gate * hat_cell_gate)
        out_state = out_gate * torch.tanh(cell_state)
        
        return out_state, cell_state
