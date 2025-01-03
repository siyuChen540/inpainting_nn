"""
    @Author: mikey.zhaopeng 
    @Date: 2025-01-03 16:52:04 
    @Last Modified by: mikey.zhaopeng
    @Last Modified time: 2025-01-03 16:52:32
"""
import torch
from torch import nn
import torch.nn.functional as F
from .basic import DSConv2d, SELayer

class ConvLSTM_cell(nn.Module):
    """
    A ConvLSTM cell with optional frequency-domain convolution (fconv).
    """
    def __init__(self, input_channels, kernel_size, features_num, fconv=True, frames_len=10, device='cuda'):
        super().__init__()
        self.input_channels = input_channels
        self.features_num = features_num
        self.kernel_size = kernel_size
        self.fconv = fconv
        self.padding = (kernel_size - 1) // 2
        self.frames_len = frames_len
        self.device = device

        groups_num = max(1, (4 * self.features_num) // 4)
        channel_num = 4 * self.features_num

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_channels + self.features_num, channel_num, self.kernel_size, padding=self.padding),
            nn.GroupNorm(groups_num, channel_num)
        )

        if fconv:
            self.semi_conv = nn.Sequential(
                nn.Conv2d(2 * (self.input_channels + self.features_num), channel_num, self.kernel_size, padding=self.padding),
                nn.GroupNorm(groups_num, channel_num),
                nn.LeakyReLU(inplace=True)
            )
            self.global_conv = nn.Sequential(
                nn.Conv2d(8 * self.features_num, 4 * self.features_num, self.kernel_size, padding=self.padding),
                nn.GroupNorm(groups_num, channel_num)
            )

    def forward(self, inputs, hidden_state=None):
        hx, cx = self._init_hidden(inputs, hidden_state)
        output_frames = []

        for t in range(self.frames_len):
            x = inputs[t].to(hx.device)

            hy, cy = self._step(x, hx, cx)
            output_frames.append(hy)
            hx, cy = hy, cy

        return torch.stack(output_frames), (hy, cy)

    def _init_hidden(self, inputs, hidden_state):
        if hidden_state is not None:
            return hidden_state

        bsz = inputs.size(1) if inputs is not None else 1
        hx = torch.zeros(bsz, self.features_num, self.shape[0], self.shape[1], device=self.device)
        cx = torch.zeros_like(hx)
        return hx, cx

    def _step(self, x: torch.Tensor, hx: torch.Tensor, cx: torch.Tensor):
        concat = torch.cat((x, hx), dim=1)
        gates_out = self.conv(concat)           # 

        if self.fconv:
            gates_out = self._fconv(concat, gates_out)

        in_gate, forget_gate, hat_cell_gate, out_gate = torch.split(gates_out, self.features_num, dim=1)
        in_gate = torch.sigmoid(in_gate)
        forget_gate = torch.sigmoid(forget_gate)
        hat_cell_gate = torch.tanh(hat_cell_gate)
        out_gate = torch.sigmoid(out_gate)

        cy = (forget_gate * cx) + (in_gate * hat_cell_gate)
        hy = out_gate * torch.tanh(cy)

        return hy, cy
    
    def _fconv(self, concat, gates_out):
        # Optional frequency domain convolution
        # Fourier transform
        fft_dim = (-2, -1)
        freq = torch.fft.rfftn(concat, dim=fft_dim, norm='ortho')
        freq = torch.stack((freq.real, freq.imag), dim=-1)
        # Rearrange for convolution
        freq = freq.permute(0, 1, 4, 2, 3).contiguous()
        N, C, _, H, W2 = freq.size()
        freq = freq.view(N, -1, H, W2)
        ffc_conv = self.semi_conv(freq)
        # iFFT
        ifft_shape = ffc_conv.shape[-2:]
        ffc_out = torch.fft.irfftn(torch.complex(ffc_conv, torch.zeros_like(ffc_conv)), s=ifft_shape, dim=fft_dim, norm='ortho')
        # Resize to match gates_out size
        ffc_out_resize = F.interpolate(ffc_out, size=gates_out.size()[-2:], mode='bilinear', align_corners=False)
        combined = torch.cat((ffc_out_resize, gates_out), 1)
        gates_out = self.global_conv(combined)
        return gates_out

class ConvGRU_cell(nn.Module):
    """
    A ConvGRU cell with optional frequency-domain convolution (fconv).
    Args Description:
        shape        : input shape (H, W)
        channels     : input channels
        kernel_size  : convolution kernel size
        features_num : number of features in the cell
        fconv        : whether to use frequency-domain convolution
        frames_len   : number of frames in the input sequence
        is_cuda      : whether to use cuda device
    """
    def __init__(self, input_channels, kernel_size, features_num, fconv=True, frames_len=10, device='cuda'):
        super().__init__()
        self.input_channels = input_channels
        self.features_num = features_num
        self.kernel_size = kernel_size
        self.fconv = fconv
        self.padding = (kernel_size - 1) // 2
        self.frames_len = frames_len
        self.device = device

        groups_num = max(1, self.features_num)  # 3 gates for GRU
        channel_num = 3 * self.features_num     # GRU has 3 gates: update, reset, candidate

        # Standard convolution for GRU gates (update, reset, candidate)
        self.conv = nn.Sequential(
            nn.Conv2d(self.channels + self.features_num, channel_num, self.kernel_size, padding=self.padding),
            nn.GroupNorm(groups_num, channel_num)
        )

        # Optional frequency-domain convolution for fconv=True  
        if fconv:
            self.semi_conv = nn.Sequential(
                nn.Conv2d(2 * (self.channels + self.features_num), channel_num, self.kernel_size, padding=self.padding),
                nn.GroupNorm(groups_num, channel_num),
                nn.LeakyReLU(inplace=True)
            )
            self.global_conv = nn.Sequential(
                nn.Conv2d(6 * self.features_num, 3 * self.features_num, self.kernel_size, padding=self.padding),
                nn.GroupNorm(groups_num, channel_num)
            )

    def forward(self, inputs, hidden_state=None):
        hx = self._init_hidden(inputs, hidden_state)
        output_frames = []

        for t in range(self.frames_len):
            x = inputs[t].to(hx.device)

            hy = self._step(x, hx)
            output_frames.append(hy)
            hx = hy

        return torch.stack(output_frames), hx  # Only return hidden state in GRU

    def _init_hidden(self, inputs, hidden_state):
        if hidden_state is not None:
            return hidden_state

        bsz = inputs.size(1) if inputs is not None else 1
        hx = torch.zeros(bsz, self.features_num, self.shape[0], self.shape[1], device=self.device)
        return hx

    def _step(self, x: torch.Tensor, hx: torch.Tensor):
        concat = torch.cat((x, hx), dim=1)
        gates_out = self.conv(concat)

        # Optional frequency domain convolution
        if self.fconv:
            gates_out = self._fconv(concat, gates_out)

        # Split gates: [reset, update, candidate]
        reset_gate, update_gate, candidate_gate = torch.split(gates_out, self.features_num, dim=1)
        reset_gate = torch.sigmoid(reset_gate)
        update_gate = torch.sigmoid(update_gate)
        candidate_gate = torch.tanh(candidate_gate)

        # Compute hidden state update
        hy = update_gate * hx + (1 - update_gate) * candidate_gate

        return hy
    
    def _fconv(self, concat, gates_out):
        # Fourier transform
        fft_dim = (-2, -1)
        freq = torch.fft.rfftn(concat, dim=fft_dim, norm='ortho')
        freq = torch.stack((freq.real, freq.imag), dim=-1)      
        # Rearrange for convolution
        freq = freq.permute(0, 1, 4, 2, 3).contiguous()
        N, C, _, H, W2 = freq.size()
        freq = freq.view(N, -1, H, W2)
        ffc_conv = self.semi_conv(freq)
        # iFFT
        ifft_shape = ffc_conv.shape[-2:]
        ffc_out = torch.fft.irfftn(torch.complex(ffc_conv, torch.zeros_like(ffc_conv)), s=ifft_shape, dim=fft_dim, norm='ortho')
        # Resize to match gates_out size
        ffc_out_resize = F.interpolate(ffc_out, size=gates_out.size()[-2:], mode='bilinear', align_corners=False)
        combined = torch.cat((ffc_out_resize, gates_out), 1)
        gates_out = self.global_conv(combined)
        return gates_out



class ConvGRU_cell_v2(ConvGRU_cell):
    """
    A ConvGRU cell with optional frequency-domain convolution (fconv) and in deep wise separable convolution.
        def __init__(self, input_channels, hidden_channels, kernel_size, 
                 use_se=False, num_frames=10, device='cuda', fourier_norm='ortho', 
                 spatial_scale_mode='bilinear'):
    """
    def __init__(self, input_channels, kernel_size, hidden_channels, fconv=True, num_frames=10, device='cuda'):
        super().__init__(input_channels, kernel_size, hidden_channels, fconv, num_frames, device)
        self.channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.fconv = fconv
        self.padding = (kernel_size - 1) // 2
        self.num_frames = num_frames
        self.device = device

        groups_num = max(1, self.hidden_channels)  # 3 gates for GRU
        channel_num = 3 * self.hidden_channels  # GRU has 3 gates: update, reset, candidate

        # Standard convolution for GRU gates (update, reset, candidate)
        self.conv = nn.Sequential(
            DSConv2d(self.channels + self.hidden_channels, channel_num, self.kernel_size, padding=self.padding),
            nn.GroupNorm(groups_num, channel_num)
        )

        # Optional frequency-domain convolution for fconv=True
        if fconv:
            self.semi_conv = nn.Sequential(
                DSConv2d(2 * (self.channels + self.hidden_channels), channel_num, self.kernel_size, padding=self.padding),
                nn.GroupNorm(groups_num, channel_num),
                nn.LeakyReLU(inplace=True)
            )
            self.global_conv = nn.Sequential(
                DSConv2d(6 * self.hidden_channels, 3 * self.hidden_channels, self.kernel_size, padding=self.padding),
                nn.GroupNorm(groups_num, channel_num)
            )


class FTCGRUCell(nn.Module):
    """
    Frequency and Temporal Convolutional Gated Recurrent Unit (FTCGRU) Cell.
    
    This cell integrates convolutional operations in the frequency domain and utilizes depthwise separable convolutions optionally enhanced with Squeeze-and-Excitation (SE) blocks.
    
    Args:
        input_channels (int): Number of input channels.
        hidden_channels (int): Number of hidden channels in the GRU cell.
        kernel_size (int or tuple): Size of the convolutional kernel.
        use_se (bool, optional): Whether to use Squeeze-and-Excitation blocks. Default is False.
        num_frames (int, optional): Number of frames to process (sequence length). Default is 10.
        device (str, optional): Device to perform computations on ('cuda' or 'cpu'). Default is 'cuda'.
        fourier_norm (str, optional): Normalization method for Fourier transforms. Default is 'ortho'.
        spatial_scale_mode (str, optional): Interpolation mode for spatial scaling. Default is 'bilinear'.
    """
    def __init__(self, input_channels, hidden_channels, kernel_size, 
                 use_se=False, num_frames=10, device='cuda', fourier_norm='ortho', 
                 spatial_scale_mode='bilinear'):
        super(FTCGRUCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        self.num_frames = num_frames
        self.device = device
        self.use_se = use_se
        self.fourier_norm = fourier_norm
        self.spatial_scale_mode = spatial_scale_mode

        # Number of groups for Group Normalization
        self.num_groups = max(1, self.hidden_channels)
        # Total number of gates: update, reset, and candidate
        self.num_gates = 3 * self.hidden_channels
        
        # Convolution to compute all gates at once
        self.conv = nn.Conv2d(
            in_channels=self.input_channels + self.hidden_channels, 
            out_channels=self.num_gates, 
            kernel_size=self.kernel_size, 
            padding=self.padding
        )
        
        # Group Normalization
        self.group_norm = nn.GroupNorm(self.num_groups, self.num_gates)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        
        # Batch Normalization for frequency-domain convolution
        self.batch_norm = nn.BatchNorm2d(self.num_gates)
        
        # Depthwise Separable Convolution in the frequency domain
        self.freq_conv = DSConv2d(
            in_channels=2 * (self.input_channels + self.hidden_channels), 
            out_channels=self.num_gates, 
            kernel_size=self.kernel_size, 
            padding=self.padding
        )
        
        # Global Convolution after combining spatial and frequency features
        self.global_conv = nn.Conv2d(
            in_channels=6 * self.hidden_channels, 
            out_channels=3 * self.hidden_channels, 
            kernel_size=self.kernel_size, 
            padding=self.padding
        )
        
        # Optional Squeeze-and-Excitation layer
        if self.use_se:
            self.se_layer = SELayer(6 * self.hidden_channels)
    
    def forward(self, inputs=None, hidden_state=None):
        """
        Forward pass of the FTCGRU cell.
        
        Args:
            inputs (torch.Tensor, optional): Input tensor of shape (sequence_length, batch_size, input_channels, height, width).
                                             If None, a zero tensor is used.
            hidden_state (torch.Tensor, optional): Previous hidden state tensor of shape (batch_size, hidden_channels, height, width).
        
        Returns:
            tuple:
                - torch.Tensor: Output tensor containing all hidden states for each frame, shape (sequence_length, batch_size, hidden_channels, height, width).
                - torch.Tensor: Final hidden state tensor, shape (batch_size, hidden_channels, height, width).
        """
        hidden_state = self._initialize_hidden(inputs, hidden_state)
        output_frames = []

        for t in range(self.num_frames):
            # Get the t-th input frame
            input_t = inputs[t].to(hidden_state.device)

            # Perform a single GRU step
            hidden_state = self._gru_step(input_t, hidden_state)
            output_frames.append(hidden_state)

        # Stack all hidden states across the temporal dimension
        return torch.stack(output_frames), hidden_state  # Return all hidden states and the final hidden state
    
    def _initialize_hidden(self, inputs, hidden_state):
        """
        Initialize the hidden state.
        
        Args:
            inputs (torch.Tensor, optional): Input tensor to determine batch size.
            hidden_state (torch.Tensor, optional): Existing hidden state.
        
        Returns:
            torch.Tensor: Initialized hidden state tensor.
        """
        if hidden_state is not None:
            return hidden_state
        else:   
            batch_size = inputs.size(1)
            height, width = inputs.size(-2), inputs.size(-1)

            # Initialize hidden state with zeros
            hidden_state = torch.zeros(
                batch_size, self.hidden_channels, 
                height, width, 
                device=self.device
            )

            return hidden_state
    
    def _gru_step(self, input_tensor: torch.Tensor, hidden: torch.Tensor):
        """
        Perform a single GRU step with convolutional and frequency-domain operations.
        
        Args:
            input_tensor (torch.Tensor): Input tensor at current time step, shape (batch_size, input_channels, height, width).
            hidden (torch.Tensor): Previous hidden state tensor, shape (batch_size, hidden_channels, height, width).
        
        Returns:
            torch.Tensor: Updated hidden state tensor, shape (batch_size, hidden_channels, height, width).
        """
        # Concatenate input and hidden state along the channel dimension
        combined = torch.cat((input_tensor, hidden), dim=1)
        # Compute gate outputs
        gates = self.conv(combined)
        gates = self.leaky_relu(gates)

        # Fourier transform along the spatial dimensions
        fft_dims = (-2, -1)
        freq_domain = torch.fft.rfftn(combined, dim=fft_dims, norm=self.fourier_norm)
        freq_domain = torch.stack((freq_domain.real, freq_domain.imag), dim=-1)
        
        # Rearrange tensor for convolution
        freq_domain = freq_domain.permute(0, 1, 4, 2, 3).contiguous()
        N, C, _, H, W_freq = freq_domain.size()
        freq_domain = freq_domain.view(N, -1, H, W_freq)
        
        if self.use_se:
            # Apply Squeeze-and-Excitation if enabled
            freq_domain = self.se_layer(freq_domain)
        
        # Apply depthwise separable convolution in the frequency domain
        freq_features = self.freq_conv(freq_domain)
        freq_features = self.batch_norm(freq_features)
        freq_features = self.leaky_relu(freq_features)
        
        # Inverse Fourier transform to return to spatial domain
        ifft_shape = freq_features.shape[-2:]
        freq_complex = torch.complex(freq_features, torch.zeros_like(freq_features))
        spatial_features = torch.fft.irfftn(freq_complex, s=ifft_shape, dim=fft_dims, norm=self.fourier_norm)
        
        # Resize spatial features to match gate outputs
        spatial_features_resized = F.interpolate(
            spatial_features, 
            size=gates.size()[-2:], 
            mode=self.spatial_scale_mode, 
            align_corners=False
        )
        
        # Combine frequency and spatial features
        combined_features = torch.cat((spatial_features_resized, gates), dim=1)
        combined_features = self.global_conv(combined_features)
        combined_features = self.group_norm(combined_features)
        
        # Split combined features into reset, update, and candidate gates
        reset_gate, update_gate, candidate_gate = torch.split(
            combined_features, self.hidden_channels, dim=1
        )
        reset_gate = torch.sigmoid(reset_gate)
        update_gate = torch.sigmoid(update_gate)
        candidate_gate = torch.tanh(candidate_gate)
        
        # Compute the updated hidden state
        updated_hidden = update_gate * hidden + (1 - update_gate) * candidate_gate
        
        return updated_hidden

class singleFrameFTCGRUCell(FTCGRUCell):
    def __init__(self, input_shape, input_channels, hidden_channels, kernel_size, 
                 use_se=False, num_frames=1, device='cuda', fourier_norm='ortho', 
                 spatial_scale_mode='bilinear'):
        super().__init__(input_shape, input_channels, hidden_channels, kernel_size, 
                         use_se, num_frames, device, fourier_norm, spatial_scale_mode)

    def forward(self, input=None, hidden_state=None):
        hidden_state = super()._initialize_hidden(input, hidden_state)
        input = input.to(hidden_state.device)
        assert len(input.shape) == 4, "input diementions number is not 4"
        return super()._gru_step(input, hidden_state)