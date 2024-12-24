#########################
# Model Building
#########################


import torch
import torch.nn.functional as F
from torch import nn

class DSConv2d(nn.Module):
    """
    Depthwise Separable Convolution.
    First performs a depthwise convolution, then a pointwise convolution.
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding=1, bias=False):
        super(DSConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
    
class SELayer(nn.Module):
    """
    Squeeze-and-Excitation (SE) Layer.
    
    This layer adaptively recalibrates channel-wise feature responses by explicitly modeling interdependencies between channels.
    
    Args:
        num_channels (int): Number of input channels.
        reduction_ratio (int, optional): Reduction ratio for the bottleneck. Default is 16.
    """
    def __init__(self, num_channels, reduction_ratio=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels // reduction_ratio, num_channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        batch_size, num_channels, _, _ = x.size()
        squeeze = self.avg_pool(x).view(batch_size, num_channels)
        excitation = self.fc(squeeze).view(batch_size, num_channels, 1, 1)
        scaled = x * excitation.expand_as(x)
        return scaled 

def mkLayer(block_params: dict) -> nn.Sequential:
    """
    Given an OrderedDict-like structure of layer parameters, build a nn.Sequential block.
    The keys should contain layer type hints like 'conv', 'pool', 'deconv', 'relu', 'leaky'.
    """
    layers = []
    for layer_name, params in block_params.items():
        if 'pool' in layer_name:
            # expected params: [kernel_size, stride, padding]
            k, s, p = params
            layer = nn.MaxPool2d(kernel_size=k, stride=s, padding=p)
            layers.append((layer_name, layer))
        elif 'deconv' in layer_name:
            # expected params: [inC, outC, kernel_size, stride, padding]
            inC, outC, k, st, pad = params
            layer = nn.ConvTranspose2d(inC, outC, k, stride=st, padding=pad)
            layers.append((layer_name, layer))
        elif 'conv' in layer_name:
            # expected params: [inC, outC, kernel_size, stride, padding]
            inC, outC, k, st, pad = params
            layer = nn.Conv2d(inC, outC, k, stride=st, padding=pad)
            layers.append((layer_name, layer))
        else:
            raise NotImplementedError(f"Layer type not recognized in: {layer_name}")

        # Add activation if hinted in the layer name
        if 'relu' in layer_name:
            layers.append((f'relu_{layer_name}', nn.ReLU(inplace=True)))
        elif 'leaky' in layer_name:
            layers.append((f'leaky_{layer_name}', nn.LeakyReLU(negative_slope=0.2, inplace=True)))

    # Extract layers without names for nn.Sequential
    return nn.Sequential(*[layer for _, layer in layers])


class ConvLSTM_cell(nn.Module):
    """
    A ConvLSTM cell with optional frequency-domain convolution (fconv).
    """
    def __init__(self, shape, channels, kernel_size, features_num, fconv=True, frames_len=10, device='cuda'):
        super().__init__()
        self.shape = shape
        self.channels = channels
        self.features_num = features_num
        self.kernel_size = kernel_size
        self.fconv = fconv
        self.padding = (kernel_size - 1) // 2
        self.frames_len = frames_len
        self.device = device

        groups_num = max(1, (4 * self.features_num) // 4)
        channel_num = 4 * self.features_num

        self.conv = nn.Sequential(
            nn.Conv2d(self.channels + self.features_num, channel_num, self.kernel_size, padding=self.padding),
            nn.GroupNorm(groups_num, channel_num)
        )

        if fconv:
            self.semi_conv = nn.Sequential(
                nn.Conv2d(2 * (self.channels + self.features_num), channel_num, self.kernel_size, padding=self.padding),
                nn.GroupNorm(groups_num, channel_num),
                nn.LeakyReLU(inplace=True)
            )
            self.global_conv = nn.Sequential(
                nn.Conv2d(8 * self.features_num, 4 * self.features_num, self.kernel_size, padding=self.padding),
                nn.GroupNorm(groups_num, channel_num)
            )

    def forward(self, inputs=None, hidden_state=None):
        hx, cx = self._init_hidden(inputs, hidden_state)
        output_frames = []

        for t in range(self.frames_len):
            if inputs is None:
                x = torch.zeros(hx.size(0), self.channels, self.shape[0], self.shape[1], device=hx.device)
            else:
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

        # Optional frequency domain convolution
        if self.fconv:
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

        in_gate, forget_gate, hat_cell_gate, out_gate = torch.split(gates_out, self.features_num, dim=1)
        in_gate = torch.sigmoid(in_gate)
        forget_gate = torch.sigmoid(forget_gate)
        hat_cell_gate = torch.tanh(hat_cell_gate)
        out_gate = torch.sigmoid(out_gate)

        cy = (forget_gate * cx) + (in_gate * hat_cell_gate)
        hy = out_gate * torch.tanh(cy)

        return hy, cy


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
    def __init__(self, shape, channels, kernel_size, features_num, fconv=True, frames_len=10, device='cuda'):
        super().__init__()
        self.shape = shape
        self.channels = channels
        self.features_num = features_num
        self.kernel_size = kernel_size
        self.fconv = fconv
        self.padding = (kernel_size - 1) // 2
        self.frames_len = frames_len
        self.device = device

        groups_num = max(1, self.features_num)  # 3 gates for GRU
        channel_num = 3 * self.features_num  # GRU has 3 gates: update, reset, candidate

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

    def forward(self, inputs=None, hidden_state=None):
        hx = self._init_hidden(inputs, hidden_state)
        output_frames = []

        for t in range(self.frames_len):
            if inputs is None:
                x = torch.zeros(hx.size(0), self.channels, self.shape[0], self.shape[1], device=hx.device)
            else:
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

        # Split gates: [reset, update, candidate]
        reset_gate, update_gate, candidate_gate = torch.split(gates_out, self.features_num, dim=1)
        reset_gate = torch.sigmoid(reset_gate)
        update_gate = torch.sigmoid(update_gate)
        candidate_gate = torch.tanh(candidate_gate)

        # Compute hidden state update
        hy = update_gate * hx + (1 - update_gate) * candidate_gate

        return hy

class ConvGRU_cell_v2(ConvGRU_cell):
    """
    A ConvGRU cell with optional frequency-domain convolution (fconv) and in deep wise separable convolution.
    """
    def __init__(self, shape, channels, kernel_size, features_num, fconv=True, frames_len=10, device='cuda'):
        super().__init__(shape, channels, kernel_size, features_num, fconv, frames_len, device)
        self.shape = shape
        self.channels = channels
        self.features_num = features_num
        self.kernel_size = kernel_size
        self.fconv = fconv
        self.padding = (kernel_size - 1) // 2
        self.frames_len = frames_len
        self.device = device

        groups_num = max(1, self.features_num)  # 3 gates for GRU
        channel_num = 3 * self.features_num  # GRU has 3 gates: update, reset, candidate

        # Standard convolution for GRU gates (update, reset, candidate)
        self.conv = nn.Sequential(
            DSConv2d(self.channels + self.features_num, channel_num, self.kernel_size, padding=self.padding),
            nn.GroupNorm(groups_num, channel_num)
        )

        # Optional frequency-domain convolution for fconv=True
        if fconv:
            self.semi_conv = nn.Sequential(
                DSConv2d(2 * (self.channels + self.features_num), channel_num, self.kernel_size, padding=self.padding),
                nn.GroupNorm(groups_num, channel_num),
                nn.LeakyReLU(inplace=True)
            )
            self.global_conv = nn.Sequential(
                DSConv2d(6 * self.features_num, 3 * self.features_num, self.kernel_size, padding=self.padding),
                nn.GroupNorm(groups_num, channel_num)
            )


class FTCGRUCell(nn.Module):
    """
    Frequency and Temporal Convolutional Gated Recurrent Unit (FTCGRU) Cell.
    
    This cell integrates convolutional operations in the frequency domain and utilizes depthwise separable convolutions optionally enhanced with Squeeze-and-Excitation (SE) blocks.
    
    Args:
        input_shape (tuple): Spatial dimensions of the input feature map (height, width).
        input_channels (int): Number of input channels.
        hidden_channels (int): Number of hidden channels in the GRU cell.
        kernel_size (int or tuple): Size of the convolutional kernel.
        use_se (bool, optional): Whether to use Squeeze-and-Excitation blocks. Default is False.
        num_frames (int, optional): Number of frames to process (sequence length). Default is 10.
        device (str, optional): Device to perform computations on ('cuda' or 'cpu'). Default is 'cuda'.
        fourier_norm (str, optional): Normalization method for Fourier transforms. Default is 'ortho'.
        spatial_scale_mode (str, optional): Interpolation mode for spatial scaling. Default is 'bilinear'.
    """
    def __init__(self, input_shape, input_channels, hidden_channels, kernel_size, 
                 use_se=False, num_frames=10, device='cuda', fourier_norm='ortho', 
                 spatial_scale_mode='bilinear'):
        super(FTCGRUCell, self).__init__()
        self.input_shape = input_shape  # (height, width)
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
        hidden = self._initialize_hidden(inputs, hidden_state)
        output_frames = []

        for t in range(self.num_frames):
            if inputs is None:
                # Use zero tensor if no input is provided
                input_t = torch.zeros(
                    hidden.size(0), self.input_channels, 
                    self.input_shape[0], self.input_shape[1], 
                    device=hidden.device
                )
            else:
                # Get the t-th input frame
                input_t = inputs[t].to(hidden.device)

            # Perform a single GRU step
            hidden = self._gru_step(input_t, hidden)
            output_frames.append(hidden)

        # Stack all hidden states across the temporal dimension
        return torch.stack(output_frames), hidden  # Return all hidden states and the final hidden state
    
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

        if inputs is not None:
            batch_size = inputs.size(1)
        else:
            batch_size = 1

        # Initialize hidden state with zeros
        hidden = torch.zeros(
            batch_size, self.hidden_channels, 
            self.input_shape[0], self.input_shape[1], 
            device=self.device
        )
        return hidden
    
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


class Encoder(nn.Module):
    """
    Encoder composed of multiple Conv/Pool layers and ConvLSTM cells in sequence.
    Optimized for reduced memory consumption.
    """
    def __init__(self, child_nets_params, convlstm_cells):
        super().__init__()
        assert len(child_nets_params) == len(convlstm_cells)
        self.block_num = len(child_nets_params)
        self.child_cells = nn.ModuleList([mkLayer(params) for params in child_nets_params])
        self.convlstm_cells = nn.ModuleList(convlstm_cells)

    def forward(self, inputs: torch.Tensor):
        # inputs shape: (batch, frames, C, H, W) -> (frames, batch, C, H, W)
        inputs = inputs.transpose(0, 1)
        hidden_states = []

        for i in range(self.block_num):
            # Removed flip operation for memory optimization
            # Apply child cell (conv)
            fnum, bsz, ch, h, w = inputs.size()
            reshaped = inputs.reshape(-1, ch, h, w)
            processed = self.child_cells[i](reshaped)
            _, nch, nh, nw = processed.size()
            processed = processed.view(fnum, bsz, nch, nh, nw)

            # Apply ConvLSTM
            outputs, state_stage = self.convlstm_cells[i](processed, None)
            hidden_states.append(state_stage)
            inputs = outputs

        return tuple(hidden_states)

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

class ED(nn.Module):
    """
    ED model: Encoder-Decoder architecture that first encodes input frames 
    into hidden states and then decodes them back into predicted frames.
    """
    def __init__(self, encoder, decoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, inputs):
        state_list = self.encoder(inputs)
        output = self.decoder(state_list)
        return output

def test_model_cost():
    from collections import OrderedDict
    channels = [4, 8, 16]
    shape_scale = (1440, 2040)
    fconv = True
    frames = 3
    device = 'cuda'
    c_1, c_2, c_3 = channels            # [4,8,16]
    shapes_w, shapes_h = shape_scale    # [48,48]
    
    # Optimized Encoder and Decoder Parameters
    encoder_params = [
        [
            OrderedDict({'conv1_leaky_1': [1,   c_1, 3, 1, 1]}),
            OrderedDict({'conv2_leaky_1': [c_2, c_2, 3, 2, 1]}),
            OrderedDict({'conv3_leaky_1': [c_3, c_3, 3, 2, 1]}),
        ],
        [   
            ConvGRU_cell(
                shape=(shapes_w,   shapes_h),    channels=c_1, 
                kernel_size=3, features_num=c_2, fconv=fconv, 
                frames_len=frames, device=device),
            ConvGRU_cell(
                shape=(shapes_w//2,shapes_h//2), channels=c_2, 
                kernel_size=3, features_num=c_3, fconv=fconv, 
                frames_len=frames, device=device),
            ConvGRU_cell(
                shape=(shapes_w//4,shapes_h//4), channels=c_3, 
                kernel_size=3, features_num=c_3, fconv=fconv, 
                frames_len=frames, device=device)
        ]
    ]

    decoder_params = [
        [
            OrderedDict({'deconv1_leaky_1': [c_3, c_3, 4, 2, 1]}),
            OrderedDict({'deconv2_leaky_1': [c_3, c_3, 4, 2, 1]}),
            OrderedDict({                                       
                'conv3_leaky_1': [c_2, c_1, 3, 1, 1],
                'conv4_leaky_1': [c_1,   1, 1, 1, 0]
            }),
        ],
        [
            ConvGRU_cell(
                shape=(shapes_w//4,shapes_h//4), channels=c_3, 
                kernel_size=3, features_num=c_3, fconv=fconv, 
                frames_len=frames, device=device),
            ConvGRU_cell(
                shape=(shapes_w//2,shapes_h//2), channels=c_3, 
                kernel_size=3, features_num=c_3, fconv=fconv, 
                frames_len=frames, device=device),
            ConvGRU_cell(
                shape=(shapes_w,   shapes_h),    channels=c_3, 
                kernel_size=3, features_num=c_2, fconv=fconv, 
                frames_len=frames, device=device),
        ]
    ]

    # Initialize Encoder and Decoder
    encoder = Encoder(encoder_params[0], encoder_params[1]).to(device)
    decoder = Decoder(decoder_params[0], decoder_params[1]).to(device)

    # Initialize the ED model
    model = ED(encoder, decoder).to(device)
    x = torch.randn(1, frames, 1, shapes_w, shapes_h).to(device)
    y = model(x)
    print("gpu memory cost of ED model: {:.2f} GB".format(torch.cuda.memory_reserved(device) / 1024 / 1024 / 1024))
    print("Memory cost of ED model: {:.2f} MB".format(sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024))
    
def test_deepwise_separable_conv():
    model = DSConv2d(3, 64, 3, padding=1)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)
    
def test_convgru_cell_2():
    model = ConvGRU_cell_v2((224, 224), 3, 3, 64, fconv=True, frames_len=10, device=True)
    x = torch.randn(10, 1, 3, 224, 224)
    y, _ = model(x)
    print(y.shape)
    print(_.shape)

def compare_gpu_memory_cost_of_convgru_cell_and_convlstm_cell_and_convgru_cell_v2():
    # Compare memory cost of ConvGRU_cell and ConvLSTM_cell
    divice='cuda'
    model = ConvGRU_cell((224, 224), 3, 3, 64, frames_len=10).to(divice)
    x = torch.randn(10, 1, 3, 224, 224)
    y, _ = model(x)
    print("Memory cost of ConvGRU_cell: {:.2f} MB".format(sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024))
    # clear torch cache
    del model, x, y, _
    torch.cuda.empty_cache()

    
    model = ConvLSTM_cell((224, 224), 3, 3, 64, frames_len=10).to(divice)
    x = torch.randn(10, 1, 3, 224, 224)
    y, _ = model(x)
    print("Memory cost of ConvLSTM_cell: {:.2f} MB".format(sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024))
    # clear torch cache
    del model, x, y, _
    torch.cuda.empty_cache()


    # Compare memory cost of ConvGRU_cell_v2 and ConvGRU_cell
    model = ConvGRU_cell_v2((224, 224), 3, 3, 64, frames_len=10).to(divice)
    x = torch.randn(10, 1, 3, 224, 224)
    y, _ = model(x)
    print("Memory cost of ConvGRU_cell_v2: {:.2f} MB".format(sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024))


if __name__ == '__main__':
    # test_deepwise_separable_conv()
    # test_convgru_cell_2()
    # compare_gpu_memory_cost_of_convgru_cell_and_convlstm_cell_and_convgru_cell_v2()
    test_model_cost()