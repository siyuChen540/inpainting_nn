"""
@author: Siyu Chen
@date: 2025.1.28
@file:recurrent.py
@description:
    This module provides a set of convolutional recurrent neural 
    network (RNN) cells with optional frequency-domain convolution 
    and depthwise-separable convolution.

@Classes:
    BaseFrequencyRNNCell: A base RNN cell that provides common 
        hidden initialization and optional frequency-domain 
        operations for child classes.
    ConvLSTMCell: A convolutional LSTM cell supporting optional 
        frequency-domain convolution.
    ConvGRUCell: A convolutional GRU cell supporting optional 
        frequency-domain convolution.
    ConvGRUCellV2: A convolutional GRU cell variant using 
        depthwise-separable convolution.
    FTCGRUCell: A frequency and temporal convolutional GRU cell 
        with optional SE blocks and depthwise-separable conv.
    SingleFrameFTCGRUCell: A specialized FTCGRUCell that processes 
        only one frame.
"""

import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional, Tuple
from .basic import DSConv2d, SELayer


class BaseFrequencyRNNCell(nn.Module):
    """
    Base RNN cell that includes shared hidden initialization and optional frequency-domain
    operations for child classes.

    Attributes:
        input_channels (int): Number of input channels.
        hidden_channels (int): Number of hidden channels.
        kernel_size (int): Convolution kernel size.
        use_ftc (bool): Whether to use frequency-domain convolution.
        num_frames (int): Number of frames in sequence.
        device (str): Device for computation ('cuda' or 'cpu').
        fourier_norm (str): Normalization to use in Fourier transform.
        spatial_scale_mode (str): Interpolation mode for resizing in frequency domain.
        padding (int): Convolutional padding derived from kernel_size.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int,
        use_ftc: bool = False,
        num_frames: int = 10,
        device: str = "cuda",
        fourier_norm: str = "ortho",
        spatial_scale_mode: str = "bilinear",
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.use_ftc = use_ftc
        self.num_frames = num_frames
        self.device = device
        self.fourier_norm = fourier_norm
        self.spatial_scale_mode = spatial_scale_mode
        self.padding = (kernel_size - 1) // 2

    def _init_hidden_2d(
        self,
        inputs: Optional[torch.Tensor],
        hidden_state: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Initializes a 2D hidden state if none is provided.

        Args:
            inputs (torch.Tensor, optional): Input tensor of shape [T, B, C, H, W].
            hidden_state (torch.Tensor, optional): Hidden tensor of shape [B, hidden_channels, H, W].

        Returns:
            torch.Tensor: Newly initialized or existing hidden state on the correct device.
        """
        if hidden_state is not None:
            return hidden_state.to(self.device)
        if inputs is not None:
            batch_size = inputs.size(1)
            height, width = inputs.size(-2), inputs.size(-1)
        else:
            batch_size, height, width = 1, 1, 1
        return torch.zeros(
            batch_size, self.hidden_channels, height, width, device=self.device
        )

    def _init_hidden_2d_lstm(
        self,
        inputs: Optional[torch.Tensor],
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initializes a 2D LSTM hidden + cell state if none is provided.

        Args:
            inputs (torch.Tensor, optional): Input of shape [T, B, C, H, W].
            hidden_state (tuple, optional): (h, c) states.

        Returns:
            (torch.Tensor, torch.Tensor): LSTM hidden and cell states.
        """
        if hidden_state is not None:
            return (
                hidden_state[0].to(self.device),
                hidden_state[1].to(self.device),
            )
        if inputs is not None:
            batch_size = inputs.size(1)
            height, width = inputs.size(-2), inputs.size(-1)
        else:
            batch_size, height, width = 1, 1, 1
        hx = torch.zeros(
            batch_size, self.hidden_channels, height, width, device=self.device
        )
        cx = torch.zeros_like(hx)
        return hx, cx

    def _apply_frequency_convolution(
        self, combined: torch.Tensor, conv_module: nn.Module, gates_out: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies frequency-domain convolution to the concatenated input and hidden state.

        Args:
            combined (torch.Tensor): Combined input + hidden of shape [B, C, H, W].
            conv_module (nn.Module): Frequency convolution module (semi_conv).
            gates_out (torch.Tensor): Current gate output in spatial domain.

        Returns:
            torch.Tensor: Updated gate outputs after frequency operation and global convolution.
        """
        with torch.no_grad() if not self.use_ftc else torch.enable_grad():
            fft_dim = (-2, -1)
            freq = torch.fft.rfftn(combined, dim=fft_dim, norm=self.fourier_norm)
            freq = torch.stack((freq.real, freq.imag), dim=-1)
            freq = freq.permute(0, 1, 4, 2, 3).contiguous()  # [B, C, 2, H, W//2+1]
            bsz, chn, _, h, w2 = freq.size()
            freq = freq.view(bsz, -1, h, w2)  # merge channel & complex dim
            ffc_out = conv_module(freq)
            ifft_shape = ffc_out.shape[-2:]
            ffc_out = torch.fft.irfftn(
                torch.complex(ffc_out, torch.zeros_like(ffc_out)),
                s=ifft_shape,
                dim=fft_dim,
                norm=self.fourier_norm,
            )
            ffc_out_resize = F.interpolate(
                ffc_out,
                size=gates_out.size()[-2:],
                mode=self.spatial_scale_mode,
                align_corners=False,
            )
            return ffc_out_resize
        return gates_out


class ConvLSTMCell(BaseFrequencyRNNCell):
    """
    Convolutional LSTM cell with optional frequency-domain convolution.

    The cell follows these equations:
        i_t = sigmoid(W_i * [x_t, h_{t-1}])
        f_t = sigmoid(W_f * [x_t, h_{t-1}])
        g_t = tanh   (W_g * [x_t, h_{t-1}])
        o_t = sigmoid(W_o * [x_t, h_{t-1}])
        c_t = f_t * c_{t-1} + i_t * g_t
        h_t = o_t * tanh(c_t)
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int,
        use_ftc: bool = True,
        num_frames: int = 10,
        device: str = "cuda",
        fourier_norm: str = "ortho",
        spatial_scale_mode: str = "bilinear",
    ) -> None:
        super().__init__(
            input_channels,
            hidden_channels,
            kernel_size,
            use_ftc,
            num_frames,
            device,
            fourier_norm,
            spatial_scale_mode,
        )
        groups_num = max(1, (4 * self.hidden_channels) // 4)
        channel_num = 4 * self.hidden_channels

        self.conv = nn.Sequential(
            nn.Conv2d(
                self.input_channels + self.hidden_channels,
                channel_num,
                self.kernel_size,
                padding=self.padding,
            ),
            nn.GroupNorm(groups_num, channel_num),
        )

        if self.use_ftc:
            self.semi_conv = nn.Sequential(
                nn.Conv2d(
                    2 * (self.input_channels + self.hidden_channels),
                    channel_num,
                    self.kernel_size,
                    padding=self.padding,
                ),
                nn.GroupNorm(groups_num, channel_num),
                nn.LeakyReLU(inplace=True),
            )
            self.global_conv = nn.Sequential(
                nn.Conv2d(
                    8 * self.hidden_channels,
                    4 * self.hidden_channels,
                    self.kernel_size,
                    padding=self.padding,
                ),
                nn.GroupNorm(groups_num, channel_num),
            )

    def forward(
        self,
        inputs: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of ConvLSTM.

        Args:
            inputs (torch.Tensor): [T, B, C, H, W] input sequence.
            hidden_state (tuple, optional): (h, c) states.

        Returns:
            (torch.Tensor, (torch.Tensor, torch.Tensor)):
                Stacked hidden states for each timestep, final (h, c).
        """
        hx, cx = self._init_hidden_2d_lstm(inputs, hidden_state)
        outputs = []

        for t in range(self.num_frames):
            x_t = inputs[t].to(hx.device)
            hx, cx = self._step(x_t, hx, cx)
            outputs.append(hx)

        return torch.stack(outputs), (hx, cx)

    def _step(
        self,
        x: torch.Tensor,
        hx: torch.Tensor,
        cx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single LSTM step.

        Args:
            x (torch.Tensor): [B, C, H, W] input at time t.
            hx (torch.Tensor): [B, hidden_channels, H, W] previous hidden state.
            cx (torch.Tensor): [B, hidden_channels, H, W] previous cell state.

        Returns:
            (hx, cx): Updated LSTM hidden and cell state.
        """
        combined = torch.cat([x, hx], dim=1)
        gates_out = self.conv(combined)

        if self.use_ftc:
            freq_out = self._apply_frequency_convolution(
                combined, self.semi_conv, gates_out
            )
            gates_out = torch.cat([freq_out, gates_out], dim=1)
            gates_out = self.global_conv(gates_out)

        in_gate, forget_gate, cell_gate, out_gate = torch.split(
            gates_out, self.hidden_channels, dim=1
        )
        in_gate = torch.sigmoid(in_gate)
        forget_gate = torch.sigmoid(forget_gate)
        cell_gate = torch.tanh(cell_gate)
        out_gate = torch.sigmoid(out_gate)

        cx_new = forget_gate * cx + in_gate * cell_gate
        hx_new = out_gate * torch.tanh(cx_new)
        return hx_new, cx_new


class ConvGRUCell(BaseFrequencyRNNCell):
    """
    Convolutional GRU cell with optional frequency-domain convolution.
    GRU includes reset, update, and candidate gates.

    The cell follows:
        r_t = sigmoid(W_r * [x_t, h_{t-1}])
        z_t = sigmoid(W_z * [x_t, h_{t-1}])
        n_t = tanh   (W_n * [x_t, r_t * h_{t-1}])
        h_t = z_t * h_{t-1} + (1 - z_t) * n_t
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int,
        use_ftc: bool = True,
        num_frames: int = 10,
        device: str = "cuda",
        fourier_norm: str = "ortho",
        spatial_scale_mode: str = "bilinear",
    ) -> None:
        super().__init__(
            input_channels,
            hidden_channels,
            kernel_size,
            use_ftc,
            num_frames,
            device,
            fourier_norm,
            spatial_scale_mode,
        )
        groups_num = max(1, self.hidden_channels)
        channel_num = 3 * self.hidden_channels

        self.conv = nn.Sequential(
            nn.Conv2d(
                self.input_channels + self.hidden_channels,
                channel_num,
                self.kernel_size,
                padding=self.padding,
            ),
            nn.GroupNorm(groups_num, channel_num),
        )

        if self.use_ftc:
            self.semi_conv = nn.Sequential(
                nn.Conv2d(
                    2 * (self.input_channels + self.hidden_channels),
                    channel_num,
                    self.kernel_size,
                    padding=self.padding,
                ),
                nn.GroupNorm(groups_num, channel_num),
                nn.LeakyReLU(inplace=True),
            )
            self.global_conv = nn.Sequential(
                nn.Conv2d(
                    6 * self.hidden_channels,
                    3 * self.hidden_channels,
                    self.kernel_size,
                    padding=self.padding,
                ),
                nn.GroupNorm(groups_num, channel_num),
            )

    def forward(
        self,
        inputs: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of ConvGRU.

        Args:
            inputs (torch.Tensor): [T, B, C, H, W] input sequence.
            hidden_state (torch.Tensor, optional): [B, hidden_channels, H, W].

        Returns:
            (torch.Tensor, torch.Tensor): (stacked hidden states, final hidden state).
        """
        hx = self._init_hidden_2d(inputs, hidden_state)
        outputs = []

        for t in range(self.num_frames):
            x_t = inputs[t].to(hx.device)
            hx = self._step(x_t, hx)
            outputs.append(hx)

        return torch.stack(outputs), hx

    def _step(
        self,
        x: torch.Tensor,
        hx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Single GRU step.

        Args:
            x (torch.Tensor): [B, C, H, W] input at time t.
            hx (torch.Tensor): [B, hidden_channels, H, W] previous hidden state.

        Returns:
            torch.Tensor: Updated hidden state.
        """
        combined = torch.cat([x, hx], dim=1)
        gates_out = self.conv(combined)

        if self.use_ftc:
            freq_out = self._apply_frequency_convolution(
                combined, self.semi_conv, gates_out
            )
            gates_out = torch.cat([freq_out, gates_out], dim=1)
            gates_out = self.global_conv(gates_out)

        reset_gate, update_gate, candidate_gate = torch.split(
            gates_out, self.hidden_channels, dim=1
        )
        reset_gate = torch.sigmoid(reset_gate)
        update_gate = torch.sigmoid(update_gate)
        candidate_gate = torch.tanh(candidate_gate)
        return update_gate * hx + (1 - update_gate) * candidate_gate


class ConvGRUCellV2(ConvGRUCell):
    """
    A variant of the ConvGRU cell that uses depthwise-separable convolution.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int,
        use_se: bool = False,
        num_frames: int = 10,
        device: str = "cuda",
        fourier_norm: str = "ortho",
        spatial_scale_mode: str = "bilinear",
        use_ftc: bool = False,
    ) -> None:
        super().__init__(
            input_channels,
            hidden_channels,
            kernel_size,
            use_ftc,
            num_frames,
            device,
            fourier_norm,
            spatial_scale_mode,
        )
        self.use_se = use_se
        groups_num = max(1, self.hidden_channels)
        channel_num = 3 * self.hidden_channels

        self.conv = nn.Sequential(
            DSConv2d(
                self.input_channels + self.hidden_channels,
                channel_num,
                self.kernel_size,
                padding=self.padding,
            ),
            nn.GroupNorm(groups_num, channel_num),
        )

        if self.use_ftc:
            self.semi_conv = nn.Sequential(
                DSConv2d(
                    2 * (self.input_channels + self.hidden_channels),
                    channel_num,
                    self.kernel_size,
                    padding=self.padding,
                ),
                nn.GroupNorm(groups_num, channel_num),
                nn.LeakyReLU(inplace=True),
            )
            self.global_conv = nn.Sequential(
                DSConv2d(
                    6 * self.hidden_channels,
                    3 * self.hidden_channels,
                    self.kernel_size,
                    padding=self.padding,
                ),
                nn.GroupNorm(groups_num, channel_num),
            )


class FTCGRUCell(BaseFrequencyRNNCell):
    """
    Frequency and Temporal Convolutional GRU (FTCGRU) Cell with optional Squeeze-and-Excitation
    and depthwise-separable convolution. It combines spatial and frequency-domain operations
    to update the hidden state.

    Attributes:
        use_se (bool): Whether to use a Squeeze-and-Excitation layer.
        freq_conv (nn.Module): Depthwise-separable convolution used in frequency domain.
        global_conv (nn.Module): Global convolution after combining frequency and spatial features.
        group_norm (nn.GroupNorm): Group normalization applied to gates.
        batch_norm (nn.BatchNorm2d): Batch normalization applied to frequency features.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int,
        use_se: bool = False,
        num_frames: int = 10,
        device: str = "cuda",
        fourier_norm: str = "ortho",
        spatial_scale_mode: str = "bilinear",
    ) -> None:
        super().__init__(
            input_channels,
            hidden_channels,
            kernel_size,
            use_ftc=True,  # We always do frequency conv here
            num_frames=num_frames,
            device=device,
            fourier_norm=fourier_norm,
            spatial_scale_mode=spatial_scale_mode,
        )
        self.use_se = use_se

        self.num_gates = 3 * self.hidden_channels
        self.num_groups = max(1, self.hidden_channels)

        self.conv = nn.Conv2d(
            in_channels=self.input_channels + self.hidden_channels,
            out_channels=self.num_gates,
            kernel_size=self.kernel_size,
            padding=self.padding,
        )
        self.group_norm = nn.GroupNorm(self.num_groups, self.num_gates)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.batch_norm = nn.BatchNorm2d(self.num_gates)

        self.freq_conv = DSConv2d(
            in_channels=2 * (self.input_channels + self.hidden_channels),
            out_channels=self.num_gates,
            kernel_size=self.kernel_size,
            padding=self.padding,
        )
        self.global_conv = nn.Conv2d(
            in_channels=6 * self.hidden_channels,
            out_channels=self.num_gates,
            kernel_size=self.kernel_size,
            padding=self.padding,
        )
        if self.use_se:
            self.se_layer = SELayer(6 * self.hidden_channels)

    def forward(
        self,
        inputs: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass over T frames. Returns all hidden states and final hidden.

        Args:
            inputs (torch.Tensor): [T, B, C, H, W] input sequence.
            hidden_state (torch.Tensor, optional): [B, hidden_channels, H, W] init hidden state.

        Returns:
            (torch.Tensor, torch.Tensor): (stacked all hidden states, final hidden state).
        """
        hx = self._init_hidden_2d(inputs, hidden_state)
        outputs = []

        for t in range(self.num_frames):
            x_t = inputs[t].to(hx.device)
            hx = self._step(x_t, hx)
            outputs.append(hx)

        return torch.stack(outputs), hx

    def _step(
        self,
        x: torch.Tensor,
        hx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Single step update of the FTCGRU.

        Args:
            x (torch.Tensor): [B, C, H, W] input at time t.
            hx (torch.Tensor): Hidden state from previous step.

        Returns:
            torch.Tensor: Updated hidden state at time t.
        """
        combined = torch.cat((x, hx), dim=1)
        gates_out = self.conv(combined)
        gates_out = self.leaky_relu(gates_out)

        # Frequency transform
        fft_dim = (-2, -1)
        freq_domain = torch.fft.rfftn(combined, dim=fft_dim, norm=self.fourier_norm)
        freq_domain = torch.stack((freq_domain.real, freq_domain.imag), dim=-1)
        freq_domain = freq_domain.permute(0, 1, 4, 2, 3).contiguous()
        bsz, chn, _, h, w2 = freq_domain.size()
        freq_domain = freq_domain.view(bsz, -1, h, w2)

        if self.use_se:
            # Optionally apply SE in frequency space
            freq_domain = self.se_layer(freq_domain)

        freq_features = self.freq_conv(freq_domain)
        freq_features = self.batch_norm(freq_features)
        freq_features = self.leaky_relu(freq_features)

        ifft_shape = freq_features.shape[-2:]
        freq_complex = torch.complex(freq_features, torch.zeros_like(freq_features))
        spatial_features = torch.fft.irfftn(
            freq_complex, s=ifft_shape, dim=fft_dim, norm=self.fourier_norm
        )
        spatial_features_resized = F.interpolate(
            spatial_features,
            size=gates_out.size()[-2:],
            mode=self.spatial_scale_mode,
            align_corners=False,
        )

        combined_features = torch.cat((spatial_features_resized, gates_out), dim=1)
        combined_features = self.global_conv(combined_features)
        combined_features = self.group_norm(combined_features)

        reset_gate, update_gate, candidate_gate = torch.split(
            combined_features, self.hidden_channels, dim=1
        )
        reset_gate = torch.sigmoid(reset_gate)
        update_gate = torch.sigmoid(update_gate)
        candidate_gate = torch.tanh(candidate_gate)

        out = update_gate * hx + (1 - update_gate) * candidate_gate
        return out


class SingleFrameFTCGRUCell(FTCGRUCell):
    """
    A specialized FTCGRUCell that processes only a single frame (num_frames=1).
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int,
        use_se: bool = False,
        num_frames: int = 1,
        device: str = "cuda",
        fourier_norm: str = "ortho",
        spatial_scale_mode: str = "bilinear",
    ) -> None:
        super().__init__(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            use_se=use_se,
            num_frames=num_frames,
            device=device,
            fourier_norm=fourier_norm,
            spatial_scale_mode=spatial_scale_mode,
        )

    def forward(
        self,
        input_tensor: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for a single frame. Returns the updated hidden state.

        Args:
            input_tensor (torch.Tensor): [B, C, H, W] input.
            hidden_state (torch.Tensor, optional): [B, hidden_channels, H, W].

        Returns:
            torch.Tensor: Updated hidden state.
        """
        hx = self._init_hidden_2d(input_tensor.unsqueeze(0), hidden_state)
        input_tensor = input_tensor.to(hx.device)
        assert len(input_tensor.shape) == 4, "Input must be [B, C, H, W]."
        return self._step(input_tensor, hx)
