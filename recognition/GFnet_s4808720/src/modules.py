import torch
import torch.nn as nn
import torch.fft
import math
from timm.models.layers import DropPath, trunc_normal_, to_2tuple
from functools import partial


class FeedForwardBlock(nn.Module):
    """
    Implements a simple Feed-Forward Network (MLP) used inside the FourierBlock.

    Block consists of two fully connected layers with a non-linear activation 
    and dropout for regularisation. Allow network to learn complex 
    feature interactions after global filtering.

    Args:
        dim (int): Input and output feature dimension.
        hidden_dim (int, optional): Hidden layer dimension. Defaults to 4 * dim.
        drop (float): Dropout probability. Defaults to 0.0.
        act_layer (nn.Module): Activation function to use (default: nn.SiLU).
    """

    def __init__(self, dim, hidden_dim=None, drop=0., act_layer=nn.SiLU):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4 
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop) 

    def forward(self, x):
        """
        Forward pass through the feedforward network.
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SpectralFilter(nn.Module):
    """
    Applies a global Fourier-domain filtering operation to model long-range spatial dependencies.

    The input features are transformed into the frequency domain using FFT,
    multiplied element-wise by a learned complex-valued filter, and then
    transformed back to the spatial domain using the inverse FFT.

    Args:
        dim (int): Number of feature channels.
        height (int): Height of the learned Fourier filter.
        width (int): Width of the learned Fourier filter.
        drop_rate (float): Dropout rate applied after filtering.
    """
    def __init__(self, dim, height=14, width=14, drop_rate=0.0):
        super().__init__()
        self.height = height
        self.width = width  
        self.filter = nn.Parameter(torch.randn(height, width, dim, 2) * 0.02) # multiply by 0.02 to keep inital filter close to neutral
        self.drop = nn.Dropout(drop_rate)
        
    def forward(self, x):
        """
        Performs a Fourier transform -> applies learned filter -> inverse transform.
        """
        B, N, C = x.shape 
        side = int(math.sqrt(N))
        x = x.view(B, side, side, C).to(torch.float32) # ensure compatibility with FFT operations

        # foward FFT
        freq = torch.fft.rfft2(x, dim=(1, 2), norm='ortho') # applies 2D FFT over height and width dimensions
        filt = torch.view_as_complex(self.filter) # converts to complex tensor

        # apply frequency filter
        freq = freq * filt

        # inverse FFT
        x =  torch.fft.irfft2(freq, s=(side, side), dim=(1, 2), norm='ortho')
        x = x.view(B, N, C)
        return self.drop(x)


# ChannelAttention



# FourierBlock 



# PatchEmbedding



# GFnetAlzheimer