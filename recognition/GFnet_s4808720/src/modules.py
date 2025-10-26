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


class ChannelAttention(nn.Module):
    """
    Lightweight attention mechanism that adaptively reweights feature channels.

    It computes global channel statistics using average pooling, processes them
    through a small MLP, and rescales the input tensor accordingly.

    Args:
        dim (int): Number of input channels.
        reduction (int): Reduction ratio for bottleneck in the attention MLP.
    """

    def __init__(self, dim, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False), # reduce dimensionality 
            nn.SiLU(),                                    # apply non-linear activation 
            nn.Linear(dim // reduction, dim, bias=False), # restore dimensionality
            nn.Sigmoid()                                  # scale to [0, 1] for attention scaling
        )

    def forward(self, x):
        """
        Forward pass: computes channel attention and applies reweighting.
        """
        # compute mean across spatial dimensions
        # pass describtor through MLP to get attention weights per channel
        # scale input tensor by the learned attention weights
        return x * self.fc(x.mean(dim=1, keepdim=True))


# FourierBlock 
class FourierBlock(nn.Module):
    """
    A single processing block of GFNet combining normalisation, global Fourier filtering, optional channel attention, and a feed-forward MLP.

    This block functions similarly to a Transformer encoder but replaces self-attention with a frequency-domain global filter.

    Args:
        dim (int): Feature dimension of the input.
        mlp_ratio (float): Hidden dimension ratio for the feedforward MLP.
        drop (float): Dropout rate for MLP and filtering.
        drop_path (float): Stochastic depth rate for residual connections.
        norm_layer (nn.Module): Normalisation layer (default: LayerNorm).
        use_attention (bool): Whether to apply channel attention after filtering.
    """

    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., 
                norm_layer=nn.LayerNorm, use_attention=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = SpectralFilter(dim)

        # optional channel attention mechanism
        self.attn = ChannelAttention(dim) if use_attention else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = FeedForwardBlock(dim, hidden_dim, drop=drop)

    def forward(self, x):
        """
        Forward pass: applies normalization -> global filtering -> attention -> MLP,
        with residual connections between each sublayer.
        """

        # apply norm -> fourier filter -> (optional) channel attention
        # then add residual connection
        x = x + self.drop_path(self.attn(self.filter(self.norm1(x))))

        # apply norm -> feedforward MLP
        # then add another residual connection with possible stochastic drop
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        # return the enhanced feature map
        return x


# PatchEmbedding



# GFnetAlzheimer