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


# SpetralFilter



# ChannelAttention



# FourierBlock 



# PatchEmbedding



# GFnetAlzheimer