import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

NewAxis = None

def zero_module(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

def root_mean_squares(x: Tensor) -> Tensor:
    x_squared = x ** 2
    rms = torch.sqrt(x_squared.mean(dim=-1, keepdim=True))
    return rms

class RMSNorm(nn.Module):
    def __init__(self, model_dim: int, eps: float = 1e-8):
        super().__init__()

        self.scale = nn.parameter.Parameter(
            torch.ones(model_dim)[NewAxis, NewAxis, :]
        )
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        rms = root_mean_squares(x)
        x_rms = x / (rms + self.eps)
        out = x_rms * self.scale
        return out

class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()

        self.linear_1 = nn.Linear(dim_in, dim_out)
        self.linear_2 = nn.Linear(dim_in, dim_out)
    
    def forward(self, x: Tensor) -> Tensor:
        out_1 = self.linear_1(x)
        out_2 = self.linear_2(x)
        out = F.gelu(out_1) * out_2
        return out

class FFN(nn.Module):
    def __init__(self, model_dim: int, ffn_dim: int, dropout_p: float = 0.0):
        super().__init__()

        self.layers = nn.Sequential(
            GEGLU(model_dim, ffn_dim),
            nn.Dropout(dropout_p),
            zero_module(nn.Linear(ffn_dim, model_dim)),
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.layers(x)
        out = out + x
        return out
