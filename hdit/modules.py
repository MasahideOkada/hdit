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

# adopted from https://github.com/huggingface/pytorch-image-models/blob/3234daf783a78014e5ca2215ea41c4b7c3380517/timm/layers/patch_embed.py#L25
class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int):
        super().__init__()

        self.proj = nn.Conv2d(in_channels, embed_dim, patch_size, stride=patch_size)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        x = x.flatten(start_dim=-2).transpose(-2, -1)
        return x

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()

        self.scale = nn.parameter.Parameter(
            torch.ones(dim)[NewAxis, NewAxis, :]
        )
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        rms = root_mean_squares(x)
        x_rms = x / (rms + self.eps)
        out = x_rms * self.scale
        return out

class GEGLU(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        self.linear_1 = nn.Linear(in_dim, out_dim)
        self.linear_2 = nn.Linear(in_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        out_1 = self.linear_1(x)
        out_2 = self.linear_2(x)
        out = F.gelu(out_1) * out_2
        return out

class FFN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout_p: float = 0.0):
        super().__init__()

        self.layers = nn.Sequential(
            GEGLU(in_dim, hidden_dim),
            nn.Dropout(dropout_p),
            zero_module(nn.Linear(hidden_dim, in_dim)),
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.layers(x)
        out = out + x
        return out

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, activation: str = "gelu"):
        super().__init__()

        match activation.lower():
            case "gelu":
                act_fn = F.gelu
            case "relu":
                act_fn = F.relu
            case "silu":
                act_fn = F.silu
            case _:
                raise NotImplementedError(f"`{activation}` is invalid")

        self.linear_in = nn.Linear(in_dim, hidden_dim)
        self.linear_out = nn.Linear(hidden_dim, in_dim)
        self.act_fn = act_fn
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.linear_in(x)
        x = self.act_fn(x)
        x = self.linear_out(x)
        return x

class PixelUnshuffleAndProjection(nn.Module):
    def __init__(self, factor: int, in_dim: int, out_dim: int, bias: bool = False):
        super().__init__()

        self.pixel_unshuffle = nn.PixelUnshuffle(factor)
        self.proj = nn.Linear(in_dim * (factor ** 2), out_dim, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, num_tokens, in_dim = x.shape
        w = int(num_tokens ** 0.5)
        h = w

        x = x.transpose(-2, -1).reshape(batch_size, in_dim, h, w)
        x = self.pixel_unshuffle(x)
        x = x.flatten(start_dim=-2).transpose(-2, -1)
        x = self.proj(x)
        return x

class Lerp(nn.Module):
    def __init__(self, initial_value: float = 0.0):
        super().__init__()

        self.coeff = nn.parameter.Parameter(torch.tensor(initial_value))
    
    def forward(self, x_skip: Tensor, x_upsampled: Tensor) -> Tensor:
        x_merged = self.coeff * x_skip + (1.0 - self.coeff) * x_upsampled
        return x_merged
