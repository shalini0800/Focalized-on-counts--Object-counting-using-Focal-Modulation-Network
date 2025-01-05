from torch import nn
from torch.nn import Module
from torch import Tensor
from torch.fft import fft
from torch.nn import Softmax
import torch
import numpy as np
import torch.nn.functional as F
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d as adaptpool2d

true, false = True, False
null = None

'''
expands end dimensions
'''


def expand(x: Tensor, dims: int) -> Tensor:
    for _ in range(dims):
        x = x[..., None]
    return x
