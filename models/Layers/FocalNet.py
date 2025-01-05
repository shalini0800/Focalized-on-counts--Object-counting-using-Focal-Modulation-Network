from utils.consts import *
from utils.Layers import MLP, PatchEmbedding


class FocalModulate(Module):
    
    r"""
    
    Focal Modulation Layer as proposed in https://arxiv.org/pdf/2203.11926.pdf
    
    Args:
        dims: dimension of input
        levels: number of focal levels
        k_s: first context kernel size
        dp: drop out rate
        nl: non-linearity function
    
    """
    
    def __init__(self, dims: int, levels: int, k_s: int = 3, dp: float = 0., nl=nn.GELU()):
        super().__init__()

        self.dims = dims
        self.levels = levels
        self.nl = nl

        self.proj, self.proj_out = nn.Linear(dims, 2*dims + levels + 1), nn.Linear(dims, dims)
        self.proj_mod = nn.Conv2d(dims, dims, 1)  # same as using a linear layer
        self.dropout = nn.Dropout(dp)
        self.hc = nn.ModuleList([nn.Sequential(nn.Conv2d(dims, dims, 2*level + k_s, padding=(2*level + k_s)//2, groups=dims), nl) for level in range(levels)])  # depth wise convolutions for hierarchical contextualization

    def forward(self, x: Tensor) -> Tensor:
        # print("intial  input in focal modulate",x.shape)
        b, h, w, c = x.shape

        q, z, gates = self.proj(x).permute(0, 3, 1, 2).split((self.dims, self.dims, self.levels+1), 1)
        device = gates.device
        # print(device)
        modulator: Tensor = torch.zeros((b, self.dims, h, w))
        modulator = modulator.to(device)
        for level in range(self.levels):
            z = self.hc[level](z)
            modulator += z * gates[:, level:level+1]
        modulator += gates[:, self.levels:] * self.nl(adaptpool2d(z, (1, 1)))
        x = q * self.proj_mod(modulator)
        return self.dropout(self.proj_out(x.permute(0, 2, 3, 1)))


class FocalLayer(Module):

    def __init__(self, dims: int, levels: int = 3, k_s: int = 3, scale_r: int = 4, dp: float = 0., layer_scale: float = 1e-4):
        super().__init__()

        self.mlp = MLP(scale_r*dims, dims)
        self.modulator = FocalModulate(dims, levels, k_s, dp)

        self.norm1 = nn.LayerNorm(dims, eps=1e-6)
        self.norm2 = nn.LayerNorm(dims, eps=1e-6)

        self.ls1 = nn.Parameter(layer_scale*torch.ones(1, 1, dims), requires_grad=true) if layer_scale != 0. else 1.
        self.ls2 = nn.Parameter(layer_scale*torch.ones(1, 1, dims), requires_grad=true) if layer_scale != 0. else 1.

    def forward(self, x: Tensor):
        # print("intial input in focal layer",x.shape)
        b, c, h, w = x.shape

        x = x.permute(0, 2, 3, 1)

        shortcut = x.view(b, h*w, c)
        # print(shortcut.shape)

        x = self.modulator(self.norm1(x)).view(b, h*w, c)
        # print("x after modulator",x.shape)
        x = self.ls1*x + shortcut
        shortcut = x

        x = self.mlp(self.norm2(x.view(b, h*w, c)))
        x = self.ls2*x + shortcut
        return x.view(b, h, w, c).permute(0, 3, 1, 2)


class FocalBlock(Module):

    def __init__(self, dims: int, dims_out: int, depth: int, levels: int = 3, k_s: int = 3, scale_r: int = 4, dp: float = 0., layer_scale: float = 1e-4, last: bool = false):
        super().__init__()

        self.layers = nn.ModuleList([FocalLayer(dims, levels, k_s, scale_r, dp, layer_scale) for _ in range(depth)])

        self.embed = PatchEmbedding(dims, dims_out, patch_size=2) if not last else nn.Identity() # does not require positional information as focal modulation is transition invariant

    def forward(self, x: Tensor) -> Tensor:
        # print("initial input to focal block",x.shape)
        for layer in self.layers:
            x = layer(x)
        # print("shape before embed",x.shape)
        x= self.embed(x)
        # print("final output of focalblock",x.shape)
        # print("i am focalblock")

        return x




#if __name__ == "__main__":

 #   tnsr = torch.randn((5, 128, 64, 64))
  #  fm = FocalModulate(128, 3)

   # print(fm(tnsr).shape)
