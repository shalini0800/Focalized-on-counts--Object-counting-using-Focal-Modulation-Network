from models.Layers.FocalNet import FocalBlock
from utils.Layers import PatchEmbedding
from utils.consts import *


class FocalNet(Module):

    def __init__(self, dims: int = 192, depths=[4, 4, 5], levels: int = 3, k_s: int = 3, scale_r: int = 4, dp: float = 0., layer_scale: float = 1e-4):
        super().__init__()

        self.model = nn.ModuleList([])

        self.inp_embed = PatchEmbedding(3, dims)  # does not require positional embedding as focal modulation is invariant to translation

        for i, depth in enumerate(depths):
            self.model.append(FocalBlock(dims * 2**i, dims * 2**(i+1), depth, levels, k_s, scale_r, dp, layer_scale, i == len(depths)-1))

    def forward(self, x: Tensor) -> Tensor:
        # print("initial input",x.shape)
        x = self.inp_embed(x)
        # print("input after embed",x.shape)
        for block in self.model:
            x = block(x)
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1)
        x = x.view(b,h*w,c)
        #x = adaptpool2d(x, (1, 1)).flatten(1)
        # print("final output of focalnet",x.shape)
        # print("i am focal net")
        return x





#if __name__ == "__main__":
 #   tnsr = torch.randn(26, 3, 384, 384)
  #  fn = FocalNet()
   # print(fn(tnsr).shape)
