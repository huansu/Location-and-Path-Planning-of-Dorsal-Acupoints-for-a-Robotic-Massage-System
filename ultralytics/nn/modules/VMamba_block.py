import torch
import torch.nn as nn
from .conv import Conv, DWConv, ConvLayer1D, ConvLayer2D, LayerNorm1D
from .block import FFN

__all__ = (
    'LiteViM'
)


class TeLU(nn.Module):
    def __init__(self, ):
        super(TeLU, self).__init__()

    def forward(self, x):
        return x * torch.tanh(torch.exp(x))


class SSD(nn.Module):
    def __init__(self, d_model, ssd_expand=1, A_init_range=(1, 16), state_dim=64):
        super().__init__()
        self.ssd_expand = ssd_expand
        self.d_inner = int(self.ssd_expand * d_model)
        self.state_dim = state_dim  # N

        self.BCdt_proj = ConvLayer1D(d_model // 2, state_dim, 1, norm=None, act_layer=None)

        self.dw = ConvLayer2D(state_dim, state_dim, 3, 1, 1, groups=state_dim, norm=None, act_layer=None,
                              bn_weight_init=0)


        A = torch.empty(self.state_dim, dtype=torch.float32).uniform_(*A_init_range)
        self.A = torch.nn.Parameter(A)
        self.act = nn.SiLU()
        self.D = nn.Parameter(torch.ones(1))
        self.D._no_weight_decay = True

        self.coeffs = nn.Parameter(torch.tensor([1.0, 0.5, 0.8]), requires_grad=True)
        self.DW = DWConv(d_model, d_model)

    def forward(self, x, H, W):
        batch, _, L = x.shape

        x, gate = torch.chunk(x, 2, dim=1)

        BCdt = self.dw(self.BCdt_proj(x).view(batch, -1, H, W)).flatten(2)

        BCdt_expanded = BCdt.unsqueeze(1)
        weighted_BCdt = BCdt_expanded * self.coeffs.view(1, 3, 1, 1)
        B, C, dt = [t.squeeze(1) for t in torch.chunk(weighted_BCdt, chunks=3, dim=1)]

        A = (dt + self.A.view(1, -1, 1)).softmax(-1)
        AB = (A * B)
        h = x @ AB.transpose(-2, -1)

        y = h @ C

        y = torch.cat([y, gate], 1)

        y = self.DW(y.view(batch, -1, H, W).contiguous())

        return y, h


class LiteViMBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., ssd_expand=1, state_dim=64):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio

        self.mixer = SSD(dim, ssd_expand=ssd_expand, state_dim=state_dim)
        self.norm = LayerNorm1D(dim)

        self.dwconv1 = ConvLayer2D(self.dim, self.dim, 3, padding=1, groups=self.dim, bn_weight_init=0, act_layer=None)
        self.dwconv2 = ConvLayer2D(self.dim, self.dim, 3, padding=1, groups=self.dim, bn_weight_init=0, act_layer=None)

        self.ffn = FFN(in_dim=dim, dim=int(dim * mlp_ratio))

        # LayerScale
        self.alpha = nn.Parameter(1e-4 * torch.ones(4, dim), requires_grad=True)

        self.act = nn.SiLU()



    def forward(self, x):
        alpha = torch.sigmoid(self.alpha).view(4, -1, 1, 1)

        # DWconv1
        x = (1 - alpha[0]) * x + alpha[0] * self.dwconv1(x)

        # HSM-SSD
        x_prev = x
        H, W = x.shape[2:]

        x, _ = self.mixer(self.norm(x.flatten(2)), H, W)

        x = (1 - alpha[1]) * x_prev + alpha[1] * x

        # DWConv2
        x = (1 - alpha[2]) * x + alpha[2] * self.dwconv2(x)

        # FFN
        x = (1 - alpha[3]) * x + alpha[3] * self.ffn(x)
        return x, _


class LiteViM(nn.Module):
    def __init__(self, in_dim, out_dim, n, state_dim=64, mlp_ratio=4.,ssd_expand=1):
        super().__init__()
        self.branch = out_dim / in_dim == 2
        self.DWConv = DWConv(in_dim, out_dim)


        if self.branch:
            self.branchConv = DWConv(in_dim // 2, in_dim // 2)
            self.blocks = nn.ModuleList([
                LiteViMBlock(dim=in_dim // 2, mlp_ratio=mlp_ratio, ssd_expand=ssd_expand, state_dim=state_dim) for _ in
                range(n)])
        else:
            self.branchConv = DWConv(out_dim // 2, out_dim // 2)
            self.blocks = nn.ModuleList([
                LiteViMBlock(dim=out_dim // 2, mlp_ratio=mlp_ratio, ssd_expand=ssd_expand, state_dim=state_dim) for _ in
                range(n)])

    def forward(self, x):
        if self.branch:
            x_in = x
            x, y = torch.chunk(x, 2, dim=1)
            # y = self.branchConv(y)
            for blk in self.blocks:
                x, _ = blk(x)
            x = torch.cat([x_in, x, y], dim=1)
        else:
            x = self.DWConv(x)
            x, y = torch.chunk(x, 2, dim=1)
            # y = self.branchConv(y)
            for blk in self.blocks:
                x, _ = blk(x)
            x = torch.cat((x, y), dim=1)
        return x