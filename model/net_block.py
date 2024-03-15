import torch
import torch.nn as nn
import math
from .net_utils import eval_sh

class SinCosEmbedding(nn.Module):
    def __init__(self, sys_params):
        super(SinCosEmbedding, self).__init__() 
        self.sys_param = sys_params
        self.device = self.sys_param["device_type"]
        self.n_freqs = self.sys_param["emb_freqs_xyz"]
        self.barf_mode = self.sys_param["barf_mask"]
        self.barf_start = self.sys_param["barf_start"]
        self.barf_end = self.sys_param["barf_end"]
        self.in_channels = 3
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = self.in_channels*(len(self.funcs)*self.n_freqs+1)
        self.freq_bands = 2**torch.linspace(0, self.n_freqs-1, self.n_freqs, device=self.device)

    def forward(self, x, step_r):
        shape = x.shape
        spectrum = x[...,None]*self.freq_bands
        sin,cos = spectrum.sin(), spectrum.cos()
        input_enc = torch.stack([sin, cos],dim=-2)
        x_enc = input_enc.view(shape[0],-1)
        if self.barf_mode:
            alpha = (step_r - self.barf_start)/(self.barf_end - self.barf_start)*self.n_freqs
            k = torch.arange(self.n_freqs, dtype=torch.float32, device=self.device)
            weight = (1-(alpha-k).clamp_(min=0,max=1).mul_(math.pi).cos_())/2
            shape = x_enc.shape
            x_enc = x_enc.view(-1, self.n_freqs)*weight
            x_enc = x_enc.view(shape[0], -1)
        x_enc = torch.cat([x, x_enc],dim=-1)

        return x_enc
    
class CorseFine_NeRF(nn.Module):
    def __init__(self, sys_params, type="coarse"):
        super(CorseFine_NeRF, self).__init__()
        self.in_channels_xyz = 3*(2*sys_params["emb_freqs_xyz"] + 1) #63
        self.deg = sys_params["MLP_deg"]
        if type == "coarse":                
            self.depth = sys_params["coarse_MLP_depth"]
            self.width = sys_params["coarse_MLP_width"]
            self.skips = sys_params["coarse_MLP_skip"]
        elif type == "fine":                
            self.depth = sys_params["fine_MLP_depth"]
            self.width = sys_params["fine_MLP_width"]
            self.skips = sys_params["fine_MLP_skip"]

        for i in range(self.depth):
            if i == 0:
                layer = nn.Linear(self.in_channels_xyz, self.width)
            elif i in self.skips:
                layer = nn.Linear(self.width + self.in_channels_xyz, self.width)
            else:
                layer = nn.Linear(self.width, self.width)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.sigma = nn.Sequential(nn.Linear(self.width, self.width),
                                   nn.ReLU(True),
                                   nn.Linear(self.width, 1))
        self.sh = nn.Sequential(nn.Linear(self.width, self.width),
                                   nn.ReLU(True),
                                   nn.Linear(self.width, 3 * (self.deg + 1)**2))
        
    def forward(self, x, dirs):
        xyz_ = x        
        for i in range(self.depth):
            if i in self.skips:
                xyz_ = torch.cat([x, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)
        sigma = self.sigma(xyz_)
        sh = self.sh(xyz_)
        rgb = eval_sh(deg=self.deg, sh=sh.reshape(-1, 3, (self.deg + 1)**2), dirs=dirs) # sh: [..., C, (deg + 1) ** 2]
        rgb = torch.sigmoid(rgb)
        out = torch.cat([sigma, rgb], -1)
        
        return out    