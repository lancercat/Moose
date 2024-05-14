
import torch;
from torch import nn;
from torch.nn import functional as trnf

class neko_add_embint_se(nn.Module):
    def __init__(this,w=16,h=16,c=32):
        super(neko_add_embint_se, this).__init__()
        this.param=torch.nn.Parameter(torch.rand(1,c,h,w)*2-1)
    def forward(this,x):
        N,C,H,W=x.shape;
        return torch.cat([x,trnf.interpolate(this.param,[H,W],mode="bilinear").repeat(N,1,1,1)],dim=1);

