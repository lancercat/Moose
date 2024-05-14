import torch
from torch import nn

from neko_sdk.cfgtool.argsparse import neko_get_arg


class neko_concat_dev(nn.Module):
    def device(this):
        return this.mean.data.device;
    def __init__(this,param):
        super().__init__();
        mean=neko_get_arg("mean",param,[127.5]);
        var=neko_get_arg("var",param,[128]);
        mean_var_img=neko_get_arg("2dstat",param,False);
        if(mean_var_img):
            this.mean=torch.nn.Parameter(torch.tensor(mean).float().squeeze(0),False);
            this.var = torch.nn.Parameter(torch.tensor(var).float().squeeze(0), False);
        else:
            this.mean=torch.nn.Parameter(torch.tensor(mean).float().unsqueeze(-1).unsqueeze(-1).unsqueeze(0),False);
            this.var = torch.nn.Parameter(torch.tensor(var).float().unsqueeze(-1).unsqueeze(-1).unsqueeze(0), False);

    def forward(this,imagelist):
        imgt=torch.stack([torch.tensor(i) for i in imagelist]).permute(0,3,1,2).contiguous().to(this.mean.data.device)-this.mean;
        return imgt/this.var;
    


