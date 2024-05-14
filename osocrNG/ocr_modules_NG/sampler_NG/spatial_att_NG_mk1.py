import torch
from torch import nn
from torch.nn import functional as trnf

from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_spatial_kit.embeddings.neko_emb_intr import neko_add_embint_se


# derives from previous SE mk3
# NG will have spatial encoding by default, and also multi-headed by default.
# if you just want one head, just set nparts to 1.

class spatial_attention_NG_mk1(nn.Module):
    def set_se_engine(this,params):
        se_channel = neko_get_arg("se_channel", params, 32);
        this.se_engine=neko_add_embint_se(16,16,se_channel);
        pass;
    def set_core(this,params):
        ifc=neko_get_arg("ifc",params,32);
        se_channel = neko_get_arg("num_se_channel", params, 32);

        nparts=neko_get_arg("nparts",params,1);
        this.core = torch.nn.Sequential(
            torch.nn.Conv2d(
                ifc+se_channel, ifc+se_channel, (3, 3), (1, 1), (1, 1),
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(ifc+se_channel),
            torch.nn.Conv2d(ifc+se_channel, nparts, (1, 1)),
            torch.nn.Sigmoid(),
        );


    def __init__(this,params):
        super(spatial_attention_NG_mk1, this).__init__();
        this.set_se_engine(params);
        this.set_core(params);
        this.detached=neko_get_arg("detached",params,True);

    def forward(this, input):
        if(this.detached):
            x=input[0].detach();
        else:
            x = input[0];
        d=input[-1];
        if(x.shape[-1]!=d.shape[-1]):
            x=trnf.interpolate(x,[d.shape[-2],d.shape[-1]],mode="area");
        return this.core(this.se_engine(x));

class spatial_attention_NG_mk1DA(spatial_attention_NG_mk1):

    def forward(this, afeat,ffeat):
        if(this.detached):
            x=afeat[0].detach();
        else:
            x = afeat[0];
        d=ffeat[-1];
        if(x.shape[-1]!=d.shape[-1]):
            x=trnf.interpolate(x,[d.shape[-2],d.shape[-1]],mode="area");
        return this.core(this.se_engine(x));
