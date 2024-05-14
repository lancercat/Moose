import torch

from neko_sdk.neko_framework_NG.bogog2_modules.neko_abstract_bogo_g2 import neko_abstract_bogo_g2
from neko_sdk.cfgtool.argsparse import neko_get_arg

class neko_transposed_fe(neko_abstract_bogo_g2):
    def setup(this,args):
        this.k=neko_get_arg("k",args);

    def forward(this, x):
        return this.fe(torch.rot90(x,this.k,dims=[2,3]));

def config_neko_transposed_fe(fe_name,k):
    return {
        "bogo_mod": neko_transposed_fe,
        "args":
        {
            "k":k,
            "mod_cvt":
            {
                "fe":fe_name,
            },
        }
    }