
import os

import torch
from torch import nn
from torch.nn import parallel as trnp
from neko_sdk.neko_framework_NG.neko_modular_NG import neko_modular_NG
from transformers import AutoModel,AutoFeatureExtractor
# A wrapper that allows you take huggingface automodels into your asset.
# Note you need to do the house keeping of constructing it.
class neko_HF_AutoModel_wrapper(nn.Module):
    TYPE=AutoModel;
    PARAM_HF_path="hf_path";
    PARAM_HF_pretrain="hf_pretrain";
    PARAM_HF_args="param_hf_args";
    def __init__(this,params):
        super().__init__();
        if(params[this.PARAM_HF_pretrain]):
            this.model = this.TYPE.from_pretrained(params[this.PARAM_HF_path],**params[this.PARAM_HF_args]);
        else:
            this.model = this.TYPE.from_config(params[this.PARAM_HF_path],**params[this.PARAM_HF_args]);
    def forward(this, *argv, **kwargs):
        return this.model(*argv,**kwargs);

class neko_HF_AutoFeatureExtractor_wrapper(neko_HF_AutoModel_wrapper):
    TYPE=AutoFeatureExtractor;