import torch
from torch import nn
from torch.nn import functional as trnf

from neko_sdk.cfgtool.argsparse import neko_get_arg


class osclsNG(nn.Module):
    def __init__(this, param):
        super(osclsNG, this).__init__();
        this.setuploss(param);

    def setuploss(this, param):
        this.criterion_CE = nn.CrossEntropyLoss();
        # this.aceloss=
        this.ignore_index = neko_get_arg("ignore_index",param,-1);

    def forward(this, outcls, label_flatten):
        w = torch.ones_like(torch.ones(outcls.shape[-1])).to(outcls.device).float();
        # w[-1] = 0;
        clsloss = trnf.cross_entropy(outcls, label_flatten, w,ignore_index=this.ignore_index);
        return clsloss;

class oslenlossNG(nn.Module):
    def __init__(this, param):
        super(oslenlossNG, this).__init__();
    def forward(this, lencls, gtlen):
        # w[-1] = 0.1;
        lenloss = trnf.cross_entropy(lencls,gtlen,ignore_index=-1);
        return lenloss;

class osocrlossNG(nn.Module):
    def __init__(this, param):
        super(osocrlossNG, this).__init__();
        this.clsloss=osclsNG(param);
        this.lenloss = oslenlossNG(param);
    def forward(this, outcls,lencls,label_flatten,gtlen_):
        # w[-1] = 0.1;
        clsloss=this.clsloss(outcls,label_flatten);
        # label_flattenk=label_flatten+0;
        # label_flattenk[label_flatten==(outcls.shape[-1]-1)]=-1;
        # clslossk = this.clsloss(outcls, label_flattenk);
        gtlen=gtlen_.clone()
        with torch.no_grad():
            gtlen[gtlen>=lencls.shape[-1]]=-1;
        lenloss = this.lenloss(lencls,gtlen);
        return lenloss+clsloss, {"cls_loss":clsloss.item(), "len_loss":lenloss.item()};

