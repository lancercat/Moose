import torch

from neko_sdk.neko_score_merging import scatter_cvt;


class neko_openset_linear_classifier(torch.nn.Module):
    #    n=nB or n=nB*nT   nxnC       kxnC
    def set_para(this):
        this.UNK_SCR=torch.nn.Parameter(torch.zeros(1).float());
    def __init__(this,param):
        super(neko_openset_linear_classifier, this).__init__();
        this.set_para();
    def get_unk_scr(this,flat_emb):
        return this.UNK_SCR.expand(flat_emb.shape[0],1);
    def get_scr(this,flat_emb,protos):
        return flat_emb.matmul(protos.T);
    def forward(this,flat_emb,protos,plabel):
        out_res_=torch.cat([this.get_scr(flat_emb,protos),this.get_unk_scr(flat_emb)],-1);
        if(plabel is not None):
            scores = scatter_cvt(out_res_, plabel);
        else:
            scores= out_res_;
        return scores;

