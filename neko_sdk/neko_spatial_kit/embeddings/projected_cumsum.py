import torch
class neko_cumsum_se_2d(torch.nn.Module):
    def __init__(this,och):
        super(neko_cumsum_se_2d, this).__init__();
        this.projector=torch.nn.Sequential(
            torch.nn.Conv2d(2,och,1,bias=False),
            torch.nn.Tanh(),
        );
    def forward(this,mask):
        mi=(mask>0.09)*torch.cat([mask.cumsum(-1),mask.cumsum(-2)],1);
        mi/=mi.max(-2,keepdim=True)[0].max(-1,keepdim=True)[0];
        return this.projector(mi);
