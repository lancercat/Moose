
import torch;
from torch import nn;


class neko_cord_se(nn.Module):
    def __init__(this,dim=2):
        super(neko_cord_se, this).__init__();
        if(dim!=2):
            print("But why?");
            exit(9);
        this.sew=-1;
        this.seh=-1;
        this.se=None;
        this.dim=2;
        this.devinc=nn.Parameter(torch.tensor([0]),requires_grad=False);
    def forward(this,w,h):
        if(w != this.sew or h!=this.seh):
            this.seh=h;
            this.sew=w;
            this.se=torch.stack(torch.meshgrid([torch.arange(h)/h,torch.arange(w)/w])).float().to(this.devinc.device).unsqueeze(0);
        return this.se.to(this.devinc.device);
class neko_cord_se_feat(nn.Module):
    def __init__(this,dim=2):
        super(neko_cord_se_feat, this).__init__();
        this.se=neko_cord_se(dim);
    def forward(this,feat):
        return this.se(feat.shape[3],feat.shape[2]);


if __name__ == '__main__':
    a=torch.rand([1,3,8,32]);
    fe=neko_cord_se_feat(2);
    print(fe(a).shape)