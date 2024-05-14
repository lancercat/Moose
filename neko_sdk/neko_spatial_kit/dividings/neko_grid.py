import torch
from torch import nn


# 1*16, 2*8, 4*4, 8*2, 16*1
class neko_origin_grid(nn.Module):
    def __init__(this, region_shape):
        super().__init__();
        axis = [torch.arange(-1, 1 + 0.0000009, 2. / (d - 1), dtype=torch.float) for d in region_shape];
        this.origingrid = torch.nn.Parameter(torch.stack(torch.meshgrid(axis),dim=-1));
    def forward(this):
        return this.origingrid;

def neko_mkgrid_swin(dimwincount,origingrid):
    szs = (2 / torch.tensor(dimwincount, device=origingrid.device));
    offs=torch.stack(torch.meshgrid(
        [torch.range(0, i - 1, device=origingrid.device) for i in dimwincount]), dim=-1) * szs + (szs / 2);
    wins = (origingrid.unsqueeze(0).unsqueeze(0)* szs*0.5)+(offs.unsqueeze(-2).unsqueeze(-2)-1);
    return wins;

if __name__ == '__main__':
    import numpy as np
    h=8
    w=2
    g=neko_origin_grid([32,32])
    swin=neko_mkgrid_swin([h,w],g());
    swin=torch.flip(swin,dims=[-1]);
    from torch.nn import functional as trnf
    import cv2
    img=cv2.imread("/home/lasercat/Pictures/nep.png");
    print(img.shape)
    cv2.imshow("reconstruct",img)
    cv2.waitKey(0);

    winds=trnf.grid_sample(torch.tensor(img).permute(2, 0, 1).unsqueeze(0).repeat(h*w, 1, 1, 1).float(),
                     swin.reshape(h*w, 32, 32, 2));
    winds=winds.reshape(h,w,3,32,32);
    r=winds.reshape(h, w, 3, 32, 32).permute(0, 3, 1, 4, 2).reshape(h*32, w*32, 3).detach().numpy().astype(np.uint8);
    print(r.shape);
    cv2.imshow("reconstruct",r)
    cv2.waitKey(0);
    # winds=trnf.grid_sample(torch.tensor(img).permute(2,0,1).unsqueeze(0),swin.reshape(16,32,32,2));
