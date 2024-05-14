from torch import nn
from torch.nn import functional as trnf

from neko_sdk.PA.utils import paranormal_scaled_2d
from neko_sdk.cfgtool.argsparse import neko_get_arg;
from osocrNG.ocr_modules_NG.sampler_NG.temporal_att_NG_mk1 import LCAM_NG_mk1


class neko_CAM_stop_mpgo_seintr(LCAM_NG_mk1):
    def __init__(this,param):
        this.detached=neko_get_arg("detached",param,True);
        this.n_parts = neko_get_arg("n_parts",param,1);
        this.detached = neko_get_arg("detached", param, True);
        this.n_parts_paranormal = neko_get_arg("n_parts_paranormal",param,1);
        this.maxT=param["maxT"];

        this.paranormal_T=this.maxT*this.n_parts_paranormal;
        this.lift=neko_get_arg("lift",param,0.1);
        super(neko_CAM_stop_mpgo_seintr, this).__init__(param);
    def arm_pred_stop(this, ich_stop, maxT):
        this.lenpred = nn.Linear(ich_stop, maxT, False);
        this.meanstdpred=nn.Sequential(
            nn.Linear(ich_stop, this.paranormal_T * 4),
            nn.Sigmoid(),
        )

    def att_len_core(this,input,mask,t_override):
        x = input[0]
        for i in range(0, len(this.fpn)):
            x = this.fpn[i](x) + input[i + 1]
        conv_feats = []
        stoppers = []
        for i in range(0, len(this.convs)):
            x = this.convs[i](x)
            conv_feats.append(x)
        stoppers.append(x.mean(-1).mean(-1));
        for i in range(0, len(this.deconvs) - 1):
            x = this.deconvs[i](x)
            f = conv_feats[len(conv_feats) - 2 - i]
            stoppers.append(f.mean(-1).mean(-1));
            x = x[:, :, :f.shape[2], :f.shape[3]] + f
        Ts = torch.cat(stoppers, dim=-1);
        leng = this.lenpred(Ts);
        if (t_override is None):
            t_override = leng.argmax(-1)[0];
        att = this.make_att(x, t_override);
        if (mask is not None):
            if (mask.shape[-2] != x.shape[-2] or mask.shape[-1] != x.shape[-1]):
                mask_ = trnf.interpolate(mask, [x.shape[-2], x.shape[-1]], mode="bilinear");
            else:
                mask_ = mask;
            att = att * mask_;
        # No spikes! features must hold ~1/100 of the image
        meanstd = this.meanstdpred(Ts).reshape(leng.shape[0],-1,4)+0.009;
        return att, leng,meanstd;

    def forward(this, input_, mask=None, t_override=None):
        if this.detached:
            input = [this.semods[i](input_[i].detach()) for i in range(len(input_))]
        else:
            input = [this.semods[i](input_[i]) for i in range(len(input_))]
        return this.feat_aggr(input, mask, t_override);

    def feat_aggr(this,input,mask,t_override):
        x, leng,meanstd_=this.att_len_core(input,mask,t_override);
        N,T,H,W=x.shape[0], x.shape[1] // (this.n_parts),x.shape[2],x.shape[3];
        meanstd=meanstd_*torch.tensor([H,W,H,W],dtype=torch.float32,device=meanstd_.device);
        cord_grid=torch.stack(torch.meshgrid(torch.arange(0,H,device=x.device),torch.arange(0,W,device=x.device)),dim=-1);
        xg=paranormal_scaled_2d(cord_grid, meanstd[:,:,:2], meanstd[:,:,2:]);
        x = x.reshape(N, T , this.n_parts, H, W);
        xg = xg.reshape(N, T, this.n_parts_paranormal, H, W);

        a=(xg[:,:,:1]*(1-this.lift)+this.lift)*x
        return a,leng;

    def feat_aggr_d(this, input, mask, t_override):
        x, leng, std_ = this.att_len_core(input, mask, t_override);
        N, T, H, W = x.shape[0], x.shape[1] // (this.n_parts), x.shape[2], x.shape[3];
        std = std_ * torch.tensor([H, W], dtype=torch.float32, device=std_.device);
        cord_grid = torch.stack(
            torch.meshgrid(torch.arange(0, H, device=x.device), torch.arange(0, W, device=x.device)), dim=-1);
        mean = (x.unsqueeze(-1) * cord_grid.unsqueeze(0).unsqueeze(0)).sum(2).sum(2) / (
                    x.unsqueeze(-1).sum(2).sum(2) + 0.0000009);
        xg = paranormal_scaled_2d(cord_grid, mean, std);

        x = x.reshape(N, T, this.n_parts, H, W);
        xg = xg.reshape(N, T, this.n_parts_paranormal, H, W);

        a = torch.cat([xg[:, :, :1] , x[:, :, 1:]],dim=2)
        return a,None;



# G means ghost, as the  attention resembles a paranormal distribution,
# and it also floats around like a ghost
class neko_CAM_stop_mpg_seintr_NG(neko_CAM_stop_mpgo_seintr):
    def arm_pred_stop(this, ich_stop, maxT):
        this.lenpred = nn.Linear(ich_stop, maxT, False);
        this.varpred = nn.Sequential(
            nn.Linear(ich_stop, maxT * 2),
            nn.Sigmoid(),
        );
    def att_len_core(this,input,mask,t_override):
        x = input[0]
        for i in range(0, len(this.fpn)):
            x = this.fpn[i](x) + input[i + 1]
        conv_feats = []
        stoppers = []
        for i in range(0, len(this.convs)):
            x = this.convs[i](x)
            conv_feats.append(x)
        stoppers.append(x.mean(-1).mean(-1));
        for i in range(0, len(this.deconvs) - 1):
            x = this.deconvs[i](x)
            f = conv_feats[len(conv_feats) - 2 - i]
            stoppers.append(f.mean(-1).mean(-1));
            x = x[:, :, :f.shape[2], :f.shape[3]] + f
        Ts = torch.cat(stoppers, dim=-1);
        leng = this.lenpred(Ts);
        if (t_override is None):
            t_override = leng.argmax(-1)[0];
        att = this.make_att(x, t_override);
        if (mask is not None):
            if (mask.shape[-2] != x.shape[-2] or mask.shape[-1] != x.shape[-1]):
                mask_ = trnf.interpolate(mask, [x.shape[-2], x.shape[-1]], mode="bilinear");
            else:
                mask_ = mask;
            att = att * mask_;
        # No spikes! features must hold ~1/100 of the image
        std = this.varpred(Ts).reshape(leng.shape[0],leng.shape[1],-1)+0.009;
        return att, leng,std;

    def feat_aggr(this,input,mask,t_override):
        x, leng,std_=this.att_len_core(input,mask,t_override);
        N,T,H,W=x.shape[0], x.shape[1] // (this.n_parts),x.shape[2],x.shape[3];
        std=std_*torch.tensor([H,W],dtype=torch.float32,device=std_.device);
        cord_grid=torch.stack(torch.meshgrid(torch.arange(0,H,device=x.device),torch.arange(0,W,device=x.device)),dim=-1);
        mean=(x.unsqueeze(-1)*cord_grid.unsqueeze(0).unsqueeze(0)).sum(2).sum(2)/(x.unsqueeze(-1).sum(2).sum(2)+0.0000009);
        xg=paranormal_scaled_2d(cord_grid, mean, std);

        x = x.reshape(N, T , this.n_parts , H, W);
        xg = xg.reshape(N, T , this.n_parts , H, W);
        a=(xg[:,:,:1]*(1-this.lift)+this.lift)*x
        return a,leng;

