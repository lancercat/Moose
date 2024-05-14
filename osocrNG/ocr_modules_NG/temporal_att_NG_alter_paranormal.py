from torch.nn import functional as trnf
from neko_sdk.seq2seq.neko_sling_shot_transformer import neko_multihead_slingshot_attention_block
from neko_sdk.neko_spatial_kit.embeddings.neko_emb_intr import neko_add_embint_se
import torch
from neko_sdk.PA.utils import meanstd2mask
class neko_LoL_NG_legacy(torch.nn.Module):
    def __init__(this,param):
        super().__init__();
        this.detached=param["detached"];
        this.quiry_embs=torch.nn.Parameter(torch.rand([param["maxT"]+1,param["num_channels"]]));
        this.feat_adaptors=[];
        this.se = neko_add_embint_se(c=param["num_se_channels"]);
        this.emb_size=param["num_channels"];
        for i in range(len(param["scales"])):
            m=torch.nn.Sequential(
                torch.nn.Conv2d(param["scales"][i][0],param["num_channels"],(1,1)),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(param["num_channels"]),
            );
            this.feat_adaptors.append(m);
            this.register_module("feat_adaptor_"+str(i),m);
        this.situation_adaptor=torch.nn.Sequential(
                torch.nn.Conv2d(param["num_assessment_ch"],param["num_channels"],(1,1)),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(param["num_channels"]),
            )
        this.cores=[];
        for i in range(len(param["scales"])):
            m=neko_multihead_slingshot_attention_block(param["num_channels"],2,3);
            this.cores.append(m);
            this.register_module("core_"+str(i),m);

        this.lenpred=torch.nn.Linear(param["num_channels"],param["maxT"]+1);
        this.Apred=torch.nn.Linear(param["num_channels"],param["scales"][-1][0]);
        pass;
    
    def forward(this,tensorimg, situation, feature_, img_size, mask=None):
        nB=situation.shape[0];
        if(this.detached):
            feature=[this.se(f.detach()) for f in feature_]
        else:
            feature=feature_;
        q=this.quiry_embs.repeat(nB,1,1);
        kvs=[this.situation_adaptor(situation).reshape(nB,this.emb_size,-1).permute(0,2,1),None];
        for i in range(len(feature)):
            kvs[-1]=this.feat_adaptors[i](feature[i]).reshape(nB,this.emb_size,-1).permute(0,2,1);
            q,kvs=this.cores[i](q,kvs)
        lenpred=this.lenpred(q[:,0]);
        attn_raw=(this.Apred(q[:,1:]).unsqueeze(-1).unsqueeze(-1)*feature[-1].unsqueeze(1)).sum(2,keepdim=True);
        attn_dash_raw=q[:,1:].bmm2(kvs[1].permute(0,2,1)).reshape(attn_raw.shape);
        attn_dash_norm=trnf.sigmoid((attn_raw+attn_dash_raw)/8);# sqrtd
        return attn_dash_norm,lenpred;

from osocrNG.ocr_modules_NG.sampler_NG.temporal_att_NG_mk1 import LCAM_NG_mk1
class neko_LoL_NG_fake_cam(torch.nn.Module):
    def __init__(this, param):
        super().__init__();
        this.core=LCAM_NG_mk1(param)
        this.detached = param["detached"];
        pass;

    def forward(this, tensorimg,situation, feature_, img_size, mask=None):
        nB = tensorimg.shape[0];

        return this.core(feature_)


class neko_LoL_NG_direct(torch.nn.Module):
    def __init__(this, param):
        super().__init__();
        this.detached = param["detached"];
        this.quiry_embs = torch.nn.Parameter(torch.rand([param["maxT"] + 1, param["num_channels"]]));
        this.feat_adaptors = [];
        this.se = neko_add_embint_se(c=param["num_se_channels"]);
        this.emb_size = param["num_channels"];
        for i in range(len(param["scales"])):
            m = torch.nn.Sequential(
                torch.nn.Conv2d(param["scales"][i][0], param["num_channels"], (1, 1)),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(param["num_channels"]),
            );
            this.feat_adaptors.append(m);
            this.register_module("feat_adaptor_" + str(i), m);
        this.situation_adaptor = torch.nn.Sequential(
            torch.nn.Conv2d(param["num_assessment_ch"], param["num_channels"], (1, 1)),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(param["num_channels"]),
        )
        this.cores = [];
        for i in range(len(param["scales"])):
            m = neko_multihead_slingshot_attention_block(param["num_channels"], 2, 3);
            this.cores.append(m);
            this.register_module("core_" + str(i), m);

        this.lenpred = torch.nn.Linear(param["num_channels"], param["maxT"] + 1);
        this.Apred = torch.nn.Linear(param["num_channels"], 4);
        pass;

    def forward(this, tensorimg, situation, feature_, img_size, mask=None):
        nB = situation.shape[0];
        if (this.detached):
            feature = [this.se(f.detach()) for f in feature_]
        else:
            feature = feature_;
        q = this.quiry_embs.repeat(nB, 1, 1);
        kvs = [this.situation_adaptor(situation).reshape(nB, this.emb_size, -1).permute(0, 2, 1), None];
        for i in range(len(feature)):
            kvs[-1] =trnf.adaptive_avg_pool2d(this.feat_adaptors[i](
                feature[i]),(4,4)).reshape(nB, this.emb_size, -1).permute(0, 2, 1);
            q, kvs = this.cores[i](q, kvs)
        lenpred = this.lenpred(q[:, 0]);
        meanstd = trnf.sigmoid(this.Apred(q[:, 1:]));
        A=meanstd2mask(meanstd,img_size,(tensorimg.shape[2],tensorimg.shape[3]));
        return A, lenpred,meanstd;
