import torch

from neko_sdk.CnC.situation_assess.neko_eff_assessment import neko_assess_r12_direct_se
from neko_sdk.CnC.situation_assess.neko_r45_assessment import neko_assess_r45_direct_se
from neko_sdk.cfgtool.argsparse import neko_get_arg


def config_r45_assess_mod(param,cfg_dict,path,name):
    feat_ch=neko_get_arg("assessment_visfeat_ch",param,384);
    se_ch=neko_get_arg("assessment_se_ch",param,128);

    mod_param = {
        "save_each": neko_get_arg("save_each", param, 20000),
        "save_path": neko_get_arg("save_path",param,path),
        "save_name": neko_get_arg("save_name",param,name),
        "mod_type": neko_assess_r12_direct_se,
        "mod_param": {
            "indim":neko_get_arg("inpch",param,3),
            "outdim":feat_ch,
            "sedim":se_ch,
           }
    };
    cfg_dict[name]=mod_param;
    return cfg_dict;
def arm_dom_assessments_r45_direct_se(param,modcfgdict,bogocfgdict,prefix,save_path):
    assess_backbone_name = prefix+neko_get_arg("assess_backbone_name", param,"assessment");
    assess_backbone_param=neko_get_arg("assess_backbone_param",param,{"indim":3,"assessment_visfeat_ch":384,"assessment_se_ch":128});
    modcfgdict=config_r45_assess_mod(assess_backbone_param,modcfgdict,save_path,assess_backbone_name)
    return modcfgdict,bogocfgdict;

if __name__ == '__main__':
    mp=config_r45_assess_mod({},{},"Nep","Nep")["Nep"]["mod_param"];
    mod=neko_assess_r45_direct_se(mp);
    mod.cuda();
    ret=mod(torch.rand([48,3,48,48],dtype=torch.float32,device="cuda:0"));
    pass;
