import os

from osocrNG.common_data_presets.mjst_nonmask import get_eng_test_v1,arm_mjst_hydra_v1
from osocrNG.configs.typical_anchor_setups.nonoverlap import get_hydra_v3_anchor_1h1v_1_eval,get_hydra_v3_anchor_2h1v_6_05, get_hydra_v3_anchor_1h1v_1
from neko_sdk.cfgtool.platform_cfg import platform_cfg

def get_presets_common(cfg:platform_cfg):
    anchors=get_hydra_v3_anchor_1h1v_1_eval();
    saveto = os.path.join(cfg.save_root, os.path.basename(os.getcwd()), "models/");
    logto = os.path.join(cfg.log_root, os.path.basename(os.getcwd()), "logs/");
    testds=get_eng_test_v1(cfg.data_root);
    return anchors,saveto,logto,testds;



def get_presets_2h1v_62_long(cfg:platform_cfg):
    anchors,saveto,logto,testds=get_presets_common(cfg);
    trmeta= os.path.join(cfg.data_root, "dicts", "dab62cased.pt");
    data_engine=arm_mjst_hydra_v1;
    ecnt=5;
    icnt=200000;
    return anchors,saveto,logto,trmeta,testds,data_engine,ecnt,icnt;
