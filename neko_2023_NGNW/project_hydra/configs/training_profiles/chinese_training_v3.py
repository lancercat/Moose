import os
from neko_sdk.cfgtool.platform_cfg import platform_cfg
from neko_sdk.environment.root import find_data_root, find_model_root
from osocrNG.common_data_presets.openset_nonmask import arm_chslat_v1, get_osocr_test_oldjpn, get_chs_training_meta
from osocrNG.configs.typical_anchor_setups.nonoverlap import get_hydra_v3_anchor_2h1v_6_05,\
    get_hydra_v3_anchor_1h1v_05, get_hydra_v3_anchor_1h1v_1

def get_presets_common_long(cfg:platform_cfg):
    saveto = os.path.join(cfg.save_root, os.path.basename(os.getcwd()), "models/");
    logto = os.path.join(cfg.log_root, os.path.basename(os.getcwd()), "logs/");
    os.makedirs(saveto,exist_ok=True);
    os.makedirs(logto, exist_ok=True);
    ecnt=5;
    icnt=200000;

    return saveto,logto,ecnt,icnt;


def get_presets_1h1v_1_long(cfg:platform_cfg,bss=48):
    anchors=get_hydra_v3_anchor_1h1v_1(bss);

    trmeta=get_chs_training_meta(cfg.data_root)
    data_engine=arm_chslat_v1;

    testds=get_osocr_test_oldjpn(cfg.data_root);

    saveto,logto,ecnt,icnt=get_presets_common_long(cfg);

    return anchors,saveto,logto,trmeta,testds,data_engine,ecnt,icnt;

def get_presets_1h1v_05_long(cfg:platform_cfg,bss=48):
    anchors=get_hydra_v3_anchor_1h1v_05(bss);

    trmeta=get_chs_training_meta(cfg.data_root)
    data_engine=arm_chslat_v1;

    testds=get_osocr_test_oldjpn(cfg.data_root);

    saveto,logto,ecnt,icnt=get_presets_common_long(cfg);

    return anchors,saveto,logto,trmeta,testds,data_engine,ecnt,icnt;




def get_presets_2h1v_62_long(cfg:platform_cfg):
    anchors=get_hydra_v3_anchor_2h1v_6_05();

    trmeta=get_chs_training_meta(cfg.data_root);
    data_engine=arm_chslat_v1;

    testds=get_osocr_test_oldjpn(cfg.data_root);

    saveto, logto, ecnt, icnt = get_presets_common_long(cfg);

    return anchors,saveto,logto,trmeta,testds,data_engine,ecnt,icnt;



################History

def get_presets_common_param(dir_root):
    anchors = get_hydra_v3_anchor_2h1v_6_05()
    # saveto = os.path.join(dir_root, "models/")
    #logto = os.path.join(dir_root, "logs")
    testds = get_osocr_test_oldjpn(dir_root)

    return anchors, testds #, saveto, logto






def get_presets_long_params(dir_root):
    anchors, testds = get_presets_common_param(dir_root)  #, saveto, logto
    trmeta = get_chs_training_meta(dir_root)
    data_engine = arm_chslat_v1
    ecnt = 5
    icnt = 200000

    return anchors, trmeta, testds, data_engine, ecnt, icnt #, saveto, logto

from osocrNG.configs.typical_anchor_setups.overlap import get_hydra_v3o_anchor_multling

def get_presets_long_overlap():
    anchors=get_hydra_v3o_anchor_multling();
    saveto = os.path.join(find_model_root(), os.path.basename(os.getcwd()), "models/");
    logto = os.path.join(find_model_root(), os.path.basename(os.getcwd()), "logs");
    trmeta=get_chs_training_meta(find_data_root())
    testds=get_osocr_test_oldjpn(find_data_root());
    data_engine=arm_chslat_v1;
    ecnt=5;
    icnt=200000;
    return anchors,saveto,logto,trmeta,testds,data_engine,ecnt,icnt;

def get_fake_presets():
    anchors=get_hydra_v3_anchor_2h1v_6_05();
    saveto = os.path.join(find_model_root(), os.path.basename(os.getcwd()), "models/");
    logto = os.path.join(find_model_root(), os.path.basename(os.getcwd()), "logs");
    trmeta=get_chs_training_meta(find_data_root())
    testds=get_osocr_test_oldjpn(find_data_root());
    data_engine=arm_chslat_v1;
    ecnt=5;
    icnt=10000;
    return anchors,saveto,logto,trmeta,testds,data_engine,ecnt,icnt;

def get_debug_presets_hori_only_single():
    anchors=get_hydra_v3_anchor_1h1v_1();
    saveto = os.path.join(find_model_root(), os.path.basename(os.getcwd()), "models/");
    logto = os.path.join(find_model_root(), os.path.basename(os.getcwd()), "logs");
    trmeta=get_chs_training_meta(find_data_root())
    testds=get_osocr_test_oldjpn(find_data_root());
    data_engine=arm_chslat_v1;
    ecnt=5;
    icnt=10000;
    return anchors,saveto,logto,trmeta,testds,data_engine,ecnt,icnt;

