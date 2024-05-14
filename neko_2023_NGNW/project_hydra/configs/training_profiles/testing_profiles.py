import os
from neko_sdk.cfgtool.platform_cfg import platform_cfg
from osocrNG.common_data_presets.openset_nonmask import get_osocr_test_jpn_hv,get_osocr_test_jpn_hv_full,get_osocr_test_oldjpn,get_osocr_test_image_based
from osocrNG.common_data_presets.mjst_nonmask import get_eng_test_v1

def get_paths(cfg:platform_cfg):
    saveto = os.path.join(cfg.save_root, os.path.basename(os.getcwd()), "models/");
    logto = os.path.join(cfg.log_root, os.path.basename(os.getcwd()), "logs/");
    os.makedirs(saveto, exist_ok=True);
    os.makedirs(logto, exist_ok=True);
    return saveto,logto;

def get_open_bench_ostr_test(cfg:platform_cfg):
    saveto,logto=get_paths(cfg);
    testds=get_osocr_test_oldjpn(cfg.data_root);
    return saveto,logto,testds;
def get_open_bench_moostr_test(cfg:platform_cfg,v2h=-9):
    saveto,logto=get_paths(cfg);
    testds=get_osocr_test_jpn_hv(cfg.data_root,v2h);
    return saveto,logto,testds;
def get_open_bench_moostr_test_full(cfg:platform_cfg,v2h=-9):
    saveto,logto=get_paths(cfg);
    testds=get_osocr_test_jpn_hv_full(cfg.data_root,v2h);
    return saveto,logto,testds;
def get_open_single_image_demo(cfg:platform_cfg,data_root,dst,lang,v2h=-9):
    saveto,_=get_paths(cfg);
    testds,logto=get_osocr_test_image_based(data_root,dst,lang,v2h);
    return saveto,logto,testds;
def get_close_bench_www_test(cfg:platform_cfg,v2h=-9):
    saveto,logto=get_paths(cfg);
    testds = get_eng_test_v1(cfg.data_root,v2h);
    return saveto,logto,testds;