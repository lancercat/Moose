import os
from neko_sdk.cfgtool.platform_cfg import platform_cfg
from osocrNG.common_data_presets.openset_nonmask import arm_chslat_v1, get_chs_training_meta
from osocrNG.common_data_presets.mjst_nonmask import arm_mjst_hydra_v1

def get_itr(ecnt=5):
    ecnt=ecnt;
    icnt=200000;
    return ecnt,icnt;

def get_open_bench_moostr_train_profile(cfg:platform_cfg,anchors,v2h=-9,data_queue_name="data_queue_"):
    ecnt,icnt=get_itr(2);
    trmeta=get_chs_training_meta(cfg.data_root)
    agent_dict = {};
    qdict = {};
    # build loaders---This will be moved out of train function later.
    agent_dict, qdict = arm_chslat_v1(
        agent_dict, qdict, {
            "data_queue_name": data_queue_name,
            "dataroot": cfg.data_root,
            "adict": anchors,
            "vert_to_hori": v2h,
        });
    return ecnt,icnt,trmeta,agent_dict, qdict;

def get_open_bench_moostr_train_v2h_profile(cfg:platform_cfg,anchors,data_queue_name="data_queue_"):
    return get_open_bench_moostr_train_profile(cfg,anchors,anchors["short"]["ratio"],data_queue_name);


def get_close_bench_www_training_profile(cfg:platform_cfg,anchors,v2h=-9,data_queue_name="data_queue_"):
    ecnt,icnt=get_itr(5);
    trmeta = os.path.join(cfg.data_root, "dicts", "dab62cased.pt");
    agent_dict = {};
    qdict = {};
    # build loaders---This will be moved out of train function later.
    agent_dict, qdict = arm_mjst_hydra_v1(
        agent_dict, qdict, {
            "data_queue_name": data_queue_name,
            "dataroot": cfg.data_root,
            "adict": anchors,
            "vert_to_hori": v2h,
        });
    return ecnt,icnt,trmeta,agent_dict, qdict;
