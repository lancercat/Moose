from neko_sdk.cfgtool.platform_cfg import platform_cfg
from neko_2023_NGNW.project_hydra.configs.training_profiles.testing_profiles import get_open_bench_moostr_test,get_close_bench_www_test
from neko_2023_NGNW.project_hydra.configs.training_profiles.training_profiles import get_open_bench_moostr_train_profile,get_close_bench_www_training_profile,get_open_bench_moostr_train_v2h_profile


def get_open_bench_moostr_train(cfg:platform_cfg,anchors,v2h=-9,data_queue_name="data_queue_"):
    ecnt,icnt,trmeta,agent_dict, qdict=get_open_bench_moostr_train_profile(cfg,anchors,v2h,data_queue_name)
    saveto,logto,testds=get_open_bench_moostr_test(cfg,v2h);
    return saveto,logto,ecnt,icnt,trmeta,agent_dict, qdict,testds;

def get_open_bench_moostr_train_v2h(cfg:platform_cfg,anchors,data_queue_name="data_queue_"):
    return get_open_bench_moostr_train(cfg,anchors,anchors["short"]["ratio"],data_queue_name);


def get_close_bench_www_train(cfg:platform_cfg,anchors,v2h=-9,data_queue_name="data_queue_"):
    ecnt,icnt,trmeta,agent_dict, qdict=get_close_bench_www_training_profile(cfg,anchors,v2h,data_queue_name);
    saveto,logto,testds=get_close_bench_www_test(cfg,v2h);
    return saveto,logto,ecnt,icnt,trmeta,agent_dict, qdict,testds;
