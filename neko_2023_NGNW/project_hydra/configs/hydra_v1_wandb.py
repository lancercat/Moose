import copy
from osocrNG.names import default_ocr_variable_names as dvn
from osocrNG.configs.typical_agent_setups.ocr_agents.testing_reporter import get_result_logging_agent

import torch
from neko_2023_NGNW.project_hydra.configs.hydra_v1 import get_test_hydra_core_routine
from neko_2023_NGNW.project_hydra.configs.hydra_v1_bases import get_test_hydra_routine_share_all
from neko_2023_NGNW.project_hydra.configs.hydra_v1_bases2 import get_test_hydra_routine_share_bbn
from osocrNG.configs.typical_agent_setups.ocr_agents.testing_reporter_wandb import get_wandb_mo_reporters,get_wandb_reporters
def get_test_hydra_routine_wandb(params):
    cfg=get_test_hydra_core_routine(params);
    cfg["params"]["reporters"]= get_wandb_reporters(params);
    return cfg;

def get_test_share_bbn_routine_wandb(params):
    cfg=get_test_hydra_routine_share_bbn(params);
    cfg["params"]["reporters"]= get_wandb_reporters(params);
    return cfg;
def get_test_share_all_routine_wandb(params):
    cfg=get_test_hydra_routine_share_all(params);
    cfg["params"]["reporters"]=get_wandb_reporters(params);
    return cfg;

def get_test_hydra_routine_mo_wandb(params):
    cfg=get_test_hydra_core_routine(params);
    cfg["params"]["reporters"]=get_wandb_mo_reporters(params);
    return cfg;
def get_test_share_bbn_routine_mo_wandb(params):
    cfg=get_test_hydra_routine_share_bbn(params);
    cfg["params"]["reporters"]=get_wandb_mo_reporters(params);
    return cfg;
def get_test_share_all_routine_mo_wandb(params):
    cfg=get_test_hydra_routine_share_all(params);
    cfg["params"]["reporters"]=get_wandb_mo_reporters(params);
    return cfg;
def get_test_share_bbn_routine_mos_wandb(params):
    cfg=get_test_hydra_routine_share_bbn(params);
    cfg["params"]["reporters"] = get_result_logging_agent(params["iocvt_dict"][dvn.pred_text_name],
                                                          params["iocvt_dict"][dvn.raw_label_name],
                                                          params["iocvt_dict"][dvn.raw_image_name],
                                                          params["iocvt_dict"][dvn.tdict_name],
                                                          params["iocvt_dict"][dvn.proto_label_name],
                                                          params["iocvt_dict"][dvn.tensor_proto_img_name],
                                                          params["logto"],
                                                          );
    cfg["params"]["reporters"]["case_inv_acr_time"]=get_wandb_mo_reporters(params)["case_inv_acr_time"];

    return cfg;

