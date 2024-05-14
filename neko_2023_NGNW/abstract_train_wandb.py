from neko_2023_NGNW.abstract_train import train_core_stub
from neko_sdk.neko_framework_NG.agents.loss_logging_agent_wandb import get_neko_logging_agent_wandb
from neko_sdk.cfgtool.platform_cfg import neko_platform_cfg

def train_core(anchors, tests, modset, routine_engine, agent_dict,qdict, cfg: neko_platform_cfg,
          data_queue_name="data_queue_", ecnt=5, icnt=200000):
    return train_core_stub(anchors,tests,modset,routine_engine,agent_dict,qdict,cfg,data_queue_name,ecnt,icnt,get_neko_logging_agent_wandb);


def train(anchors, tests, modset, routine_engine, data_engine, cfg: neko_platform_cfg,
          data_queue_name="data_queue_", ecnt=5, icnt=200000,v2h=-9):
    agent_dict = {};
    qdict = {};
    # build loaders---This will be moved out of train function later.
    agent_dict, qdict = data_engine(
        agent_dict, qdict, {
            "data_queue_name": data_queue_name,
            "dataroot": cfg.data_root,
            "adict": anchors,
            "vert_to_hori": v2h,
        });
    return train_core(anchors, tests, modset, routine_engine, agent_dict, qdict, cfg,
    data_queue_name, ecnt, icnt);




