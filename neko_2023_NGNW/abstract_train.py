import os.path

from neko_sdk.environment.root import find_data_root
from neko_sdk.neko_framework_NG.UAE.neko_trainer_agent import neko_trainer_agent
from neko_sdk.neko_framework_NG.agents.loss_logging_agent import get_neko_logging_agent
from neko_sdk.neko_framework_NG.workspace import neko_environment
from neko_sdk.cfgtool.platform_cfg import neko_platform_cfg
from osocrNG.configs.typical_agent_setups.ocr_agents.visocr_agent import get_show_batch_agent
from osocrNG.names import default_ocr_variable_names as dvn
from multiprocessing import active_children


def train_core_cat(anchors, tests, modset, routine_engine, agent_dict, qdict, cfg: neko_platform_cfg,
          data_queue_name="data_queue_", ecnt=5, icnt=200000):
    #modset.to(cfg.devices[0]);
    modset.to(cfg.devices[0]);

    rnams, rdict = routine_engine(anchors, data_queue_name)
    #
    # for n in rnams:
    #     rdict[n]["params"]["agent_list"].append("debug_visualize");
    #     rdict[n]["params"]["debug_visualize"]=get_show_batch_agent(
    #         n,n+"_tensor_image",n+"_"+dvn.raw_label_name
    #     );


    e = neko_environment(modset=modset);

def get_trainer(rnams,rdict,tests,ecnt,icnt,cfg:neko_platform_cfg,logging_engine):
    # set up trainer.
    trainer_param = {
        "routine_names": rnams,
        "routine_dict": rdict,
        "pretest_names": [],
        "pretest_dict": {
        },
        "tester_names": ["std_test"],
        "tester_dict": {
            "std_test": tests,
        },
        "posttest_names": [],
        "posttest_dict": {

        },
        "epoch_logger_names": [],
        "epoch_logger_dict": {},
        "iter_logger_names": ["loss_and_time"],
        "iter_logger_dict": {"loss_and_time": logging_engine(cfg=cfg)},
        "epoch_cnt": ecnt,
        "iter_cnt": icnt,
        "devices": cfg.devices
    }
    print(trainer_param)
    trainer = neko_trainer_agent(
        trainer_param
    )
    return trainer;
def train_core_stub(anchors, tests, modset, routine_engine, agent_dict,qdict, cfg: neko_platform_cfg,
          data_queue_name, ecnt, icnt,logger_engine):
    modset.to(cfg.devices[0]);

    rnams, rdict = routine_engine(anchors, data_queue_name)
    #
    # for n in rnams:
    #     rdict[n]["params"]["agent_list"].append("debug_visualize");
    #     rdict[n]["params"]["debug_visualize"]=get_show_batch_agent(
    #         n,n+"_tensor_image",n+"_"+dvn.raw_label_name
    #     );
    #
    e = neko_environment(modset=modset);
    trainer = get_trainer(rnams, rdict, tests, ecnt, icnt, cfg, logger_engine);

    # glue things up with queue
    for k in qdict:
        e.replace_queue(k, qdict[k])

    # engage data loader
    for a in agent_dict:
        agent_dict[a]["agent"].start(agent_dict[a]["params"], e)

    trainer.start_sync({}, e)

    # launch trainer
    for a in agent_dict:
        try:
            agent_dict[a]["agent"].stop();
        except:
            print("unstoppable")
    for a in active_children():
        a.terminate();
        exit(0);

def train_core(anchors, tests, modset, routine_engine, agent_dict, qdict, cfg: neko_platform_cfg, device,
          data_queue_name="data_queue_", ecnt=5, icnt=200000):
    #modset.to(cfg.devices[0]);
    modset.to(device);

    rnams, rdict = routine_engine(anchors, data_queue_name)

    # for n in rnams:
    #     rdict[n]["params"]["agent_list"].append("debug_visualize");
    #     rdict[n]["params"]["debug_visualize"]=get_show_batch_agent(
    #         n,n+"_tensor_image",n+"_"+dvn.raw_label_name
    #     );

def train_core(anchors, tests, modset, routine_engine, agent_dict,qdict, cfg: neko_platform_cfg,
          data_queue_name="data_queue_", ecnt=5, icnt=200000):

    return train_core_stub(anchors,tests,modset,routine_engine,agent_dict,qdict,cfg,data_queue_name,ecnt,icnt,get_neko_logging_agent);

train_core_cat=train_core;

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
    return train_core_cat(anchors, tests, modset, routine_engine, agent_dict, qdict, cfg,
    data_queue_name, ecnt, icnt);




