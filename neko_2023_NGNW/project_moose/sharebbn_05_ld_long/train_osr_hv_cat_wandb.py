import os.path

from neko_sdk.cfgtool.platform_cfg import neko_platform_cfg
import sys
from neko_2023_NGNW.abstract_train_wandb import train_core
from loadout import training_protocol,training_model_core,get_test_mose_wandb,anchor_setup 

# modset.load("_E3_I0");
if __name__ == '__main__':
    if(len(sys.argv)>1):
        cfg=neko_platform_cfg(sys.argv[1]);
    else:
        cfg=neko_platform_cfg(None);
    cfg.arm_wandb();
    anchors=anchor_setup();
    saveto,logto,ecnt,icnt,trmeta,agent_dict, qdict,testds=training_protocol(cfg,anchors);
    modset,routine_engine=training_model_core(anchors,saveto,logto,trmeta);
    tests=get_test_mose_wandb(anchors,cfg.run,testds);
    train_core(anchors,tests,modset,routine_engine,agent_dict,qdict,cfg,ecnt=ecnt,icnt=icnt);
