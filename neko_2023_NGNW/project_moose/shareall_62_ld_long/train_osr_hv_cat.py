from neko_sdk.cfgtool.platform_cfg import neko_platform_cfg
import sys
from neko_2023_NGNW.abstract_train import train_core_cat
from loadout import training_protocol,training_model,anchor_setup

# modset.load("_E3_I0");
if __name__ == '__main__':
    if(len(sys.argv)>1):
        cfg=neko_platform_cfg(sys.argv[1]);
    else:
        cfg=neko_platform_cfg(None);
    anchors=anchor_setup();
    saveto,logto,ecnt,icnt,trmeta,agent_dict,qdict,testds=training_protocol(cfg,anchors);
    modset,routine_engine,tests=training_model(anchors,saveto,logto,trmeta,testds);
    train_core_cat(anchors,tests,modset,routine_engine,agent_dict,qdict,cfg,ecnt=ecnt,icnt=icnt);
