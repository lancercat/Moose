import os

from neko_sdk.cfgtool.platform_cfg import neko_platform_cfg
import sys
from neko_sdk.neko_framework_NG.workspace import  neko_environment

from loadout import get_test_mose_wandb_mos,db_testing_protocol_full,anchor_setup,training_model_core

# modset.load("_E3_I0");
if __name__ == '__main__':
    if(len(sys.argv)>1):
        cfg=neko_platform_cfg(sys.argv[1]);
    else:
        cfg=neko_platform_cfg(None);
    cfg.arm_wandb();
    anchors=anchor_setup();
    saveto, logto, testds=db_testing_protocol_full(cfg);
    modset, _ = training_model_core(anchors, saveto, logto, None);
    modset.load("_E1");
    modset.eval_mode();
    modset.to("cuda:0");
    os.makedirs(logto,exist_ok=True);
    print(logto);
    tests=get_test_mose_wandb_mos(anchors,cfg.run,testds,logto=logto);

    e = neko_environment(modset=modset);
    ta = tests["agent"](tests["params"]);
    ta.take_action({}, e);

