import os

from neko_sdk.cfgtool.platform_cfg import neko_platform_cfg
import sys
from neko_sdk.neko_framework_NG.workspace import  neko_environment

from loadout import testing_core_si,img_testing_protocol,anchor_setup

def do_test(cfg,src,dst,lang):
    anchors = anchor_setup();

    saveto, logto, testds = img_testing_protocol(cfg,src,dst,lang);


    modset,tests = testing_core_si(anchors, saveto, logto, testds);
    modset.load("_E2");
    modset.eval_mode();
    modset.to("cuda:0");
    e = neko_environment(modset=modset);
    ta = tests["agent"](tests["params"]);
    ta.take_action({}, e);


# modset.load("_E3_I0");
if __name__ == '__main__':
    if(len(sys.argv)>1):
        cfg=neko_platform_cfg(sys.argv[1]);
    else:
        cfg=neko_platform_cfg(None);
    anchors=anchor_setup();


    SRC = "/home/lasercat/ssddata/athenaNG";
    DST = "/home/lasercat/ssddata/athenaNG_results";
    langs = ["russian", "english", "korean","cypriots"];
    for l in langs:
        do_test(cfg, SRC, DST, l)
