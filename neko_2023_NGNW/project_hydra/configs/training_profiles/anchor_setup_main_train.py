
from neko_2023_NGNW.project_hydra.configs.hydra_v1_bases import get_hydra_v1_modules_sharenone,get_hydra_v1_modules_shareall,get_base_hydrav1_share_all
from neko_2023_NGNW.project_hydra.configs.hydra_v1 import get_hydra_v1_modules, get_base_hydrav1
from neko_2023_NGNW.project_hydra.configs.hydra_v1_bases2 import get_hydra_v1_modules_share_bbn,get_base_hydrav1_share_bbn

def get_hydra_mose_training(anchors, saveto, logto, trmeta):
    modset = get_hydra_v1_modules(anchors, saveto, logto, trmeta, decay=0.00001);
    routine_engine = get_base_hydrav1;
    return modset,routine_engine;
def get_hydra_moseXL_training(anchors, saveto, logto, trmeta):
    modset = get_hydra_v1_modules(anchors, saveto, logto, trmeta, decay=0.00001,expf=2);
    routine_engine = get_base_hydrav1;
    return modset,routine_engine;

def get_share_none_mose_training(anchors, saveto, logto, trmeta):
    modset= get_hydra_v1_modules_sharenone(anchors, saveto, logto, trmeta, decay=0.00001);
    routine_engine = get_base_hydrav1;
    return modset,routine_engine;
def get_share_all_mose_training(anchors, saveto, logto, trmeta):
    modset= get_hydra_v1_modules_shareall(anchors, saveto, logto, trmeta, decay=0.00001);
    routine_engine = get_base_hydrav1_share_all;
    return modset,routine_engine;
def get_share_backbone_mose_training(anchors, saveto, logto, trmeta):
    modset= get_hydra_v1_modules_share_bbn(anchors, saveto, logto, trmeta, decay=0.00001);
    routine_engine = get_base_hydrav1_share_bbn;
    return modset,routine_engine;

def get_hydra_moseXL_sharebbn_training(anchors, saveto, logto, trmeta):
    modset = get_hydra_v1_modules_share_bbn(anchors, saveto, logto, trmeta, decay=0.00001,expf=2);
    routine_engine = get_base_hydrav1_share_bbn;
    return modset,routine_engine;


def get_hydra_moseXL_sharebbn_trainingHD(anchors, saveto, logto, trmeta):
    modset = get_hydra_v1_modules_share_bbn(anchors, saveto, logto, trmeta, decay=0.001,expf=2);
    routine_engine = get_base_hydrav1_share_bbn;
    return modset,routine_engine;



def get_hydra_mose_trainingHD(anchors, saveto, logto, trmeta):
    modset = get_hydra_v1_modules(anchors, saveto, logto, trmeta, decay=0.001);
    routine_engine = get_base_hydrav1;
    return modset,routine_engine;
def get_hydra_moseXL_trainingHD(anchors, saveto, logto, trmeta):
    modset = get_hydra_v1_modules(anchors, saveto, logto, trmeta, decay=0.001,expf=2);
    routine_engine = get_base_hydrav1;
    return modset,routine_engine;

def get_share_none_mose_trainingHD(anchors, saveto, logto, trmeta):
    modset= get_hydra_v1_modules_sharenone(anchors, saveto, logto, trmeta, decay=0.001);
    routine_engine = get_base_hydrav1;
    return modset,routine_engine;
def get_share_all_mose_trainingHD(anchors, saveto, logto, trmeta):
    modset= get_hydra_v1_modules_shareall(anchors, saveto, logto, trmeta, decay=0.001);
    routine_engine = get_base_hydrav1_share_all;
    return modset,routine_engine;
def get_share_backbone_mose_trainingHD(anchors, saveto, logto, trmeta):
    modset= get_hydra_v1_modules_share_bbn(anchors, saveto, logto, trmeta, decay=0.001);
    routine_engine = get_base_hydrav1_share_bbn;
    return modset,routine_engine;

def get_hydra_moseXL_sharebbn_trainingHD(anchors, saveto, logto, trmeta):
    modset = get_hydra_v1_modules_share_bbn(anchors, saveto, logto, trmeta, decay=0.001,expf=2);
    routine_engine = get_base_hydrav1_share_bbn;
    return modset,routine_engine;
