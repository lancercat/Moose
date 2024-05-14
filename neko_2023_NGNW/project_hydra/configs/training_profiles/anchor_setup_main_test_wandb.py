
from neko_2023_NGNW.project_hydra.configs.hydra_v1_wandb import get_test_hydra_routine_wandb,get_test_hydra_routine_mo_wandb,\
    get_test_hydra_routine_share_bbn,get_test_share_all_routine_wandb,get_test_share_bbn_routine_mo_wandb,get_test_share_all_routine_mo_wandb,get_test_share_bbn_routine_mos_wandb


def get_test_mose_stub(anchors,wandb_run,testds,routine,log_to="NEP_skipped_NEP"):
    tests=routine(
        {
            "iocvt_dict": {},
            "modcvt_dict": {},
            "anchors": anchors,
            "tests": testds,
            "wandb_run":wandb_run,
            "logto": log_to,
        });
    return tests

def get_test_mose_wandb(anchors, wandb_run,testds):
    return get_test_mose_stub(anchors,wandb_run,testds,get_test_hydra_routine_wandb);
def get_test_mose_mo_wandb(anchors, wandb_run,testds):
    return get_test_mose_stub(anchors,wandb_run,testds,get_test_hydra_routine_mo_wandb);


def get_test_mose_wandb_share_bbn(anchors, wandb_run,testds):
    return get_test_mose_stub(anchors,wandb_run,testds,get_test_hydra_routine_share_bbn);
def get_test_mose_wandb_share_all(anchors, wandb_run,testds):
    return get_test_mose_stub(anchors,wandb_run,testds,get_test_share_all_routine_wandb);

def get_test_mose_mo_wandb_share_bbn(anchors, wandb_run,testds):
    return get_test_mose_stub(anchors,wandb_run,testds,get_test_share_bbn_routine_mo_wandb);
def get_test_mose_mo_wandb_share_all(anchors, wandb_run,testds):
    return get_test_mose_stub(anchors,wandb_run,testds,get_test_share_all_routine_mo_wandb);


def get_test_mose_mos_wandb_share_bbn(anchors, wandb_run,testds,logto):
    return get_test_mose_stub(anchors,wandb_run,testds,get_test_share_bbn_routine_mos_wandb,logto);
