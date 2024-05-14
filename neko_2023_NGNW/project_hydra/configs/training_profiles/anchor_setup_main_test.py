
from neko_2023_NGNW.project_hydra.configs.hydra_v1_bases2 import get_hydra_v1_modules_share_bbn,get_test_hydra_routine_share_bbn_single_image


from neko_2023_NGNW.project_hydra.configs.hydra_v1 import get_hydra_v1_modules, \
    get_test_hydra_routine,get_test_hydra_routine_mo,get_test_hydra_routine_mos

def get_test_mose_stub_core(anchors, logto,testds,routine):
    tests=routine(
        {
            "iocvt_dict": {},
            "modcvt_dict": {},
            "anchors": anchors,
            "tests": testds,
            "logto":logto
        });
    return tests
def get_test_mose_stub(anchors,saveto,logto,testds,routine):
    modset = get_hydra_v1_modules(anchors, saveto, logto, None, decay=0.00001);
    tests=get_test_mose_stub_core(anchors,logto,testds,routine);
    return modset,tests;

def get_test_mose(anchors, saveto, logto,testds):
    return get_test_mose_stub(anchors,saveto,logto,testds,get_test_hydra_routine);
def get_test_mose_mo(anchors, saveto, logto,testds):
    return get_test_mose_stub(anchors,saveto,logto,testds,get_test_hydra_routine_mo);
def get_test_mose_mos(anchors, saveto, logto,testds):
    return get_test_mose_stub(anchors,saveto,logto,testds,get_test_hydra_routine_mos);

def get_test_mose_stub_sharebbnXL(anchors,saveto,logto,testds,routine):
    modset = get_hydra_v1_modules_share_bbn(anchors, saveto, logto, None, decay=0.00001,expf=2);
    tests=get_test_mose_stub_core(anchors,logto,testds,routine);
    return modset,tests;
def get_test_mose_si_sharebbnXL(anchors, saveto, logto,testds):
    return get_test_mose_stub_sharebbnXL(anchors,saveto,logto,testds,get_test_hydra_routine_share_bbn_single_image);

