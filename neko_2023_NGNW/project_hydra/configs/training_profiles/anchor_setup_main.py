
from osocrNG.configs.typical_anchor_setups.nonoverlap import get_hydra_v3_anchor_2h1v_6_05 as two_hori_main
from osocrNG.configs.typical_anchor_setups.nonoverlap import get_hydra_v3_anchor_2h1Tv_6_05 as two_hori_transpose

from osocrNG.configs.typical_anchor_setups.nonoverlap import get_hydra_v3_anchor_1h1v_05 as one_hori_main
from osocrNG.configs.typical_anchor_setups.nonoverlap import get_hydra_v3_anchor_1h1v_1 as one_hori_alter

from osocrNG.configs.typical_anchor_setups.nonoverlap import get_hydra_v3_anchor_2h0v_6 as two_hori_novert

from osocrNG.configs.typical_anchor_setups.nonoverlap import get_hydra_v3_anchor_1h0v_05s as one_hori_s_novert
from osocrNG.configs.typical_anchor_setups.nonoverlap import get_hydra_v3_anchor_1h0v_05l as one_hori_l_novert

from osocrNG.configs.typical_anchor_setups.nonoverlap_alter import get_hydra_v4_anchor_2h1v_3_05 as two_hori_v2
from osocrNG.configs.typical_anchor_setups.nonoverlap_alter import get_hydra_v4_anchor_2h1Tv_3_05 as two_hori_transpose_v2
from osocrNG.configs.typical_anchor_setups.nonoverlap_alter import get_hydra_v4_anchor_2h0v_3_05 as two_hori_novert_v2
from osocrNG.configs.typical_anchor_setups.nonoverlap_alter import get_hydra_v4_anchor_1h1v_s_05 as one_hori_s_v2
from osocrNG.configs.typical_anchor_setups.nonoverlap_alter import get_hydra_v4_anchor_1h1v_l_05 as one_hori_l_v2




from neko_2023_NGNW.project_hydra.configs.training_profiles.anchor_setup_main_train import get_hydra_mose_training,get_hydra_moseXL_training,get_share_none_mose_training
from neko_2023_NGNW.project_hydra.configs.training_profiles.anchor_setup_main_test import get_test_mose_stub_core,get_test_hydra_routine_mos



def get_share_none_mose(anchors, saveto, logto, trmeta,testds):
    modset,routine_engine= get_share_none_mose_training(anchors, saveto, logto, trmeta);
    tests=get_test_mose_stub_core(anchors, logto, testds, get_test_hydra_routine_mos);
    return modset,routine_engine,tests;

# The rotation are either handled on lmdb loader side (v2h) or padding side (force_transpose), so the routines are still hydrav1.