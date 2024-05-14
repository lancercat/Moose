########## Protocol goes here.
from neko_2023_NGNW.project_hydra.configs.training_profiles.testing_profiles import get_open_bench_moostr_test     as db_testing_protocol
from neko_2023_NGNW.project_hydra.configs.training_profiles.testing_profiles import get_open_bench_ostr_test     as db_testing_protocol_legacy
from neko_2023_NGNW.project_hydra.configs.training_profiles.testing_profiles import get_open_single_image_demo     as img_testing_protocol

from neko_2023_NGNW.project_hydra.configs.training_profiles.testing_profiles import get_open_bench_moostr_test_full     as db_testing_protocol_full

from neko_2023_NGNW.project_hydra.configs.training_profiles.common            import get_open_bench_moostr_train     as training_protocol
########## Make implementation choices here.
from neko_2023_NGNW.project_hydra.configs.training_profiles.anchor_setup_main import two_hori_main              as anchor_setup

from neko_2023_NGNW.project_hydra.configs.training_profiles.anchor_setup_main_train import get_hydra_moseXL_sharebbn_training as training_model_core
from neko_2023_NGNW.project_hydra.configs.training_profiles.anchor_setup_main_test_wandb import get_test_mose_mo_wandb_share_bbn as get_test_mose_wandb;
from neko_2023_NGNW.project_hydra.configs.training_profiles.anchor_setup_main_test_wandb import get_test_mose_mos_wandb_share_bbn as get_test_mose_wandb_mos;
from neko_2023_NGNW.project_hydra.configs.training_profiles.anchor_setup_main_test import get_test_mose_si_sharebbnXL                  as testing_core_si

method_name="mooseXL_2h1v_6_05"

