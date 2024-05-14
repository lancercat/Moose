########## Protocol goes here.
from neko_2023_NGNW.project_hydra.configs.training_profiles.common            import get_open_bench_moostr_train   as training_protocol

from neko_2023_NGNW.project_hydra.configs.training_profiles.testing_profiles import get_open_bench_moostr_test     as db_testing_protocol
from neko_2023_NGNW.project_hydra.configs.training_profiles.testing_profiles import get_open_single_image_demo     as img_testing_protocol
########## Make implementation choices here.
from neko_2023_NGNW.project_hydra.configs.training_profiles.anchor_setup_main import two_hori_main                 as anchor_setup
from neko_2023_NGNW.project_hydra.configs.training_profiles.anchor_setup_main_train import get_share_backbone_mose_training as training_model_core

from neko_2023_NGNW.project_hydra.configs.training_profiles.anchor_setup_main_test_wandb import get_test_mose_mo_wandb_share_bbn as get_test_mose_wandb;

method_name="shared_backbone_3_horizontal_1_vertical_6_05" #(dedicated data)
