from neko_2020nocr.dan.dataloaders.dataset_scene import lmdbDataset
from neko_2020nocr.dan.dataloaders.dataset_ms_semi import lmdbDataset_single_unlabeled

from torchvision import transforms
from neko_2020nocr.dan.configs.datasets.ds_paths import *
from neko_2020nocr.dan.dataloaders.joint_dataloader import neko_joint_data_loader
from torchvision import transforms

from neko_2020nocr.dan.configs.datasets.ds_paths import *
from neko_2020nocr.dan.dataloaders.dataset_ms_semi import lmdbDataset_single_unlabeled
from neko_2020nocr.dan.dataloaders.dataset_scene import lmdbDataset
from neko_2020nocr.dan.dataloaders.joint_dataloader import neko_joint_data_loader


def get_std_uncased_dsXL_semi(maxT=25,root="/home/lasercat/ssddata/",dict_dir='../../dict/dic_36.txt'):
    return {
        'dataset_train': neko_joint_data_loader,
        "subsets":{
            "mjst":{
                   "dstype":lmdbDataset,
                   "dscfg":{
                        'roots': [get_nips14(root),get_cvpr16(root)],
                        'img_height': 32,
                        'img_width': 128,
                        'transform': transforms.Compose([transforms.ToTensor()]),
                        'global_state': 'Train',
                        "maxT":maxT,
                   },
                   "loadercfg":{
                        'batch_size': 96,
                        'shuffle': True,
                        'num_workers': 3,
                    }
            },
            "unlabeled_icdar15":
            {
                "dstype": lmdbDataset_single_unlabeled,
                "dscfg": {
                    'root': get_IC15_2077(root),
                    'img_height': 32,
                    'img_width': 128,
                    'transform': transforms.Compose([transforms.ToTensor()]),
                    'global_state': 'Train',
                    "maxT": maxT,
                },
                "loadercfg": {
                    'batch_size': 12,
                    'shuffle': True,
                    'num_workers': 3,
                }
            }
        },
        'dataset_test': lmdbDataset,
        'dataset_test_args': {
            'roots': [get_iiit5k(root)],
            'img_height': 32,
            'img_width': 128,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Test',
            "maxT": maxT,
        },
        'dataloader_test': {
            'batch_size': 48,
            'shuffle': False,
            'num_workers': 3,
        },
        'te_case_sensitive':False,
        'case_sensitive': False,
        'dict_dir' : dict_dir
    }
