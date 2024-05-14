from neko_sdk.MJT.default_config import get_default_model
from neko_sdk.encoders.chunked_resnet.g2.res45G2 import neko_r45_layers_orig,\
    neko_r45_norms_orig,neko_r45_ffns_naive,neko_r45_ffns_naive_drop\
# get xxx is a handle for the factory

def get_dan_r45_layer_orig(arg_dict,path,optim_path=None):
    args={}
    args["inpch"]= arg_dict["inpch"];
    args["strides"]=arg_dict["strides"];# C x H x W
    args["ochs"]= arg_dict["ochs"];
    args["blkcnt"]=arg_dict["blkcnt"];
    args["inplace"]=arg_dict["inplace"];
    return get_default_model(neko_r45_layers_orig,args,path,arg_dict["with_optim"],optim_path);

def get_dan_r45_norm_orig(arg_dict,path,optim_path=None):
    args={}
    args["strides"] =arg_dict["strides"];
    args["ochs"] = arg_dict["ochs"];
    args["blkcnt"] = arg_dict["blkcnt"];
    args["affine"] = arg_dict["affine"];
    return get_default_model(neko_r45_norms_orig,args,path,arg_dict["with_optim"],optim_path);

def get_dan_r45_ffn_naive(arg_dict,path,optim_path=None):
    args={};
    args["fochs"] = arg_dict["ochs"];
    args["bochs"] = arg_dict["ochs"];
    drop=arg_dict["drop"];
    if(drop is None):
        return get_default_model(neko_r45_ffns_naive,args,
                             path,arg_dict["with_optim"],optim_path);
    args["drop"]=arg_dict["drop"];
    return get_default_model(neko_r45_ffns_naive_drop,args,
                             path,arg_dict["with_optim"],optim_path)
def make_res45cfgs(ich,feat_ch,expf,inplace,bn_affine,drop=None):
    args={
        "inpch": ich,
        "strides": [(1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1)],
        "blkcnt":[None, 3, 4, 6, 6, 3],
        "ochs":[int(32*expf),int(32 * expf), int(64 * expf), int(128 * expf), int(256 * expf), feat_ch],
        "inplace": inplace,
        "affine":bn_affine,
        "with_optim": True,
        "drop":drop
    }
    return args





def config_res45_core(ich,feat_ch,expf=1,inplace=True,bn_affine=True):
    layer_factory_cfg={
            "modular": get_dan_r45_layer_orig,
            "save_each": 20000,
            "with_optim": True,
            "args": make_res45cfgs(ich,feat_ch,expf,inplace,bn_affine)
        }
    bn_factory_cfg={
            "modular": get_dan_r45_norm_orig,
            "save_each": 20000,
            "with_optim": True,
            "args": make_res45cfgs(ich,feat_ch,expf,inplace,bn_affine)
        }
    return layer_factory_cfg,bn_factory_cfg;

def config_res45_ffn_naive_core(ich,feat_ch,expf=1,inplace=True,bn_affine=True,drop=None):
    layer_factory_cfg={
            "modular": get_dan_r45_layer_orig,
            "save_each": 20000,
            "with_optim": True,
            "args": make_res45cfgs(ich,feat_ch,expf,inplace,bn_affine)
        }
    ffn_factory_cfg={
        "modular": get_dan_r45_ffn_naive,
        "save_each": 20000,
        "with_optim": True,
        "args": make_res45cfgs(ich, feat_ch, expf, inplace, bn_affine,drop=drop)
    }
    bn_factory_cfg={
            "modular": get_dan_r45_norm_orig,
            "save_each": 20000,
            "with_optim": True,
            "args": make_res45cfgs(ich,feat_ch,expf,inplace,bn_affine)
        }
    return layer_factory_cfg,ffn_factory_cfg,bn_factory_cfg;
#


def get_dan_r45_ffn_g2_naive(arg_dict,path,optim_path=None):
    args={};
    args["bochs"] = arg_dict["ochs"];
    args["fochs"] = arg_dict["ochs_ffn"];
    return get_default_model(neko_r45_ffns_naive,args,
                             path,arg_dict["with_optim"],optim_path);

def make_res45cfgs_ffng2(ich,feat_ch_backbone,feat_ch_ffn,expf,inplace,bn_affine):
    args={
        "inpch": ich,
        "strides": [(1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1)],
        "blkcnt":[None, 3, 4, 6, 6, 3],
        "ochs":[int(32*expf),int(32 * expf), int(64 * expf), int(128 * expf), int(256 * expf), int(feat_ch_backbone * expf)],
        "ochs_ffn":[int(32*expf),int(32 * expf), int(64 * expf), int(128 * expf), int(256 * expf), feat_ch_ffn],
        "inplace": inplace,
        "affine":bn_affine,
        "with_optim": True
    }
    return args

def config_res45_ffn_naive_core2(ich,feat_ch,ffn_feat_ch,expf=1,inplace=True,bn_affine=True):
    layer_factory_cfg={
            "modular": get_dan_r45_layer_orig,
            "save_each": 20000,
            "with_optim": True,
            "args": make_res45cfgs_ffng2(ich,feat_ch,ffn_feat_ch,expf,inplace,bn_affine)
        }
    ffn_factory_cfg={
        "modular": get_dan_r45_ffn_g2_naive,
        "save_each": 20000,
        "with_optim": True,
        "args": make_res45cfgs_ffng2(ich, feat_ch,ffn_feat_ch, expf, inplace, bn_affine)
    }
    bn_factory_cfg={
            "modular": get_dan_r45_norm_orig,
            "save_each": 20000,
            "with_optim": True,
            "args": make_res45cfgs_ffng2(ich,feat_ch,ffn_feat_ch,expf,inplace,bn_affine)
        }
    return layer_factory_cfg,ffn_factory_cfg,bn_factory_cfg;
#
def config_res45_ffn_naive_coreg2(ich,feat_ch_backbone,feat_ch_ffn,expf=1,inplace=True,bn_affine=True):
    layer_factory_cfg={
            "modular": get_dan_r45_layer_orig,
            "save_each": 20000,
            "with_optim": True,
            "args": make_res45cfgs_ffng2(ich,feat_ch_backbone,feat_ch_ffn,expf,inplace,bn_affine)
        }
    ffn_factory_cfg={
        "modular": get_dan_r45_ffn_g2_naive,
        "save_each": 20000,
        "with_optim": True,
        "args":  make_res45cfgs_ffng2(ich,feat_ch_backbone,feat_ch_ffn,expf,inplace,bn_affine)
    }
    bn_factory_cfg={
            "modular": get_dan_r45_norm_orig,
            "save_each": 20000,
            "with_optim": True,
            "args": make_res45cfgs_ffng2(ich,feat_ch_backbone,feat_ch_ffn,expf,inplace,bn_affine)
        }
    return layer_factory_cfg,ffn_factory_cfg,bn_factory_cfg;
#
