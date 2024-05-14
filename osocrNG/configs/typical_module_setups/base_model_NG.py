import os

from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.modules.concat_mvn_dev import neko_concat_dev


def config_mvn_mods(param,cfg_dict,path,name):
    mod_param = {
        "save_each": neko_get_arg("save_each", param, 20000),
        "save_path": neko_get_arg("save_path",param,path),
        "save_name": neko_get_arg("save_name",param,name),
        "mod_type": neko_concat_dev,
        "mod_param": {
            "mean":neko_get_arg("mean",param,[127.5]),
            "var":neko_get_arg("var",param,[128]),
            }
    };
    cfg_dict[name]=mod_param;
    return cfg_dict;


def arm_imvn(param, modcfgdict, bogocfgdict, prefix, save_path):
    # using agents to control over saving does not mean it should track the saving iter
    mvn_core_name = prefix + neko_get_arg("mvn_core_name", param, "mvn_core");
    mvn_core_param = neko_get_arg("backbone_param", param, {"mean": [127.5, 127.5, 127.5], "var": 128});
    modcfgdict = config_mvn_mods(mvn_core_param, modcfgdict, save_path, mvn_core_name);

    return modcfgdict, bogocfgdict;

def arm_bmvn(param, modcfgdict, bogocfgdict, prefix, save_path):
    # using agents to control over saving does not mean it should track the saving iter
    mvn_core_name = prefix + neko_get_arg("mvn_core_name", param, "bmvn_core");
    mvn_core_param = neko_get_arg("backbone_param", param, {"mean": 0, "var": 1});
    modcfgdict = config_mvn_mods(mvn_core_param, modcfgdict, save_path, mvn_core_name);

    return modcfgdict, bogocfgdict;





from neko_sdk.encoders.chunked_resnet.g2.res45G2 import neko_r45_layers_origNG,neko_r45_norms_origNG# get xxx is a handle for the factory
def config_fe_core_mods_core(param,cfg_dict,path,name,engine):
    feat_ch = neko_get_arg("feat_ch_backbone", param, 512);
    expf = neko_get_arg("expf", param, 1);
    mod_param = {
        "save_each": neko_get_arg("save_each", param, 20000),
        "save_path": neko_get_arg("save_path",param,path),
        "save_name": neko_get_arg("save_name",param,name),
        "mod_type": engine,
        "mod_param": {
            "inpch":neko_get_arg("inpch",param,3),
            "blkcnt":neko_get_arg("blkcnt",param,[None, 3, 4, 6, 6, 3]),
            "ochs":neko_get_arg("ochs", param,
                 [int(32 * expf), int(32 * expf), int(64 * expf), int(128 * expf), int(256 * expf), int(feat_ch)]),
            "strides":neko_get_arg("strides",param,[(1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1)]),
            "inplace":neko_get_arg("inplace",param,True)
        }
    };
    cfg_dict[name]=mod_param;
    return cfg_dict;


def config_bn_mod_core(param,cfg_dict,path,name,engine):
    feat_ch=neko_get_arg("feat_ch_backbone",param,512);
    expf=neko_get_arg("expf", param, 1);

    mod_param = {
        "save_each": neko_get_arg("save_each", param, 20000),
        "save_path": neko_get_arg("save_path",param,path),
        "save_name": neko_get_arg("save_name",param,name),
        "mod_type": engine,
        "mod_param": {
            "inpch":neko_get_arg("inpch",param,3),
            "blkcnt":neko_get_arg("blkcnt",param,[None, 3, 4, 6, 6, 3]),
            "ochs":neko_get_arg("ochs", param,
                 [int(32 * expf), int(32 * expf), int(64 * expf), int(128 * expf), int(256 * expf), int(feat_ch)]),
            "affine":neko_get_arg("inplace",param,False),
            "strides": neko_get_arg("strides", param, [(1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1)]),
        }
    };
    cfg_dict[name]=mod_param;
    return cfg_dict;

def config_temporal_attention_core(param,cfg_dict,path,name,engine):
    num_se_channels=neko_get_arg("num_se_channels",param,32);
    n_vparts=neko_get_arg("n_vparts",param,0);
    n_parts=neko_get_arg("n_parts",param,1);
    expf=neko_get_arg("expf",param,1);
    feat_ch=param["feat_ch"];
    maxT=param["maxT"];
    mod_param = {
        "save_each": neko_get_arg("save_each", param, 20000),
        "save_path": neko_get_arg("save_path",param,path),
        "save_name": neko_get_arg("save_name",param,name),
        "mod_type": engine,
        "mod_param": {
            "scales":neko_get_arg("scales",param,[
                    [int(expf*32)+num_se_channels, 16, 64],
                    [int(expf*128)+num_se_channels, 8, 32],
                    [int(feat_ch)+num_se_channels, 8, 32]
                ]),
            "exps":expf,
            "depth":neko_get_arg("depth",param,8),
            "n_parts": n_parts + n_vparts,
            "maxT":maxT,
            "feat_ch":param["feat_ch"],
            "num_channels":neko_get_arg("cam_ch",param,64),
            "num_se_channels":num_se_channels,
            "detached":neko_get_arg("detached",param,True)
        }
    };
    cfg_dict[name]=mod_param;
    return cfg_dict;




def config_spatial_attention_core(param,cfg_dict,path,name,engine):
    num_se_channels=neko_get_arg("num_se_channels",param,32);
    expf=neko_get_arg("expf",param,1);
    n_parts=neko_get_arg("n_parts",param,1)
    feat_ch=param["feat_ch"];
    mod_param = {
        "save_each": neko_get_arg("save_each", param, 20000),
        "save_path": neko_get_arg("save_path",param,path),
        "save_name": neko_get_arg("save_name",param,name),
        "mod_type": engine,
        "mod_param": {
            "ifc":int(expf*32),
            "exps":expf,
            "n_parts":n_parts,
            "feat_ch":param["feat_ch"],
            "cam_ch":neko_get_arg("cam_ch",param,64),
            "num_se_channels":num_se_channels,
            "detached":neko_get_arg("detached",param,True)
        }
    };
    cfg_dict[name]=mod_param;
    return cfg_dict;



def arm_dom_fe_core(param,modcfgdict,bogocfgdict,prefix,save_path,fe_engine,bn_engine):
    expf=neko_get_arg("expf",param,1);
    backbone_core_name=neko_get_arg("backbone_core_name",param);
    fe_name=neko_get_arg("fe_name",param)
    word_bn_name=prefix+neko_get_arg("bn_name",param,"bn");
    word_bn_param = neko_get_arg("bn_param", param, {"expf":expf});
    modcfgdict=bn_engine(word_bn_param,modcfgdict,save_path,word_bn_name);
    bogocfgdict=fe_engine({"conv_name":backbone_core_name,"norm_name":word_bn_name,"name":fe_name},bogocfgdict);
    return modcfgdict,bogocfgdict




def arm_stem_no_fe_core(param,modcfgdict,bogocfgdict,prefix,save_path,proto_engine):

    arm_base_prototyper(param,modcfgdict,bogocfgdict,prefix,save_path);

    dtd_name = prefix + neko_get_arg("dtd_name", param, "dtd");
    dtd_param = neko_get_arg("dtd_param", param, {});
    modcfgdict = config_dtd(dtd_param, modcfgdict, save_path, dtd_name);

    classifer_name = prefix + neko_get_arg("classifier_name", param, "classifier");
    classifer_param = neko_get_arg("classifer_param", param, {});
    modcfgdict = config_ospredictor(classifer_param, modcfgdict, save_path, classifer_name);
    return modcfgdict, bogocfgdict;


from osocrNG.configs.typical_module_setups.feature_extraction.bogo_res45_family import config_bogo_resbinorm_g2
from osocrNG.ocr_modules_NG.sampler_NG.spatial_att_NG_mk1 import spatial_attention_NG_mk1
from osocrNG.ocr_modules_NG.sampler_NG.temporal_att_NG_mk1 import LCAM_NG_mk1

def config_bogo_fe(param,bogocfgdict):
    bogocfgdict[param["name"]]=\
        config_bogo_resbinorm_g2(param["conv_name"],param["norm_name"]);
    return bogocfgdict;


def config_fe_core_mods(param,cfg_dict,path,name):
    return config_fe_core_mods_core(param,cfg_dict,path,name,engine=neko_r45_layers_origNG);


def config_bn_mod(param,cfg_dict,path,name):
    return config_bn_mod_core(param,cfg_dict,path,name,engine=neko_r45_norms_origNG);



def config_temporal_attention(param,cfg_dict,path,name):
    return config_temporal_attention_core(param,cfg_dict,path,name,LCAM_NG_mk1);

def config_spatial_attention(param,cfg_dict,path,name):
    return config_spatial_attention_core(param,cfg_dict,path,name,spatial_attention_NG_mk1);



from osocrNG.ocr_modules_NG.sampler_NG.dtd_ng_mk1 import neko_DTDNG_mk1,neko_DTDNG_mk1mp
def config_dtd(param,cfg_dict,path,name):
    part_cnt=neko_get_arg("part_cnt",param,1);
    if(part_cnt>1):
        modtype=neko_DTDNG_mk1mp;
    else:
        modtype=neko_DTDNG_mk1;

    mod_param = {
        "save_each": neko_get_arg("save_each", param, 20000),
        "save_path": neko_get_arg("save_path",param,path),
        "save_name": neko_get_arg("save_name",param,name),
        "mod_type": modtype,
        "mod_param": {
            "dropout":neko_get_arg("dropout",param,0.0),
        }
    }
    cfg_dict[name]=mod_param;
    return cfg_dict;

from neko_sdk.neko_framework_NG.bogog2_modules.prototype_gen4 import vis_prototyper_gen4
def config_gen4_vis_prototyper(param,bogocfgdict):
    bogocfgdict[param["name"]]={
        "bogo_mod": vis_prototyper_gen4,
        "args":
            {
                "mod_cvt":
                    {
                        "backbone": param["backbone_name"],
                        "aggr": param["aggr_name"],
                    },
            }
    }
    return bogocfgdict;

from neko_sdk.neko_framework_NG.modules.openset_classifiers.multipart_classifier_kpm_NG import neko_openset_linear_classifierKPM
def config_ospredictor(param,cfg_dict,path,name):
    mod_param = {
        "save_each": neko_get_arg("save_each", param, 20000),
        "save_path": neko_get_arg("save_path",param,path),
        "save_name": neko_get_arg("save_name",param,name),
        "mod_type": neko_openset_linear_classifierKPM,
        "mod_param": {
        }
    }
    cfg_dict[name]=mod_param;
    return cfg_dict;
def arm_dom_fe(param,modcfgdict,bogocfgdict,prefix,save_path):
    return arm_dom_fe_core(param,modcfgdict,bogocfgdict,prefix,save_path,config_bogo_fe,config_bn_mod);

def arm_dom_tatt(param,modcfgdict,bogocfgdict,prefix,save_path):
    maxT = neko_get_arg("maxT", param, 25);
    feat_ch = neko_get_arg("feat_ch", param, 512);
    expf=neko_get_arg("expf",param,1);
    temporal_att_name = prefix + neko_get_arg("temporal_name", param, "temporal_att");
    temporal_att_param = neko_get_arg("temporal_att_param", param, {"maxT": maxT,"expf":expf, "feat_ch": feat_ch});
    modcfgdict = config_temporal_attention(temporal_att_param, modcfgdict, save_path, temporal_att_name);
    return modcfgdict,bogocfgdict


def arm_classifier(param,modcfgdict,bogocfgdict,prefix,save_path):
    classifer_name = prefix + neko_get_arg("classifier_name", param, "classifier");
    classifer_param = neko_get_arg("classifer_param", param, {});
    modcfgdict = config_ospredictor(classifer_param, modcfgdict, save_path, classifer_name);
    return modcfgdict, bogocfgdict;



def arm_dtd(param,modcfgdict,bogocfgdict,prefix,save_path):
    dtd_name = prefix + neko_get_arg("dtd_name", param, "dtd");
    dtd_param = neko_get_arg("dtd_param", param, {});
    modcfgdict = config_dtd(dtd_param, modcfgdict, save_path, dtd_name);
    return modcfgdict, bogocfgdict;

def arm_base_prototyper(param,modcfgdict,bogocfgdict,prefix,save_path):
    feat_ch = neko_get_arg("feat_ch", param,512);
    expf=neko_get_arg("expf",param,1);
    character_fe_name = prefix + neko_get_arg("character_fe_name", param, "character_fe");

    spatial_att_name = prefix + neko_get_arg("spatial_att_name", param, "spatial_att");
    spatial_att_param = neko_get_arg("spatial_att_param", param, {"feat_ch": feat_ch,"expf":expf});
    modcfgdict = config_spatial_attention(spatial_att_param, modcfgdict, save_path, spatial_att_name);

    prototyper_name = prefix + neko_get_arg("prototyper_name", param, "prototyper");
    bogocfgdict = config_gen4_vis_prototyper(
        {"backbone_name": character_fe_name, "aggr_name": spatial_att_name, "name": prototyper_name,"expf":expf}, bogocfgdict);

    return modcfgdict, bogocfgdict;

def arm_stem_no_fe(param,modcfgdict,bogocfgdict,prefix,save_path):
    modcfgdict, bogocfgdict=arm_base_prototyper(param, modcfgdict, bogocfgdict, prefix, save_path);
    modcfgdict, bogocfgdict=arm_dtd(param, modcfgdict, bogocfgdict, prefix, save_path);
    modcfgdict, bogocfgdict=arm_classifier(param, modcfgdict, bogocfgdict, prefix, save_path);

    return modcfgdict, bogocfgdict;

def arm_fe_core(param,modcfgdict,bogocfgdict,prefix,save_path):
    expf=neko_get_arg("expf",param,1);
    backbone_core_name = prefix + neko_get_arg("backbone_core_name", param, "backbone_core");
    backbone_core_param = neko_get_arg("backbone_param", param,{"expf":expf});
    modcfgdict = config_fe_core_mods(backbone_core_param, modcfgdict, save_path, backbone_core_name);
    return modcfgdict, bogocfgdict;


def arm_prototyper_base(param, modcfgdict, bogocfgdict, prefix, save_path):
    # using agents to control over saving does not mean it should track the saving iter
    backbone_core_name = prefix + neko_get_arg("backbone_core_name", param, "backbone_core");
    character_fe_name = prefix + neko_get_arg("character_fe_name", param, "character_fe");
    expf=neko_get_arg("expf",param,1);
    modcfgdict, bogocfgdict = arm_dom_fe({"backbone_core_name": backbone_core_name, "fe_name": character_fe_name,"expf":expf},
                                         modcfgdict, bogocfgdict, prefix + "character_", save_path);


    return modcfgdict, bogocfgdict;
def arm_fe_base(param, modcfgdict, bogocfgdict, prefix, save_path):
    modcfgdict, bogocfgdict=arm_fe_core(param, modcfgdict, bogocfgdict, prefix, save_path);
    modcfgdict, bogocfgdict=arm_prototyper_base(param, modcfgdict, bogocfgdict, prefix, save_path);
    return modcfgdict, bogocfgdict;

def arm_stem_no_util(param, modcfgdict, bogocfgdict, prefix, save_path):
    # using agents to control over saving does not mean it should track the saving iter
    modcfgdict, bogocfgdict=arm_fe_base(param, modcfgdict, bogocfgdict, prefix, save_path);
    modcfgdict, bogocfgdict = arm_stem_no_fe(param, modcfgdict, bogocfgdict, prefix, save_path)
    return modcfgdict, bogocfgdict;

def arm_stem(param,modcfgdict,bogocfgdict,prefix,save_path):
    # using agents to control over saving does not mean it should track the saving iter
    modcfgdict, bogocfgdict=arm_imvn(param, modcfgdict, bogocfgdict, prefix, save_path);
    modcfgdict, bogocfgdict=arm_fe_base(param, modcfgdict, bogocfgdict, prefix, save_path);
    modcfgdict, bogocfgdict = arm_stem_no_fe(param, modcfgdict, bogocfgdict, prefix, save_path)

    return modcfgdict, bogocfgdict;


# it is just modules.
def arm_core_modules_baseline(param,modcfgdict,bogocfgdict):
    save_path = neko_get_arg("save_path", param);
    prefix = neko_get_arg("prefix", param, "");
    backbone_core_name = prefix + neko_get_arg("backbone_core_name", param, "backbone_core");

    prefix=neko_get_arg("prefix",param,"");
    modcfgdict, bogocfgdict= arm_stem(param,modcfgdict,bogocfgdict,prefix,save_path);

    word_fe_name = prefix + neko_get_arg("word_fe_name", param, "word_fe");

    modcfgdict, bogocfgdict = arm_dom_fe({"backbone_core_name": backbone_core_name, "fe_name": word_fe_name},
                                         modcfgdict, bogocfgdict, prefix + "word_", save_path);
    modcfgdict,bogocfgdict=arm_dom_tatt(param,modcfgdict, bogocfgdict, prefix, save_path);

    return modcfgdict,bogocfgdict;


from neko_sdk.neko_framework_NG.modules.neko_label_sampler_NG import neko_prototype_sampler_NG

def config_meta_sampler_NG(param, cfg_dict, path, name):
    mod_param = {
        "save_each": neko_get_arg("save_each", param, 20000),
        "save_path": neko_get_arg("save_path",param,path),
        "save_name": neko_get_arg("save_name",param,name),
        "mod_type": neko_prototype_sampler_NG,
        "mod_param": {
            "meta_args":
                {
                    "meta_path": param["meta_path"],
                    "case_sensitive": neko_get_arg("case_sensitive", param, False),
                },
            "sampler_args":
                {
                    "max_batch_size": neko_get_arg("capacity", param, 512),
                    "val_frac": neko_get_arg("val_frac", param, 0.8),
                    "neg_servant": neko_get_arg("neg_servant", param, True),
                    "seed": neko_get_arg("seed", param, 9)
                },
        }
    };
    cfg_dict[name]=mod_param;
    return cfg_dict;


from osocrNG.configs.typical_module_setups.ococr_loss import config_ocr_loss



def config_loss_logger(param,cfg_dict,path,name):
    return cfg_dict;


def arm_base_training_modules(param,modcfgdict,bogocfgdict):
    save_path = neko_get_arg("save_path", param);
    log_path = neko_get_arg("log_path", param);
    prefix=neko_get_arg("prefix",param,"");
    sampler_name=prefix+neko_get_arg("sampler_name",param,"sampler");
    sampler_param=neko_get_arg("sampler_param",param,{
            "meta_path":param["meta_path"],
            "capacity":neko_get_arg("capacity",param,512)
        }
    );
    modcfgdict=config_meta_sampler_NG(sampler_param,modcfgdict,save_path,sampler_name);

    ocr_loss_name=prefix+neko_get_arg("ocr_loss_name",param,"ocr_loss");
    ocr_loss_param=neko_get_arg("ocr_loss_param",param,{});
    modcfgdict = config_ocr_loss(ocr_loss_param, modcfgdict, save_path, ocr_loss_name);

    loss_logging_module_name = prefix+neko_get_arg("loss_logging_name", param, "loss_logging");
    loss_logging_module_param = neko_get_arg("loss_logging_param", param, {"path":os.path.join(log_path,"loss.log")});
    modcfgdict=config_loss_logger(loss_logging_module_param,modcfgdict,save_path,loss_logging_module_name);

    training_acr_logging_module_name = prefix + neko_get_arg("training_acr_logging_name", param, "training_acr_logging");
    training_acr_logging_module_param= neko_get_arg(training_acr_logging_module_name, param, {"path":os.path.join(log_path,"training_acr.log")})

    return modcfgdict,bogocfgdict;
def arm_base_testing_modules(param,modcfgdict,bogocfgdict):
    return modcfgdict,bogocfgdict;