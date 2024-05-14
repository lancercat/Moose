import copy

import torch

from neko_2023_NGNW.project_hydra.configs.hydra_common import get_tester_hydra
from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.environment.root import find_data_root
from neko_sdk.neko_framework_NG.UAE.neko_mission_agent import neko_test_mission_agent
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_keyword_selective_execution_agent

torch.backends.cudnn.flags(benchmark=False);
from osocrNG.configs.typical_agent_setups.ocr_agents.testing_reporter import get_reporters,get_result_logging_agent,get_mo_reporters
from osocrNG.bogomods_g2.transpose_im import config_neko_transposed_fe

from neko_sdk.neko_framework_NG.neko_module_setNG import neko_module_opt_setNG,get_modular_dict


from osocrNG.configs.typical_module_setups.base_model_NG import arm_base_training_modules,arm_dom_fe,arm_stem,arm_dom_tatt

from osocrNG.configs.typical_agent_setups.base.mvn_fe_seq_pred_translate import get_tranining_routine_fpbp,get_ocr_core

from osocrNG.configs.typical_module_setups.ococr_loss import arm_ocr_loss
from neko_2023_NGNW.project_hydra.configs.names import default_hydra_variable_names as dvn
from neko_2023_NGNW.project_hydra.configs.utils.hydra_debugger import get_visualize_hydra_routine


def arm_core_modules_hydra(param, modcfgdict, bogocfgdict):
    save_path = neko_get_arg("save_path", param);
    expf=neko_get_arg("expf",param,1);
    prefix=neko_get_arg("prefix",param,"");
    backbone_core_name = prefix + neko_get_arg("backbone_core_name", param, "backbone_core");

    for k in param["adict"]["names"]:
        if("force_transpose" in param["adict"][k]):
            modcfgdict, bogocfgdict = arm_dom_fe(
                {"backbone_core_name": backbone_core_name, "fe_name": prefix + k + "_word_fe_stub", "expf": expf},
                modcfgdict, bogocfgdict, prefix + k + "_word_", save_path);
            bogocfgdict[prefix+k+"_word_fe"]=config_neko_transposed_fe(prefix+k+"_word_fe_stub",param["adict"][k]["force_transpose"]);

        else:
            modcfgdict, bogocfgdict = arm_dom_fe({"backbone_core_name": backbone_core_name, "fe_name": prefix+k+"_word_fe","expf":expf},
                                             modcfgdict, bogocfgdict, prefix + k+"_word_", save_path);

        modcfgdict,bogocfgdict=arm_dom_tatt(param,modcfgdict, bogocfgdict, prefix+k+"_", save_path);

    modcfgdict, bogocfgdict =arm_stem(param,modcfgdict,bogocfgdict,prefix,save_path);

    return modcfgdict,bogocfgdict;
def get_hydra_v1_modules(anchors,save_path,log_path,meta_path,decay=0.0005,expf=1):
    modset = neko_module_opt_setNG();
    modcfgs = {};
    bogocfgs = {};
    modcfgs, bogocfgs = arm_core_modules_hydra(
        {
            "expf":expf,
            "adict":anchors,
            "dataroot": find_data_root(),
            "save_path": save_path,
        }, modcfgs, bogocfgs);
    if(meta_path is not None):
        modcfgs, bogocfgs = arm_base_training_modules({
            "dataroot": find_data_root(),
            "save_path": save_path,
            "log_path": log_path,
            "meta_path": meta_path},
            modcfgs, bogocfgs);
        modcfgs, bogocfgs = arm_ocr_loss({
            "dataroot": find_data_root(),
            "save_path": save_path
        },
            modcfgs, bogocfgs)

    modset.arm_modules(modcfgs, bogocfgs,decay_override=decay);
    print(modcfgs, bogocfgs);

    moddict = get_modular_dict(modset);
    print(moddict.keys());
    return modset;

def get_base_hydrav1(anchors,qprefix):
    rnams=anchors["names"];
    rdict={};

    for n in rnams:
        anchor_training_routine = get_tranining_routine_fpbp(n+"_", qprefix+n, n+"_pred", n+"_loss",
                                                           "sampler", "mvn_core", "prototyper", n+"_word_fe", n+"_temporal_att",
                                                           "dtd", "classifier", "ocr_loss");
        rdict[n]=anchor_training_routine;
    return rnams,rdict;


def get_base_hydrav1_vis(anchors,qprefix):
    rnams=copy.copy(anchors["names"]);
    rnams+=[n+"_dbg" for n in anchors["names"]];
    rdict={};

    for n in anchors["names"]:
        prefix=n+"_";
        anchor_training_routine = get_tranining_routine_fpbp(prefix, qprefix+n, prefix+"pred", prefix+"loss",
                                                           "sampler", "mvn_core", "prototyper", prefix+"word_fe", prefix+"temporal_att",
                                                           "dtd", "classifier", "ocr_loss");


        visualization_routine=get_visualize_hydra_routine(prefix,prefix + "raw_image",prefix+"text_label",prefix+"tdict",prefix+"tensor_proto_img",prefix+"plabel");
        rdict[n]=anchor_training_routine;
        rdict[n+"_dbg"]=visualization_routine;
    return rnams,rdict;


def get_base_hydrav1_dbg_single(anchors,qprefix):
    rnams=anchors["names"][:-1];
    rdict={};

    for n in rnams:
        anchor_training_routine = get_tranining_routine_fpbp(n+"_", qprefix+n, n+"_pred", n+"_loss",
                                                           "sampler", "mvn_core", "prototyper", n+"_word_fe", n+"_temporal_att",
                                                           "dtd", "classifier", "ocr_loss");
        rdict[n]=anchor_training_routine;
    return rnams,rdict;


def get_base_hydrav1x4(anchors,qprefix):
    rnams=[];
    rdict={};
    for n_ in anchors["names"]:
        for i in range(4):
            n=str(i)+n_
            rnams.append(n)
            anchor_training_routine = get_tranining_routine_fpbp(n+"_", qprefix+n_, n+"_pred", n+"_loss",
                                                               "sampler", "mvn_core", "prototyper", n_+"_word_fe", n_+"_temporal_att",
                                                               "dtd", "classifier", "ocr_loss");
            rdict[n]=anchor_training_routine;
    return rnams,rdict;

def get_base_hydrav1x3(anchors,qprefix):
    rnams=[];
    rdict={};
    for n_ in anchors["names"]:
        for i in range(3):
            n=str(i)+n_
            rnams.append(n)
            anchor_training_routine = get_tranining_routine_fpbp(n+"_", qprefix+n_, n+"_pred", n+"_loss",
                                                               "sampler", "mvn_core", "prototyper", n_+"_word_fe", n_+"_temporal_att",
                                                               "dtd", "classifier", "ocr_loss");
            rdict[n]=anchor_training_routine;
    return rnams,rdict;

def get_hydra_executor_core(params):
    d={
        "agent": neko_keyword_selective_execution_agent,
        "params": {
            "selector_name":params["iocvt_dict"]["selector_name"],
            "agent_list": [],
        }
    }
    for n in params["anchors"]:
        d["params"]["agent_list"].append(n);
        d["params"][n]=get_ocr_core(params["iocvt_dict"]["raw_image_name"], params["iocvt_dict"]["len_pred_argmax_name"],
                     params["iocvt_dict"]["tdict_name"], params["iocvt_dict"]["proto_label_name"],
                     params["iocvt_dict"]["tensor_proto_vec_name"], params["iocvt_dict"]["logit_name"],
                     params["iocvt_dict"]["len_pred_logits_name"],params["iocvt_dict"]["len_pred_argmax_name"], params["iocvt_dict"]["pred_text_name"],
                     params["iocvt_dict"]["tensor_image_name"],
                     params["iocvt_dict"]["word_feature_name"],
                     params["iocvt_dict"]["attention_map_name"],
                     params["iocvt_dict"]["feat_seq_name"],
                     "mvn_core",  n + "_word_fe",  n + "_temporal_att", "dtd", "classifier");
    return d;
def arm_test_hydra_iocvt(params,prefix):
    params["iocvt_dict"][dvn.raw_image_name]=prefix+dvn.raw_image;
    params["iocvt_dict"][dvn.raw_beacon_name]=prefix+dvn.raw_beacon;
    params["iocvt_dict"][dvn.raw_bmask_name]=prefix+dvn.raw_bmask;
    params["iocvt_dict"][dvn.raw_label_name]= prefix + dvn.raw_label;
    params["iocvt_dict"][dvn.raw_image_size_name]=prefix+dvn.raw_image_size;
    params["iocvt_dict"][dvn.raw_id_name]=prefix+dvn.raw_id_name;
    params["iocvt_dict"][dvn.tensor_proto_img_name]=prefix+dvn.tensor_proto_img;

    params["iocvt_dict"]["selector_name"]=prefix+"head_selector"

    params["iocvt_dict"]["tensor_proto_vec_name"]=prefix+"tensor_proto_vec";
    params["iocvt_dict"]["proto_label_name"]=prefix+"plabel";
    params["iocvt_dict"][dvn.gtdict_name]=prefix+"gtdict";
    params["iocvt_dict"][dvn.tdict_name]=prefix+"tdict";
    params["iocvt_dict"]["global_proto_label_name"]=prefix+"gplabel";
    params["iocvt_dict"]["logit_name"]=prefix+"logit";
    params["iocvt_dict"]["len_pred_logits_name"]=prefix+"len_pred_logits";
    params["iocvt_dict"]["len_pred_argmax_name"]=prefix+"len_pred_argmax";
    params["iocvt_dict"]["tensor_image_name"]=prefix+"tensor_image";
    params["iocvt_dict"]["tensor_size_name"]=prefix+"tensor_size";
    params["iocvt_dict"]["tensor_beacon_name"]=prefix+"tensor_beacon";

    params["iocvt_dict"]["word_feature_name"]=prefix+"word_feature";
    params["iocvt_dict"]["attention_map_name"]=prefix+"attention_map";
    params["iocvt_dict"]["feat_seq_name"]=prefix+"feat_seq";
    params["iocvt_dict"]["pred_text_name"]=prefix+"pred_text";
    params["iocvt_dict"]["selector_name"]=prefix+"selected_head";
    params["iocvt_dict"]["word_feature_name"]=prefix+"word_feature";
    return params

def get_test_hydra_core_routine(params):
    prefix = neko_get_arg("prefix",params,"");
    params=arm_test_hydra_iocvt(params,prefix);

    params["modcvt_dict"]["proto_mvn_name"]=prefix+"mvn_core";

    params["modcvt_dict"]["prototyper_name"]=prefix+"prototyper";
    params["core"]=get_hydra_executor_core({
                "anchors":params["anchors"],
                "iocvt_dict": params["iocvt_dict"],
                "modcvt_dict": params["modcvt_dict"]
            });
    params["tester"]=get_tester_hydra(params);
    params["reporters"]={};
    return {
        "agent":neko_test_mission_agent,
        "params":params
    };

def get_test_hydra_routine(params):
    cfg=get_test_hydra_core_routine(params);
    cfg["params"]["reporters"]= get_reporters(params);
    return cfg;

def get_test_hydra_routine_mo(params):
    cfg=get_test_hydra_core_routine(params);
    cfg["params"]["reporters"]=get_mo_reporters(params);
    return cfg;

def get_test_hydra_routine_mos(params):
    cfg=get_test_hydra_core_routine(params);
    cfg["params"]["reporters"] = get_result_logging_agent(params["iocvt_dict"][dvn.pred_text_name],
                                                          params["iocvt_dict"][dvn.raw_label_name],
                                                          params["iocvt_dict"][dvn.raw_image_name],
                                                          params["iocvt_dict"][dvn.tdict_name],
                                                          params["iocvt_dict"][dvn.proto_label_name],
                                                          params["iocvt_dict"][dvn.tensor_proto_img_name],
                                                          params["logto"]);
    cfg["params"]["reporters"]["case_inv_acr_time"]=get_mo_reporters(params)["case_inv_acr_time"];


    return cfg;

def arm_core_modules_hydra_share_bbn(param,modcfgdict,bogocfgdict):
    save_path = neko_get_arg("save_path", param);
    prefix=neko_get_arg("prefix",param,"");
    feat_ch = neko_get_arg("feat_ch", param, 512);
    expf=neko_get_arg("expf",param,1);
    param["feat_ch"] = feat_ch;
    backbone_core_name = "backbone_core";
    modcfgdict, bogocfgdict=arm_fe_base(param, modcfgdict, bogocfgdict, prefix, save_path);

    modcfgdict, bogocfgdict = arm_dom_fe({"backbone_core_name": backbone_core_name, "fe_name":  "word_fe","expf":expf},
                                         modcfgdict, bogocfgdict, prefix + "word_", save_path);
    modcfgdict,bogocfgdict=arm_imvn(param, modcfgdict, bogocfgdict, prefix, save_path);
    modcfgdict, bogocfgdict = arm_stem_no_fe(param, modcfgdict, bogocfgdict, prefix, save_path)

    for k in param["adict"]["names"]:
        modcfgdict,bogocfgdict=arm_dom_tatt(param,modcfgdict, bogocfgdict, prefix+k+"_", save_path);

    return modcfgdict,bogocfgdict;

def get_hydra_v1_modules_share_bbn(anchors,save_path,log_path,meta_path,decay=0.0005,expf=1):
    modset = neko_module_opt_setNG();
    modcfgs = {};
    bogocfgs = {};
    modcfgs, bogocfgs = arm_core_modules_hydra_share_bbn(
        {
            "expf":expf,
            "adict":anchors,
            "dataroot": find_data_root(),
            "save_path": save_path,
        }, modcfgs, bogocfgs);
    modcfgs, bogocfgs = arm_base_training_modules({
        "dataroot": find_data_root(),
        "save_path": save_path,
        "log_path": log_path,
        "meta_path": meta_path},
        modcfgs, bogocfgs);
    modcfgs, bogocfgs = arm_ocr_loss({
        "dataroot": find_data_root(),
        "save_path": save_path
    },
        modcfgs, bogocfgs)

    modset.arm_modules(modcfgs, bogocfgs,decay_override=decay);
    print(modcfgs, bogocfgs);

    moddict = get_modular_dict(modset);
    print(moddict.keys());
    return modset;
