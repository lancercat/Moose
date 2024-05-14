from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_agent_wrapping_agent
from neko_sdk.neko_framework_NG.agents.neko_vis_prototyper_mk2 import neko_vis_prototyper_agent_mk2, \
    neko_testing_prototyper_agent_mk2
from osocrNG.configs.typical_agent_setups.proto_sampler import get_prototype_sampling


def get_prototyping_agent_mk2(tensor_proto_img_name,fe_name, character_fe_mod_name,character_att_mod_name,character_aggr_mod_name):
    return {
        "agent":neko_vis_prototyper_agent_mk2,
        "params":{
            "iocvt_dict": {
                "tensor_proto_img_name":tensor_proto_img_name,
                "tensor_proto_vec_name":fe_name,
            },
            "modcvt_dict": {
                "fe_name": character_fe_mod_name,
                "att_name" :character_att_mod_name,
                "aggr_name": character_aggr_mod_name,
            }
        }
    }

def get_training_prototyping_agent(label_name,tdict_name, plabel_name,gtdict_name,gplabel_name,tensor_proto_img_name,
                                   tensor_label_name,tensor_global_label_name,tensor_gt_length_name,tensor_proto_vec_name,
                           sampler_name,proto_mvn_name,character_fe_mod_name,character_att_mod_name,character_aggr_mod_name):
    return {
        "agent": neko_agent_wrapping_agent,
        "params":{
            "agent_list":["sampler","encoder"],
            "sampler": get_prototype_sampling(label_name,tdict_name, plabel_name,gtdict_name,gplabel_name,tensor_proto_img_name,
                                              tensor_label_name,tensor_global_label_name,tensor_gt_length_name,
                           sampler_name,proto_mvn_name),
            "encoder":get_prototyping_agent_mk2(tensor_proto_img_name,tensor_proto_vec_name,character_fe_mod_name,character_att_mod_name,character_aggr_mod_name)
        }
    }

def get_testing_prototyping_agent(tdict_name, plabel_name,gtdict_name,gplabel_name,tensor_proto_vec_name,
                           meta_holder_name,proto_mvn_name,character_fe_mod_name,character_att_mod_name,character_aggr_mod_name):
    return {
        "agent":neko_testing_prototyper_agent_mk2,
        "params":{
            "iocvt_dict":{
              "prototype_mvn":proto_mvn_name,
                "tdict_name":tdict_name,
                "plabel_name": plabel_name,
                "gtdict_name": gtdict_name,
                "gplabel_name": gplabel_name,
                "tensor_proto_vec_name":tensor_proto_vec_name,

            },
            "modcvt_dict": {
                "protomvn": proto_mvn_name,
                "fe_name": character_fe_mod_name,
                "att_name" :character_att_mod_name,
                "aggr_name": character_aggr_mod_name,
                "meta_holder_name":meta_holder_name,
            },
            "capacity":512
        }
    }
