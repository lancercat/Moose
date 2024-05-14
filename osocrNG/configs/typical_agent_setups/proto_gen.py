from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_agent_wrapping_agent
from neko_sdk.neko_framework_NG.agents.neko_vis_prototyper import neko_vis_prototyper_agent
from osocrNG.configs.typical_agent_setups.proto_sampler import get_prototype_sampling


def get_prototyping_agent(tensor_proto_img_name,tensor_proto_vec_name, prototyper_name):
    return {
        "agent":neko_vis_prototyper_agent,
        "params":{
            "iocvt_dict": {
                "tensor_proto_img_name":tensor_proto_img_name,
                "tensor_proto_vec_name":tensor_proto_vec_name,
            },
            "modcvt_dict": {
                "prototyper_name": prototyper_name,
            }
        }
    }

def get_training_prototyping_agent(label_name,tdict_name, plabel_name,gtdict_name,gplabel_name,tensor_proto_img_name,
                                   tensor_label_name,tensor_global_label_name,tensor_gt_length_name,tensor_proto_vec_name,
                           sampler_name,proto_mvn_name,prototyper_name):
    return {
        "agent": neko_agent_wrapping_agent,
        "params":{
            "agent_list":["sampler","encoder"],
            "sampler": get_prototype_sampling(label_name,tdict_name, plabel_name,gtdict_name,gplabel_name,tensor_proto_img_name,
                                              tensor_label_name,tensor_global_label_name,tensor_gt_length_name,
                                             sampler_name,proto_mvn_name),
            "encoder":get_prototyping_agent(tensor_proto_img_name,tensor_proto_vec_name,prototyper_name)
        }
    }
