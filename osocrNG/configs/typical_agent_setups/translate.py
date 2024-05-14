from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_agent_wrapping_agent
from osocrNG.configs.typical_agent_setups.os_pred import get_pred_agent, get_translate_agent


def get_pred_and_translate(feat_seq_name,tensor_proto_vec_name,proto_label_name,logit_name,tdict_name,length_name,pred_text_name,
                           pred_name):
    return {
        "agent":neko_agent_wrapping_agent,
        "params":{
            "agent_list":["pred","translate"],
            "pred":get_pred_agent(feat_seq_name,tensor_proto_vec_name,proto_label_name,logit_name,pred_name),
            "translate":get_translate_agent(logit_name,tdict_name,length_name,pred_text_name)
        }
    }
