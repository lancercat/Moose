from osocrNG.modular_agents_ocrNG.pred_subs.simple_pred_dict import simple_pred_agent,vbd_pred_ignore_agent,translate_agent


def get_pred_agent(feat_seq_name,tensor_proto_vec_name,proto_label_name,logit_name,
                   pred_name):
    return {
        "agent":simple_pred_agent,
        "params":{
            "iocvt_dict":{
                "feat_seq_name":feat_seq_name,
                "tensor_proto_vec_name":tensor_proto_vec_name,
                "proto_label_name":proto_label_name,
                "logit_name":logit_name
            },
            "modcvt_dict":{
                "pred_name":pred_name,
            }
        }
    }

def get_pred_vbd_ignore_agent(feat_seq_name,tensor_proto_vec_name,rotated_tensor_proto_vec_name,proto_label_name,logit_name,
                   pred_name,possible_rotation):
    return {
        "agent":vbd_pred_ignore_agent,
        "params":{
            vbd_pred_ignore_agent.PARAM_possible_rotation: possible_rotation,
            "iocvt_dict":{
                vbd_pred_ignore_agent.INPUT_feat_seq_name:feat_seq_name,
                vbd_pred_ignore_agent.INPUT_rotated_tensor_proto_vec_name:rotated_tensor_proto_vec_name,
                vbd_pred_ignore_agent.INPUT_proto_label_name:proto_label_name,
                vbd_pred_ignore_agent.OUTPUT_logit_name:logit_name
            },
            "modcvt_dict":{
                vbd_pred_ignore_agent.MOD_pred_name:pred_name,
            }
        }
    }

def get_translate_agent(logit_name,tdict_name,length_name,pred_text_name):
    return {
        "agent":translate_agent,
        "params":{
            "iocvt_dict":{
              "logit_name":logit_name,
              "tdict_name":tdict_name,
              "length_name":length_name,
              "pred_text_name":pred_text_name,
            },
            "modcvt_dict": {
            }
        }
    }
