from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_agent_wrapping_agent
from neko_sdk.neko_framework_NG.agents.neko_mvn_agent import get_neko_mvn_agent
from osocrNG.configs.typical_agent_setups.aggr import arm_temporal_aggr
from osocrNG.configs.typical_agent_setups.fe import get_origin_fe
from osocrNG.configs.typical_agent_setups.translate import get_pred_and_translate
from osocrNG.modular_agents_ocrNG.att_agents.temporal_att_agent import neko_temporal_attention


def get_temporal_att(feature_name,len_pred_logits_name,len_pred_argmax_name,attention_map_name,temporal_att_name):
    return {
        "agent":neko_temporal_attention,
        "params":{
            "iocvt_dict": {
                "feature_name": feature_name,
                "att_map_name": attention_map_name,
                "len_pred_logits_name":len_pred_logits_name,
                "len_pred_argmax_name":len_pred_argmax_name,
            },
            "modcvt_dict": {
                "temporal_att_name": temporal_att_name,
            }
        }
    }

def get_ocr_extractor(raw_image_name,decode_length_tensor_name,len_pred_logits_name,len_pred_argmax_name,
                 tensor_image_name,word_feature_name,attention_map_name,feat_seq_name,
                mvn_mod_name, word_fe_mod_name, temporal_att_mod_name, seq_mod_name
):


    return {
        "agent":neko_agent_wrapping_agent,
        "params":{
            "agent_list":["mvn","fe","att","aggr"],
            "mvn":get_neko_mvn_agent([raw_image_name],[tensor_image_name],mvn_mod_name),
            "fe": get_origin_fe(tensor_image_name, word_feature_name, word_fe_mod_name),
            "att": get_temporal_att(word_feature_name, len_pred_logits_name, len_pred_argmax_name, attention_map_name,
                                    temporal_att_mod_name),
            "aggr": arm_temporal_aggr(word_feature_name, decode_length_tensor_name, attention_map_name, feat_seq_name, seq_mod_name),
        }
    }


def get_ocr_core(raw_image_name,decode_length_tensor_name,tdict_name,proto_label_name,tensor_proto_vec_name,logit_name,len_pred_logits_name,len_pred_argmax_name,pred_text_name,
                 tensor_image_name,word_feature_name,attention_map_name,feat_seq_name,
                mvn_mod_name, word_fe_mod_name, temporal_att_mod_name, seq_mod_name, pred_mod_name,
):

    cd=get_ocr_extractor(raw_image_name,decode_length_tensor_name,len_pred_logits_name,len_pred_argmax_name,
                 tensor_image_name,word_feature_name,attention_map_name,feat_seq_name,
                mvn_mod_name, word_fe_mod_name, temporal_att_mod_name, seq_mod_name);
    cd["params"]["agent_list"].append("pred_translate");
    cd["params"]["pred_translate"]=get_pred_and_translate(feat_seq_name,tensor_proto_vec_name,proto_label_name,logit_name,tdict_name,decode_length_tensor_name,pred_text_name,pred_mod_name);
    return cd

def get_ocr_msr_core(
        raw_image_name,raw_bmask_name,decode_length_tensor_name,tdict_name,proto_label_name,tensor_proto_vec_name,
        logit_name,len_pred_logits_name,len_pred_argmax_name,pred_text_name,
        tensor_image_name,bmask_tensor_name,word_feature_name,stat_name,attention_map_name,feat_seq_name,
        mvn_mod_name,bmvn_mod_name, word_fe_mod_name, temporal_att_mod_name, seq_mod_name, pred_mod_name,
):
    return {
        "agent":neko_agent_wrapping_agent,
        "params":{
            "agent_list":["mvn","bmvn","fe","att","aggr","pred_translate"],
            "mvn":get_neko_mvn_agent([raw_image_name],[tensor_image_name],mvn_mod_name),
            "bmvn":get_neko_mvn_agent([raw_bmask_name],[bmask_tensor_name],bmvn_mod_name),
            "fe": get_msr_fe(tensor_image_name,bmask_tensor_name, word_feature_name,stat_name, word_fe_mod_name),
            "att": get_temporal_att(word_feature_name, len_pred_logits_name, len_pred_argmax_name, attention_map_name,
                                    temporal_att_mod_name),
            "aggr": arm_temporal_aggr(word_feature_name, decode_length_tensor_name, attention_map_name, feat_seq_name, seq_mod_name),
            "pred_translate":get_pred_and_translate(feat_seq_name,tensor_proto_vec_name,proto_label_name,logit_name,tdict_name,decode_length_tensor_name,pred_text_name,pred_mod_name)
        }
    }