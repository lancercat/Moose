from osocrNG.modular_agents_ocrNG.aggrate_agents.feat_aggr_agents import neko_word_aggr
def arm_temporal_aggr(word_feature_name,length_name,attention_map_name,feat_seq_name,seq_mod_name):
    cfg={
        "agent":neko_word_aggr,
        "params":{
            "iocvt_dict": {
                "feature_name":word_feature_name,
                "length_name":length_name,
                "attention_map_name":attention_map_name,
                "feat_seq_name":feat_seq_name,
            },
            "modcvt_dict": {
                "seq_mod_name": seq_mod_name,
            }
        }
    }
    return cfg;