
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_agent_wrapping_agent
from neko_sdk.neko_framework_NG.agents.neko_loss_backward_all_agent import get_neko_basic_backward_all_agent
from osocrNG.configs.typical_agent_setups.data_ferrier import get_osocr_data_ferrier
from osocrNG.configs.typical_agent_setups.ocr_agents.conventional_ocr_core import get_ocr_core
from osocrNG.configs.typical_agent_setups.osocr_loss import get_ocr_loss_agent
from osocrNG.configs.typical_agent_setups.proto_gen import get_training_prototyping_agent
from osocrNG.names import default_ocr_variable_names as dvn

def get_tranining_routine(prefix,ocr_date_queue_name,pred_text_name,loss_name,
                          sampler_mod_name,mvn_mod_name,prototyper_mod_name,word_fe_mod_name,
                          temporal_att_mod_name,seq_mod_name,pred_mod_name, ocr_loss_mod_name ):
    raw_image_name=prefix+dvn.raw_image;
    raw_beacon_name=prefix+dvn.raw_beacon;
    raw_bmask_name=prefix+dvn.raw_bmask;
    raw_label_name = prefix + dvn.raw_label;
    raw_image_size_name=prefix+dvn.raw_image_size;



    tensor_gt_length_name=prefix+dvn.gt_length;
    tensor_label_name=prefix+"tensor_label";
    tensor_global_label_name=prefix+"tensor_global_label";
    tensor_proto_vec_name=prefix+"tensor_proto_vec";
    tensor_proto_img_name=prefix+"tensor_proto_img";
    proto_label_name=prefix+"plabel";
    gtdict_name=prefix+"gtdict";
    tdict_name=prefix+"tdict";
    global_proto_label_name=prefix+"gplabel";
    logit_name=prefix+"logit";
    len_pred_logits_name=prefix+"len_pred";
    tensor_image_name=prefix+dvn.tensor_image;
    attention_map_name=prefix+"attention_map";
    feat_seq_name=prefix+"feature_sequence";
    len_pred_argmax_name=prefix+"len_pred_argmax";# We won't use it either way


    return {
        "agent": neko_agent_wrapping_agent,
        "params": {
            "agent_list": ["init_workspace","prototyper", "core", "loss"],
            "init_workspace":get_osocr_data_ferrier(ocr_date_queue_name,raw_image_name,raw_beacon_name,raw_image_size_name,raw_bmask_name,raw_label_name),
            "prototyper": get_training_prototyping_agent(
                raw_label_name,tdict_name,proto_label_name,gtdict_name,global_proto_label_name,tensor_proto_img_name,
                tensor_label_name,tensor_global_label_name,tensor_gt_length_name,tensor_proto_vec_name,
                sampler_mod_name,mvn_mod_name,prototyper_mod_name),
            "core": get_ocr_core(raw_image_name, tensor_gt_length_name, tdict_name, proto_label_name,
                                 tensor_proto_vec_name, logit_name, len_pred_logits_name,len_pred_argmax_name, pred_mod_name, tensor_image_name,
                                 attention_map_name, feat_seq_name, pred_text_name, mvn_mod_name, word_fe_mod_name,
                                 temporal_att_mod_name, seq_mod_name,pred_mod_name),
            "loss":get_ocr_loss_agent(tensor_label_name,logit_name,len_pred_logits_name,tensor_gt_length_name,loss_name,ocr_loss_mod_name)
        }
    };



# When you cannot afford backpropagation after everything settles and decide not to let other routine to use intermediate variables for training.
def get_tranining_routine_fpbp(prefix,ocr_date_queue_name,pred_text_name,loss_name,
                          sampler_mod_name,mvn_mod_name,prototyper_mod_name,word_fe_mod_name,
                          temporal_att_mod_name,seq_mod_name,pred_mod_name, ocr_loss_mod_name ):
    cfg=get_tranining_routine(prefix, ocr_date_queue_name, pred_text_name, loss_name,
                              sampler_mod_name, mvn_mod_name, prototyper_mod_name, word_fe_mod_name,
                              temporal_att_mod_name, seq_mod_name, pred_mod_name, ocr_loss_mod_name);

    cfg["params"]["agent_list"].append("loss_bp");
    cfg["params"]["loss_bp"]=get_neko_basic_backward_all_agent();
    return cfg;

def get_tranining_data_vis_routine_fpbp(prefix,ocr_date_queue_name,pred_text_name,loss_name,
                          sampler_mod_name,mvn_mod_name,prototyper_mod_name,word_fe_mod_name,
                          temporal_att_mod_name,seq_mod_name,pred_mod_name, ocr_loss_mod_name ):
    cfg=get_tranining_routine(prefix, ocr_date_queue_name, pred_text_name, loss_name,
                              sampler_mod_name, mvn_mod_name, prototyper_mod_name, word_fe_mod_name,
                              temporal_att_mod_name, seq_mod_name, pred_mod_name, ocr_loss_mod_name);

    cfg["params"]["agent_list"].append("loss_bp");
    cfg["params"]["loss_bp"]=get_neko_basic_backward_all_agent();
    return cfg;
