from osocrNG.modular_agents_ocrNG.output_logging_subs.basic_acr_fps_reporter import neko_debugging_image_pred_logging_reporter


def get_visualize_hydra_routine(prefix,img_name,label_name,
                                tdict_name,proto_tensor_image_name,
                                plabel_name,meta_path="NEP_skipped_NEP"):
    param={};
    param["save_path"]="NEP_skipped_NEP";
    param["meta_path"]=meta_path;
    param["show_name"]=prefix+"Meow";
    param["padvalue"]=127;
    param["iocvt_dict"]={"raw_image":img_name,
                         "text_label":label_name,
                         "tdict":tdict_name,
                         "tensor_proto_img":proto_tensor_image_name,
                         "plabel":plabel_name
                         };
    param["modcvt_dict"]={};


    # I am not into build a hellot testing agents...
    return {
        "agent":neko_debugging_image_pred_logging_reporter,
        "params":param
    }