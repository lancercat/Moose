from osocrNG.modular_agents_ocrNG.ocr_data_agents.neko_sampler import neko_label_sampler_agent
from osocrNG.names import default_ocr_variable_names as dvn
def get_prototype_sampling(label_name,tdict_name, plabel_name,gtdict_name,gplabel_name,tensor_proto_img_name,tensor_label_name,
                           tensor_global_label_name,tensor_gt_length_name,
                           sampler_name,proto_mvn_name):
    return {
        "agent":neko_label_sampler_agent,
        "params":{
            "iocvt_dict": {
                neko_label_sampler_agent.INPUT_label_name: label_name,
                neko_label_sampler_agent.OUTPUT_tdict_name:tdict_name,
                neko_label_sampler_agent.OUTPUT_gtdict_name:gtdict_name,
                neko_label_sampler_agent.OUTPUT_plabel_name:plabel_name,
                neko_label_sampler_agent.OUTPUT_gplabel_name:gplabel_name,
                neko_label_sampler_agent.OUTPUT_tensor_label_name:tensor_label_name,
                neko_label_sampler_agent.OUTPUT_tensor_global_label_name:tensor_global_label_name,
                neko_label_sampler_agent.OUTPUT_tensor_proto_img_name: tensor_proto_img_name,
                neko_label_sampler_agent.OUTPUT_tensor_gt_length_name:tensor_gt_length_name,
            },
            "modcvt_dict": {
                "sampler_name": sampler_name,
                "protomvn": proto_mvn_name
            }
        }
    }
    pass;
