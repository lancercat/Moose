from osocrNG.modular_agents_ocrNG.output_logging_subs.basic_acr_fps_reporter import case_inv_acr_fps_reporter,case_inv_multi_orientation_acr_fps_reporter,neko_debugging_image_pred_logging_reporter
from osocrNG.modular_agents_ocrNG.output_logging_subs.neko_ocr_pred_render_agents import neko_image_pred_render_agent
from osocrNG.modular_agents_ocrNG.output_logging_subs.result_export import neko_img_saver_agent,neko_text_saving_agent

def get_reporters(param):
    return{
        "case_inv_acr_time":
            {
                "agent":case_inv_acr_fps_reporter,
                "params":{
                    "iocvt_dict":param["iocvt_dict"],
                    "modcvt_dict": param["modcvt_dict"],

                }
            }
        };
def get_mo_reporters(param):
    return{
        "case_inv_acr_time":
            {
                "agent":case_inv_multi_orientation_acr_fps_reporter,
                "params":{
                    "iocvt_dict":param["iocvt_dict"],
                    "modcvt_dict": param["modcvt_dict"],

                }
            }
        };
def get_img_render_agent(pred_text,text_label,raw_image,tdict,plabel,tensor_proto_img,rendered_result,dict_path=None):
    return {
            "agent":neko_image_pred_render_agent,
            "params":{
                neko_image_pred_render_agent.PARAM_pad_value:127,
                neko_image_pred_render_agent.PARAM_test_meta_path: None,
                "iocvt_dict":{
                    neko_image_pred_render_agent.INPUT_pred_text:pred_text,
                    neko_image_pred_render_agent.INPUT_text_label: text_label,
                    neko_image_pred_render_agent.INPUT_raw_image: raw_image,
                    neko_image_pred_render_agent.INPUT_tdict: tdict,
                    neko_image_pred_render_agent.INPUT_proto_label: plabel,
                    neko_image_pred_render_agent.INPUT_tensor_proto_img: tensor_proto_img,
                    neko_image_pred_render_agent.OUTPUT_rendered_result: rendered_result,
                },
                "modcvt_dict": {

                }
            }
    };

def get_img_saving_agent(imgs,raws,export_root,prefx):
    return {
            "agent":neko_img_saver_agent,
            "params":{
                neko_img_saver_agent.PARAM_prefix:prefx,
                neko_img_saver_agent.PARAM_save_path:export_root,
                "iocvt_dict":{
                    neko_img_saver_agent.INPUT_img_list: imgs,
                    neko_img_saver_agent.INPUT_raw_image:raws,
                },
                "modcvt_dict": {

                }
            }
        };

def get_text_saving_agent(texts,export_root,prefx):
    return {
            "agent":neko_text_saving_agent,
            "params":{
                neko_text_saving_agent.PARAM_prefix:prefx,
                neko_text_saving_agent.PARAM_save_path:export_root,
                "iocvt_dict": {
                    neko_text_saving_agent.INPUT_text_list: texts
                },
                "modcvt_dict": {

                }
            }
    };


def get_result_logging_agent(pred_text,text_label,raw_image,tdict,plabel,tensor_proto_img,export_root,dict_path=None):
    return \
        {
            "result_renderer": get_img_render_agent(pred_text, text_label, raw_image, tdict, plabel,
                                                        tensor_proto_img, "rendered_result",dict_path),
            "result_im_saver": get_img_saving_agent("rendered_result",raw_image, export_root, "pred"),
            "result_txt_saver": get_text_saving_agent(pred_text, export_root, "pred"),
            "result_gt_saver": get_text_saving_agent(text_label,export_root,"gt"),
        }



def get_debugg_reporter(param):
    return {
        "case_inv_acr_time":
            {
                "agent": case_inv_acr_fps_reporter,
                "params": {
                    "iocvt_dict": param["iocvt_dict"],
                    "modcvt_dict": param["modcvt_dict"],

                }
            },
        "debugging_logger":
            {
                "agent": neko_debugging_image_pred_logging_reporter,
                "params": {
                    "iocvt_dict": param["iocvt_dict"],
                    "modcvt_dict": param["modcvt_dict"],
                    "save_path":param["save_path"],
                    "meta_path":param["meta_path"]
                }
            }
    };
