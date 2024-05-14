from osocrNG.modular_agents_ocrNG.losses.recog_loss import recog_loss_agent


def get_ocr_loss_agent(cls_label_name,cls_logit_name,len_logit_name,len_label_name,ocr_loss_name,
                       osocr_loss_mod_name):
    return {
        "agent":recog_loss_agent,
        "params":{
            "iocvt_dict":{
              "cls_label_name":cls_label_name,
              "cls_logit_name":cls_logit_name,
              "len_logit_name":len_logit_name,
              "len_label_name":len_label_name,
              "ocr_loss_name":ocr_loss_name,
            },
            "modcvt_dict": {
                "osocr_loss_mod_name":osocr_loss_mod_name,
            }
        }
    }





