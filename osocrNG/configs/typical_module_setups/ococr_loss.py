from neko_sdk.cfgtool.argsparse import neko_get_arg
from osocrNG.trainable_lossNG.os_clsloss import osocrlossNG


def config_ocr_loss(param,cfg_dict,path,name):
    mod_param = {
        "save_each": neko_get_arg("save_each", param, 20000),
        "save_path": neko_get_arg("save_path",param,path),
        "save_name": neko_get_arg("save_name",param,name),
        "mod_type": osocrlossNG,
        "mod_param": {
        }
    }
    cfg_dict[name]=mod_param;
    return cfg_dict;
def arm_ocr_loss(param,modcfgdict,bogocfgdict):
    save_path = neko_get_arg("save_path", param);
    prefix = neko_get_arg("prefix", param, "");
    ocr_loss_name = prefix + neko_get_arg("ocr_loss_name", param, "ocr_loss");
    ocr_loss_param = neko_get_arg("ocr_loss_param", param, {});
    modcfgdict = config_ocr_loss(ocr_loss_param, modcfgdict, save_path, ocr_loss_name);

    return modcfgdict,bogocfgdict;


def c(param,cfg_dict,path,name):
    mod_param = {
        "save_each": neko_get_arg("save_each", param, 20000),
        "save_path": neko_get_arg("save_path", param, path),
        "save_name": neko_get_arg("save_name", param, name),
        "mod_type": neko_paranormal_student_loss,
        "mod_param": {
        }
    }
    cfg_dict[name] = mod_param;
    return cfg_dict;

def arm_ocr_loss_student(param,modcfgdict,bogocfgdict):
    save_path = neko_get_arg("save_path", param);
    prefix = neko_get_arg("prefix", param, "");
    ocr_att_loss_name = prefix + neko_get_arg("ocr_att_loss_name", param, "ocr_att_loss");
    ocr_att_loss_param = neko_get_arg("ocr_att_loss_param", param, {});
    modcfgdict = config_ocr_student_loss(ocr_att_loss_param, modcfgdict, save_path, ocr_att_loss_name);

    return modcfgdict,bogocfgdict;