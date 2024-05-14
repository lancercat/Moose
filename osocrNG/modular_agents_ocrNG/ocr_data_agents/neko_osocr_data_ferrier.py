from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment


class neko_ocr_data_ferrier(neko_module_wrapping_agent):
    def set_mod_io(this,iocvt_dict,modcvt_dict):
        this.input_dict.queue_name=neko_get_arg("queue_name",iocvt_dict);
        this.output_dict.image_name=neko_get_arg("image_name",iocvt_dict);
        this.output_dict.beacon_name = neko_get_arg("beacon_name", iocvt_dict);
        this.output_dict.bmask_name=neko_get_arg("bmask_name",iocvt_dict);
        this.output_dict.label_name = neko_get_arg("label_name", iocvt_dict);
        this.output_dict.size_name=neko_get_arg("size_name",iocvt_dict);
    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        data=environment.queue_dict[this.input_dict.queue_name].get();
        workspace.inter_dict[this.output_dict.image_name]=data["image"];
        workspace.inter_dict[this.output_dict.beacon_name] = data["beacon"];
        workspace.inter_dict[this.output_dict.bmask_name] = data["bmask"];
        workspace.inter_dict[this.output_dict.label_name] = data["label"];
        workspace.inter_dict[this.output_dict.size_name]=data["size"];

