from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment


class neko_temporal_attention(neko_module_wrapping_agent):
    def set_mod_io(this,iocvt_dict,modcvt_dict):
        this.input_dict.feature=neko_get_arg("feature_name",iocvt_dict,"feature");
        this.output_dict.att_map=neko_get_arg("att_map_name",iocvt_dict,"att_map");
        this.output_dict.len_pred_logits_name=neko_get_arg("len_pred_logits_name",iocvt_dict,"len_pred_logits");
        this.output_dict.len_pred_argmax_name=neko_get_arg("len_pred_argmax_name",iocvt_dict,"len_pred_map")
        this.mnames.att_mod_name=neko_get_arg("temporal_att_name",modcvt_dict,"temporal_att");

    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        A, pred_length = environment.module_dict[this.mnames.att_mod_name](workspace.inter_dict[this.input_dict.feature]);
        workspace.inter_dict[this.output_dict.att_map]=A;
        workspace.inter_dict[this.output_dict.len_pred_logits_name]=pred_length;
        workspace.inter_dict[this.output_dict.len_pred_argmax_name]=pred_length.argmax(dim=-1);
        return workspace;
