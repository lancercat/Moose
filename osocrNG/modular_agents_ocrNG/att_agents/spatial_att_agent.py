from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment


class neko_spatial_attention(neko_module_wrapping_agent):
    def set_mod_io(this,iocvt_dict,modcvt_dict):
        this.input_dict.feature=neko_get_arg("proto_feature_name",iocvt_dict,"proto_feature");
        this.output_dict.att_map=neko_get_arg("proto_att_map_name",iocvt_dict,"proto_att_map");
        this.mnames.att_mod_name=neko_get_arg("spatial_att_name",modcvt_dict,"spatial_att");

    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        A = environment.module_dict[this.mnames.att_mod_name](workspace.inter_dict[this.input_dict.feature]);
        workspace.inter_dict[this.output_dict.att_map]=A;
        return workspace;
