
from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment


# We will separate seq with att, as att is not affected by the training state now.
# use gt_length as "length_name" for training and "pred_length" for testing.
class neko_mvn_agent(neko_module_wrapping_agent):
    def set_mod_io(this,iocvt_dict,modcvt_dict):
        this.input_dict.raw_image_names=neko_get_arg("raw_image_names",iocvt_dict);
        this.output_dict.tensor_image_names=neko_get_arg("tensor_image_names",iocvt_dict);
        this.mnames.mvn_mod_name=neko_get_arg("mvn_mod_name",modcvt_dict);

    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        # if we don't know for sure how long it is, we guess.
        for i in range(len(this.input_dict.raw_image_names)):
            iname=this.input_dict.raw_image_names[i];
            oname=this.output_dict.tensor_image_names[i];
            workspace.inter_dict[oname]=environment.module_dict[this.mnames.mvn_mod_name](workspace.inter_dict[iname]);
        return workspace;

def get_neko_mvn_agent(raw_image_names,tensor_image_names,mvn_mod_name):
    return {
        "agent":neko_mvn_agent,
        "params":{
            "iocvt_dict":{
                "raw_image_names":raw_image_names,
                "tensor_image_names":tensor_image_names,
            },
            "modcvt_dict":
            {
                "mvn_mod_name": mvn_mod_name,
            }
        }
    }