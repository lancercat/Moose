from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from neko_sdk.ocr_modules.io.encdec import decode_prob;
from osocrNG.names import default_ocr_variable_names as dvn
import torch
class simple_pred_agent(neko_module_wrapping_agent):
    MOD_pred_name="pred_name";
    INPUT_feat_seq_name=dvn.feat_seq_name;
    INPUT_tensor_proto_vec_name=dvn.tensor_proto_vec_name;
    INPUT_proto_label_name=dvn.proto_label_name;
    OUTPUT_logit_name=dvn.logit_name;

    def set_mod_io(this,iocvt_dict,modcvt_dict):
        this.pred = this.register(this.MOD_pred_name,modcvt_dict,this.mnames);

        this.feat_seq_name=this.register(this.INPUT_feat_seq_name,iocvt_dict,this.input_dict);
        this.tensor_proto_vec_name=this.register(this.INPUT_tensor_proto_vec_name,iocvt_dict,this.input_dict);

        this.proto_label_name=this.register(this.INPUT_proto_label_name,iocvt_dict,this.input_dict);
        this.logit_name=this.register(this.OUTPUT_logit_name,iocvt_dict,this.output_dict);

    def take_action(this, workspace:neko_workspace,environment:neko_environment):
        workspace.inter_dict[this.logit_name] = environment.module_dict[this.pred](
            workspace.inter_dict[this.feat_seq_name],
            workspace.inter_dict[this.tensor_proto_vec_name],
            workspace.inter_dict[this.proto_label_name]
        );
        return workspace;

class vbd_pred_ignore_agent(neko_module_wrapping_agent):
    MOD_pred_name="pred_name";
    INPUT_feat_seq_name=dvn.feat_seq_name;
    INPUT_rotated_tensor_proto_vec_name=dvn.rotated_tensor_proto_vec_name;
    INPUT_proto_label_name=dvn.proto_label_name;
    OUTPUT_logit_name=dvn.logit_name;
    PARAM_possible_rotation="possible_rotation";

    def set_etc(this,param):
        this.possible_rotation=neko_get_arg(this.PARAM_possible_rotation,param,[0]);
    def set_mod_io(this,iocvt_dict,modcvt_dict):
        this.pred = this.register(this.MOD_pred_name,modcvt_dict,this.mnames);

        this.feat_seq_name=this.register(this.INPUT_feat_seq_name,iocvt_dict,this.input_dict);
        this.rotated_tensor_proto_vec_name=this.register(this.INPUT_rotated_tensor_proto_vec_name,iocvt_dict,this.input_dict);
        this.proto_label_name=this.register(this.INPUT_proto_label_name,iocvt_dict,this.input_dict);
        this.logit_name=this.register(this.OUTPUT_logit_name,iocvt_dict,this.output_dict);

    def take_action(this, workspace:neko_workspace,environment:neko_environment):
        protos=[];
        plables=[];

        for k in this.possible_rotation:
            if (len(plables)):
                plables[-1] = plables[-1][:protos[-1].shape[0]]; # drop centerless sp tokens.
            protos.append(workspace.inter_dict[this.rotated_tensor_proto_vec_name][k]);

            plables.append(workspace.inter_dict[this.proto_label_name]);
        # if(workspace.inter_dict["selector"]=="vert"):
        #     pass;

        workspace.inter_dict[this.logit_name] = environment.module_dict[this.pred](
            workspace.inter_dict[this.feat_seq_name],
            torch.cat(protos,0),
            torch.cat(plables,0)
        );
        return workspace;
# it has no state(unlike loggers with histories), so it does not have a module.
class translate_agent(neko_module_wrapping_agent):
    def set_mod_io(this,iocvt_dict,modcvt_dict):
        this.input_dict.logit_name=neko_get_arg(dvn.logit_name,iocvt_dict);
        this.input_dict.tdict_name=neko_get_arg(dvn.tdict_name,iocvt_dict);
        this.input_dict.length_name=neko_get_arg(dvn.length_name,iocvt_dict);
        this.output_dict.pred_text_name=neko_get_arg(dvn.pred_text_name,iocvt_dict);
    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        try:
            outpred=decode_prob(
                workspace.inter_dict[this.input_dict.logit_name],
                workspace.inter_dict[this.input_dict.length_name],
                workspace.inter_dict[this.input_dict.tdict_name]
            )[0];
        except:
            print("badpred");
            # print(workspace.inter_dict);
            outpred=["BADPRED"];
        workspace.inter_dict[this.output_dict.pred_text_name]=outpred;
        pass;
