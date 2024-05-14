from neko_2021_mjt.modulars.neko_inflater import neko_inflater
from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment


# We will separate seq with att, as att is not affected by the training state now.

class neko_proto_aggr(neko_module_wrapping_agent):
    def set_mod_io(this,iocvt_dict,modcvt_dict):
        this.input_dict.feature=neko_get_arg("feature_name",iocvt_dict);
        this.output_dict.feat_seq=neko_get_arg("feat_name",iocvt_dict,"proto");
        this.mnames.seq=neko_get_arg("aggr_name",modcvt_dict,"aggr");

    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        # if we don't know for sure how long it is, we guess.
        out_emb = environment.module_dict[this.input_dict.seq_name](
            workspace.inter_dict[this.input_dict.feature],
            workspace.inter_dict[this.input_dict.attmap],
            workspace.inter_dict[this.input_dict.length]
        );
        fout_emb, _ = neko_inflater.inflate(out_emb, workspace.inter_dict[this.input_dict.length]);
        workspace.inter_dict[this.output_dict.feat_seq]=fout_emb;
        return workspace;
