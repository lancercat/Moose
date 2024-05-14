from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from osocrNG.ocr_modules_NG.neko_flatten_NG import neko_flatten_NG_lenpred


# We will separate seq with att, as att is not affected by the training state now.
# use gt_length as "length_name" for training and "pred_length" for testing.
class neko_word_aggr(neko_module_wrapping_agent):
    def set_mod_io(this,iocvt_dict,modcvt_dict):
        this.input_dict.feature=neko_get_arg("feature_name",iocvt_dict);
        this.input_dict.length=neko_get_arg("length_name",iocvt_dict);
        this.input_dict.attmap=neko_get_arg("attention_map_name",iocvt_dict);
        this.output_dict.feat_seq=neko_get_arg("feat_seq_name",iocvt_dict);
        this.mnames.seq_name=neko_get_arg("seq_mod_name",modcvt_dict);

    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        # if we don't know for sure how long it is, we guess.
        out_emb = environment.module_dict[this.mnames.seq_name](
            workspace.inter_dict[this.input_dict.feature][-1],
            workspace.inter_dict[this.input_dict.attmap],
            workspace.inter_dict[this.input_dict.length]);
        fout_emb, _ = neko_flatten_NG_lenpred.inflate(out_emb, workspace.inter_dict[this.input_dict.length]);
        workspace.inter_dict[this.output_dict.feat_seq]=fout_emb;
        return workspace;
