from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment


# You need no loss during evaluation, so we give the choice of skipping back to you :-)
class recog_loss_agent(neko_module_wrapping_agent):
    def set_mod_io(this,iocvt_dict,modcvt_dict):
        # Yup, classification of length and recognition, nothing more nothing less.
        # This stub sticks to basic.
        # If you need more, go grab a regularization subroutine
        this.mnames.clsloss=neko_get_arg("osocr_loss_mod_name",modcvt_dict);
        this.input_dict.clslabel=neko_get_arg("cls_label_name",iocvt_dict,"cls_label");
        this.input_dict.lenlabel=neko_get_arg("len_label_name",iocvt_dict,"len_label");
        this.input_dict.clslog=neko_get_arg("cls_logit_name",iocvt_dict,"cls_logit");
        this.input_dict.lenlog=neko_get_arg("len_logit_name",iocvt_dict,"len_logit");
        this.output_dict.ocrloss=neko_get_arg("ocr_loss_name",iocvt_dict,"ocr_loss");
        this.loss_name=neko_get_arg("loss_name",iocvt_dict,"loss");
    def take_action(this, workspace:neko_workspace,environment:neko_environment):
        ocrloss,terms=environment.module_dict[this.mnames.clsloss](
            workspace.inter_dict[this.input_dict.clslog],
            workspace.inter_dict[this.input_dict.lenlog],
            workspace.inter_dict[this.input_dict.clslabel].detach(),
            workspace.inter_dict[this.input_dict.lenlabel].detach(),
        );
        workspace.objdict[this.output_dict.ocrloss]=ocrloss;
        workspace.logdict[this.output_dict.ocrloss]=terms;

        return workspace;
