import torch

from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.workspace import neko_environment, neko_workspace
from neko_sdk.ocr_modules.io.encdec import encode_core
from osocrNG.ocr_modules_NG.neko_flatten_NG import neko_flatten_NG_lenpred
from osocrNG.names import default_ocr_variable_names as dvn


class neko_label_sampler_agent(neko_module_wrapping_agent):
    INPUT_label_name=dvn.raw_label_name;

    OUTPUT_tdict_name=dvn.tdict_name;
    OUTPUT_gtdict_name=dvn.gtdict_name;

    OUTPUT_plabel_name=dvn.proto_label_name;
    OUTPUT_gplabel_name=dvn.global_proto_label_name;

    OUTPUT_tensor_proto_img_name=dvn.tensor_proto_img_name;

    OUTPUT_tensor_label_name=dvn.tensor_label_name;
    OUTPUT_tensor_global_label_name=dvn.tensor_global_label_name;


    OUTPUT_tensor_proto_img_name=dvn.tensor_proto_img_name;
    OUTPUT_tensor_gt_length_name=dvn.tensor_gt_length_name;


    MOD_sampler_name="sampler_name";
    MOD_protomvn="protomvn";
    def set_mod_io(this,iocvt_dict,modcvt_dict):
        this.label=this.register(this.INPUT_label_name,iocvt_dict,this.input_dict);
        
        this.tdict=this.register(this.OUTPUT_tdict_name,iocvt_dict,this.output_dict);
        this.gtdict=this.register(this.OUTPUT_gtdict_name,iocvt_dict,this.output_dict);
    
        this.plabel=this.register(this.OUTPUT_plabel_name,iocvt_dict,this.output_dict);
        this.gplabel=this.register(this.OUTPUT_gplabel_name,iocvt_dict,this.output_dict);

        this.tensor_proto_img_name=this.register(this.OUTPUT_tensor_proto_img_name,iocvt_dict,this.output_dict);

        this.tensor_label_name=this.register(this.OUTPUT_tensor_label_name,iocvt_dict,this.output_dict);
        this.tensor_gt_length_name=this.register(this.OUTPUT_tensor_gt_length_name,iocvt_dict,this.output_dict);

        this.tensor_global_label_name = this.register(this.OUTPUT_tensor_global_label_name, iocvt_dict,this.output_dict,"NEP_skipped_NEP");

        this.sampler=this.register(this.MOD_sampler_name,modcvt_dict,this.mnames);
        this.protomvn=this.register(this.MOD_protomvn,modcvt_dict,this.mnames);

    def take_action(this, workspace:neko_workspace,environment:neko_environment):
        text_label=workspace.inter_dict[this.label];

        normprotos, plabel, gplabel, tdict, gtdict = \
            environment.module_dict[this.sampler].sample_charset_by_text(
                text_label, use_sp=False,device=environment.module_dict[this.protomvn].device());

        normprotos=environment.module_dict[this.mnames.protomvn](normprotos);

        plabel=plabel.to(normprotos.device);
        gplabel=gplabel.to(normprotos.device);

        gt_label,gt_length=encode_core(tdict,text_label,device=normprotos.device);
        gt_label_global,_=encode_core(gtdict,text_label,device=normprotos.device);


        workspace.inter_dict[this.plabel] = plabel;
        workspace.inter_dict[this.tdict] = tdict;
        workspace.inter_dict[this.gtdict]=gtdict;
        workspace.inter_dict[this.gplabel]=gplabel;
        workspace.inter_dict[this.tensor_proto_img_name]=normprotos;
        tenlen=torch.tensor(gt_length,device=normprotos.device);
        workspace.inter_dict[this.tensor_gt_length_name]=tenlen;
        workspace.inter_dict[this.tensor_label_name],_=neko_flatten_NG_lenpred.inflate(gt_label,tenlen);
        workspace.inter_dict[this.tensor_global_label_name],_=neko_flatten_NG_lenpred.inflate(gt_label_global,tenlen);

        if(this.tensor_global_label_name is not None):
            global_gt_label = encode_core(gtdict, text_label, device=normprotos.device);
            workspace.inter_dict[this.tensor_global_label_name]=global_gt_label;

        return workspace

