import torch
from torch.nn import functional as trnf

from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.workspace import neko_environment, neko_workspace


class neko_vis_prototyper_agent_mk2(neko_module_wrapping_agent):
    def set_mod_io(this,iocvt_dict,modcvt_dict):
        this.mnames.fe=neko_get_arg("fe_name",modcvt_dict);
        this.mnames.att_name=neko_get_arg("att_name",modcvt_dict);
        this.mnames.aggr_name=neko_get_arg("aggr_name",modcvt_dict);
        this.input_dict.protoimage=neko_get_arg("tensor_proto_img_name",iocvt_dict);
        this.output_dict.protovector=neko_get_arg("tensor_proto_vec_name",iocvt_dict);

    def take_action(this, workspace:neko_workspace,environment:neko_environment):
        feats=environment.module_dict[this.mnames.fe](
            workspace.inter_dict[this.input_dict.protoimage]
        );
        attm=environment.module_dict[this.mnames.att_name](
            workspace.inter_dict[this.input_dict.protoimage],
            feats
        )
        protos=environment.module_dict[this.mnames.aggr_name](feats,attm);
        protos=trnf.normalize(protos,p=2,dim=-1);
        workspace.inter_dict[this.output_dict.protovector]=protos;
        return workspace

class neko_testing_prototyper_agent_mk2(neko_module_wrapping_agent):
    def set_mod_io(this,iocvt_dict,modcvt_dict):
        this.mnames.protomvn=neko_get_arg("protomvn",modcvt_dict)
        this.mnames.prototyper=neko_get_arg("prototyper_name",modcvt_dict);
        this.mnames.meta_holder = neko_get_arg("meta_holder_name", modcvt_dict);
        this.output_dict.protovector = neko_get_arg("tensor_proto_vec_name", iocvt_dict);
        this.output_dict.tdict=neko_get_arg("tdict_name",iocvt_dict);
        this.output_dict.plabel=neko_get_arg("plabel_name",iocvt_dict);



    def set_etc(this,param):
        this.capacity = neko_get_arg("capacity",param,512);
    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        normprotos, plabels, gplabels, bidict, gbidict=environment.module_dict[this.mnames.meta_holder].dump_all();
        allprotovec=[];
        for i in range(0,len(normprotos),this.capacity):
            tprotoimg=environment.module_dict[this.mnames.protomvn](normprotos[i:i+normprotos]);

            feats = environment.module_dict[this.mnames.fe](
                tprotoimg,
            );
            attm = environment.module_dict[this.mnames.att_name](
                workspace.inter_dict[this.input_dict.protoimage],
                feats
            )
            protos = environment.module_dict[this.mnames.aggr_name](feats, attm);
            protos = trnf.normalize(protos, p=2, dim=-3);
            allprotovec.append(protos);
        workspace.inter_dict[this.output_dict.tensor_proto_vec_name]=torch.cat(allprotovec);
