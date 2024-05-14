# NG hoards a bunch of agents(mod). within a bunch of agents.
from easydict import EasyDict

from neko_sdk.neko_framework_NG.UAE.neko_abstract_agent import neko_abstract_sync_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment


#Agents are not usually meant for data processing, data processing is controlled in the bogomods
#agents are here to decide which module to call and to produce what in the dictionary. Thus, it is usally nothing but a wrapper.

# in simple language, do not process tensors directly here. use modules and bogos to manipulate them.

class neko_module_wrapping_agent(neko_abstract_sync_agent):
    def set_mod_io(this,iocvt_dict,modcvt_dict):
        pass;
    def set_etc(this,param):
        pass;
    def setup(this,param):
        this.input_dict = EasyDict();
        this.output_dict = EasyDict();
        this.omods = EasyDict();
        this.mnames = EasyDict();
        this.set_mod_io(param["iocvt_dict"],param["modcvt_dict"]);
        this.set_etc(param);

class neko_agent_wrapping_agent(neko_abstract_sync_agent):
    def setup(this,param):
        this.agent_n = [];
        this.agent_d={};
        for name in param["agent_list"]:
            this.agent_n.append(name);
            this.agent_d[name]=param[name]["agent"](param[name]["params"]);

    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        for n in this.agent_n:
            this.agent_d[n].take_action(workspace,environment);
        pass;

class  neko_keyword_selective_execution_agent(neko_abstract_sync_agent):
    def setup(this,param):
        this.agent_n = [];
        this.agent_d={};
        for name in param["agent_list"]:
            this.agent_n.append(name);
            this.agent_d[name]=param[name]["agent"](param[name]["params"]);
        this.selector_name=param["selector_name"];
    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        n=workspace.inter_dict[this.selector_name];
        this.agent_d[n].take_action(workspace,environment);