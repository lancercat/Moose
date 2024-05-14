import copy

from neko_sdk.MJT.common import Updata_Parameters
from neko_sdk.MJT.utils import update
from neko_sdk.neko_framework_NG.neko_modular_NG import get_default_NG_modular, neko_modular_NG
from neko_sdk.thirdparty.mmdetapply import multi_apply


def attempt_arm_bogo_list( bogolist, modcfgs, modular_dict):
    fail_list = [];
    for name in bogolist:
        cfg = modcfgs[name];
        # bogo modules are re-combination of parts of existing real and bogo modules.
        mod = cfg["bogo_mod"](cfg["args"], modular_dict);
        modular_dict[name] = mod;
        try:
            mod = cfg["bogo_mod"](cfg["args"], modular_dict);
            modular_dict[name] = mod;
        except:
            fail_list.append(name);
    return modular_dict, fail_list;

def get_modular_dict(modset,key="NEP_main_NEP"):
    modular_dict = {};
    rmd=modset.get_mods(key);
    for k in rmd:
        modular_dict[k] = rmd[k];

    list_bogo_to_arm = copy.copy(modset.bogo_modular_list);
    for i in range(40):
        if (len(list_bogo_to_arm) == 0):
            break;
        if (i):
            print("Attempt", i, "for", list_bogo_to_arm);
        modular_dict, list_bogo_to_arm = attempt_arm_bogo_list(list_bogo_to_arm, modset.bogo_config_dict,
                                                                    modular_dict);
    if (len(list_bogo_to_arm)):
        print("failed dependency for module(s):", list_bogo_to_arm, "please check dependency");
        exit(9);
    return modular_dict;





class neko_module_opt_setNG:

    # well if you insist on data paralleling, fork and put them into moddict
    def get_mods(this,key):
        if(key is "NEP_main_NEP"):
            return this.main_mods;
        else:
            return this.device_mod_dict[key];
    def clear(this):
        this.optimizers = [];
        this.optnames = [];
        this.optimizer_schedulers = [];

        this.real_modulars = {};
        this.main_mods={};
        this.device_mod_dict={};

        this.bogo_modular_list = [];
        this.bogo_config_dict = {};
        this.bogo_mappings = {};

    def register_bogo(this,name,bogocfg):
        this.bogo_modular_list.append(name);
        this.bogo_config_dict[name]=bogocfg;
    def register_real(this,name,mod):
        this.real_modulars[name] = mod;
        this.main_mods[name] = mod.model;
        if (this.real_modulars[name].optimizer is not None):
            this.optnames.append(name);
            this.optimizers.append(this.real_modulars[name].optimizer);
            this.optimizer_schedulers.append(this.real_modulars[name].optimizer_sched);

    def arm_modules(this, real_modular_cfg:dict[neko_modular_NG], bogo_mod_cfg,decay_override=None):
        this.clear();
        # we keep them to make replications;
        for name in bogo_mod_cfg:
            this.register_bogo(name,bogo_mod_cfg[name])
        for name in real_modular_cfg:
            if(decay_override):
                real_modular_cfg[name]["weight_decay"]=decay_override;
            this.register_real(name,get_default_NG_modular(real_modular_cfg[name]));

    def eval_mode(this):
        for modk in this.real_modulars:
            this.real_modulars[modk].eval();
    def to(this,device):
        for k in this.real_modulars:
            this.real_modulars[k].to(device)

    def train_mode(this):
        for modk in this.real_modulars:
            this.real_modulars[modk].train();
    def save_necessary(this,nEpoch, batch_idx):
        for modk in this.real_modulars:
            this.real_modulars[modk].save_if_needed(modk,nEpoch, batch_idx);
    def load(this,itrkey):
        for modk in this.real_modulars:
            this.real_modulars[modk].load(modk,itrkey);
    def update_para(this):
        multi_apply(update, this.optimizers);

    def update(this):
        try:
            Updata_Parameters(this.optimizers, frozen=[]);
        except:
            print("Oops");
            exit(9);
    def update_opt(this,epoch_idx):
        for opts in this.optimizer_schedulers:
            opts.step(epoch_idx);
    def zero_grad(this):
        for opt in this.optimizers:
            opt.zero_grad();
    def norm_grad(this):
        for modk in this.real_modulars:
            if (this.real_modulars[modk].save_each > 0):
                this.real_modulars[modk].normgrad();
