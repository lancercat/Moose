
import os

import torch
from torch import nn
from torch.nn import parallel as trnp


# Also, NG modular is no more callable. It is now a mere container & controller
# use a bogo to warp it up and bind it's mapping it to a device, which you may find one solution in
# Data parallel is still too much if we want fancy controls, let's hope we will support that in 3G.
# We are not going to implement p2p in NG. maybe 3G will support centerless FL.

# you can also set a server if you want to do FL
# the server is a dict, include address, server-side name and sync frequency.

class neko_modular_NG:
    def __init__(this,save_path,save_name,
                 module:nn.Module,
                 optimizer:torch.optim.Optimizer,
                 optimizer_sched:torch.optim.lr_scheduler.LRScheduler,
                 save_each=20000,
                 server=None,
                 ):
        super(neko_modular_NG, this).__init__()
        this.save_path=save_path;
        this.model=module;
        this.optimizer=optimizer;
        this.optimizer_sched=optimizer_sched;
        this.save_name=save_name;
        this.save_each=save_each;
        this.server=server;

    def detach(this):
        this.model.requires_grad_(False)
    def attach(this):
        this.model.requires_grad_(True)

    def train(this,training=True):
        this.model.train(training);
    def eval(this):
        this.model.eval();
    def normgrad(this):
        if this.save_each>0:
            nn.utils.clip_grad_norm_(this.model.parameters(), 20, 2)

    def load(this,name,itrkey):
        if(name==None):
            name=this.save_name;
        p = os.path.join(this.save_path,name + itrkey + ".pth");
        op= os.path.join(this.save_path,name+itrkey+"_opt.pth");
        osp=os.path.join(this.save_path,name+itrkey+"_opt_sched.pth");
        try:
            this.model.load_state_dict(torch.load(p))
        except:
            try:
                this.model.load_state_dict(torch.load(p).state_dict());
                print(this.save_name, "loaded as a hack");
            except:
                print(this.save_name, "cannot load MODULE", "itr",p,", starting fresh");
        try:
            this.optimizer.load_state_dict(torch.load(op));
        except:
            try:
                this.optimizer.load_state_dict(torch.load(op).state_dict());
            except:
                print(this.save_name, "cannot load OPTIMIZERS", "itr", op, ", starting fresh");
        try:
            this.optimizer_sched.load_state_dict(torch.load(osp));
        except:
            try:
                this.optimizer_sched.load_state_dict(torch.load(osp).state_dict());
            except:
                print(this.save_name, "cannot load OPTIMIZER_SCHED", "itr", osp, ", starting fresh");
        print("let me remind you that queue contents can not be restored for now");

    def to(this,dev="cuda"):
        this.model.to(dev);
    def next_epoch(this):
        this.optimizer_sched.step();

    def save_stub(this,key:str):
        torch.save(this.model.state_dict(), this.save_path + key+".pth");
        if(this.optimizer is not None):
            torch.save(this.optimizer.state_dict(), this.save_path + key + "_opt.pth");
        if(this.optimizer_sched is not None):
            torch.save(this.optimizer_sched.state_dict(), this.save_path + key + "_opt_sched.pth");

    def save(this,nEpoch):
        if(this.save_each>0 ):
            this.save_stub('_E{}'.format(nEpoch));
            this.save_stub('latest');
    def save_if_needed(this,name,nEpoch,batch_idx):
        if(this.save_each>0 and batch_idx and batch_idx%this.save_each==0):
            print("Saving", this.save_path + name+'_E{}_I{}.pth'.format(nEpoch, batch_idx))
            this.save_stub(name+'_E{}_I{}'.format(nEpoch, batch_idx));
        else:
            print("Saving", this.save_path + name + '_E{}.pth'.format(nEpoch));
            this.save_stub(name + '_E{}'.format(nEpoch));
    # don't count that on NG

    def step_and_save(this,nEpoch,batch_idx):
        this.optimizer.step();
    def replicate(this, devices):
        return trnp.replicate(this.model, devices);
from neko_sdk.cfgtool.argsparse import neko_get_arg
# The
def get_default_optim(params,lr=1.0,weight_decay=0.0005,sched_override=None):

    optimizer = torch.optim.Adadelta(params, lr=lr,weight_decay=weight_decay)
    if(sched_override is None):
        optimizer_sched = torch.optim.lr_scheduler.MultiStepLR(optimizer, [3, 5], 0.1)
    else:
        optimizer_sched=sched_override["engine"](optimizer,**sched_override["params"]);
    return optimizer, optimizer_sched;


def get_default_NG_modular(params):
    mod=params["mod_type"](params["mod_param"]);
    opt_cfg=neko_get_arg("opt",params,"NEP_skipped_NEP");
    save_each=neko_get_arg("save_each",params,20000);
    itrk=neko_get_arg("itrkey",params,"TopNep");
    weight_decay=neko_get_arg("weight_decay",params,0.0005);
    if(save_each<=0):
        opt,opt_sched=None,None;
    elif(opt_cfg is None):
        if (len(list(mod.parameters())) == 0):
            opt,opt_sched= None, None
        else:
            opt,opt_sched=get_default_optim(mod.parameters(),weight_decay=weight_decay);
    else:
        exit(9)
    modular=neko_modular_NG(
        save_path=params["save_path"],
        save_name=params["save_name"],
        module=mod,
        optimizer=opt,
        optimizer_sched=opt_sched,
        save_each=save_each);
    modular.load(None,itrk);
    return modular;


if __name__ == '__main__':
    mod=torch.nn.Conv2d(1,3,4).cuda();
    models = trnp.replicate(mod, ["cuda","cuda"]);
    pass;
