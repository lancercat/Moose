import json
import os

from neko_sdk.environment.root import find_data_root


class platform_cfg:
    def set_up(this, data_root, save_root, wandbdir):
        this.data_root = data_root;
        # this.devices=devices;
        this.wandbdir=wandbdir
        this.save_root = save_root;
        this.log_root=save_root;
        os.makedirs(this.save_root, exist_ok=True);

    def __init__(this, cfg):
        # if(cfg is None):
        #     this.set_up(find_data_root(),"~/hydra_saves/",["cuda:0"]);
        # elif(type(cfg)==str):
        #     with open(cfg, "r") as fp:
        #         c = json.load(fp);
        #         this.set_up(c["data_root"], c["save_root"], c["devices"]);
        # else:
        #     this.set_up(cfg["data_root"],cfg["save_root"],cfg["devices"]);
        if (cfg is None):
            this.set_up(find_data_root(), "~/hydra_saves/", "toUPDATE")
        elif (type(cfg) == str):
            with open(cfg, "r") as fp:
                c = json.load(fp)
                this.set_up(c["data_root"], c["save_root"],  c["wandbdir"])
        else:
            this.set_up(cfg["data_root"], cfg["save_root"], cfg["wandbdir"])


class neko_platform_cfg:
    def set_up(this, data_root, save_root,log_root,devices):
        this.data_root = data_root;
        this.log_root=log_root;
        this.devices=devices;
        this.save_root = save_root;
        os.makedirs(this.save_root, exist_ok=True);

    # if you have no wandb logger, don't call this function!
    def arm_wandb(this,project="moose",method_name=None,apikey=None):
        import wandb;
        if(method_name is None):
            method_name=os.path.basename(os.getcwd());
        this.project=project;
        this.entity=method_name;
        this.dir=os.path.join(this.log_root,this.project,this.entity);
        os.makedirs(this.dir,exist_ok=True);
        if(apikey is None):
            this.run=wandb.init(project=this.project, entity=this.entity, dir=this.dir,mode="offline");
        else:
            # initialize your key with some magic
            this.run=wandb.init(project=this.project, entity=this.entity, dir=this.dir);

    def __init__(this, cfg):
        if(cfg is None):
            this.set_up(find_data_root(),"/run/media/lasercat/data/moose_models/","/run/media/lasercat/f3a1698e-80ad-4473-8fc6-4df8c81c3831/dump/logs",["cuda:0"]);
        elif(type(cfg)==str):
            with open(cfg, "r") as fp:
                c = json.load(fp);
                this.set_up(c["data_root"], c["save_root"],c["log_root"], c["devices"]);
        else:
            this.set_up(cfg["data_root"],cfg["save_root"],cfg["log_root"],cfg["devices"]);
