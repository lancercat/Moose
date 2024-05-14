from multiprocessing import Queue as mpQueue


# a workspace is where objection, intermediate data, memories, and logging_subs data go.
class neko_workspace:
    def __init__(this,input_dict=None,local_asset_dict=None,subspaces_dict=None,epoch_idx=0,batch_idx=0,device="cuda"):
        if(input_dict is None):
            input_dict={};
        if(subspaces_dict is None):
            subspaces_dict={};
        if(local_asset_dict is None):
            local_asset_dict={};


        # something you can drop after forward pass, gets with read and write.
        # Consider using data queue if there is no need for grad passing, so you can actually run each part asynchornizely.
        # Still, we DONOT do sanity check, you are on your own.
        this.inter_dict=input_dict;
        # some modules/assets dynamically generated for the exact iteration..,
        this.local_asset_dict=local_asset_dict;
        # subspaces, if modules and variables are needed.
        this.subspaces_dict=subspaces_dict;
        # the default device, only used when the module does not know what it is on.
        this.device=device;
        # Objectives, BP starts here.
        this.objdict={};
        # Logz
        this.logdict={}; # we don't want it to print
        # Device name. That means all the data in inter_dict and memory_dict are put on thiz device.
        this.device=device;
        # Your epoch index. Feel free to use this as a criteria to enable or disable functions.
        this.epoch_idx=epoch_idx;
        # Your batch index. Feel free to use this as a criteria to enable or disable functions.
        this.batch_idx=batch_idx;

from neko_sdk.neko_framework_NG.neko_module_setNG import neko_module_opt_setNG,get_modular_dict

# They will be changeable by agents, but just cannot be dropped after each iteration.
class neko_environment:
    def replace_queue(this,name,q=None):
        if(q is None):
            # USS Quincy
            q=mpQueue(maxsize=8);
        this.queue_dict[name]=q;
    # probably a died API.
    # You know I am just a cat and thus cannot remember everything I wrote...
    def view(this,mod_cvt_dict,queue_dict):
        vmodset=this.modset.view(mod_cvt_dict);

        pass;
    def save_mods(this):
        this.modset.save_necessary(this.epoch_idx,this.batch_idx);
    def __init__(this,assets_dict=None,queue_dict=None,modset:neko_module_opt_setNG=None):
        this.modset=modset;
        if(modset is not None):
            this.module_dict=get_modular_dict(modset);
        else:
            this.module_dict={};
        if(assets_dict is None):
            assets_dict={};
        if(queue_dict is None):
            queue_dict={};
        this.batch_idx=0;
        this.epoch_idx=0;
        # something other than modules
        this.assets_dict=assets_dict;
        # blocking queues, for async uses.
        # please drop grad before doing anything to it.
        this.queue_dict=queue_dict;
