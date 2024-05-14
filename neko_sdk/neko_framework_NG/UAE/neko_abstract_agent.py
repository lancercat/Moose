import multiprocessing

from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from neko_sdk.cfgtool.argsparse import neko_get_arg

class neko_abstract_agent:
    def register(this,local_name,param,cvtdict,default=None):
        global_name=neko_get_arg(local_name,param,default);
        cvtdict[local_name]=global_name;
        return global_name;

    def setup(this,param):
        pass;


    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        print("Not implemented", workspace.epoch_idx,",",workspace.batch_idx);
        workspace.batch_idx=workspace.batch_idx+1;
        return workspace,environment;

    def __init__(this,param):
        this.setup(param);


class neko_abstract_sync_agent(neko_abstract_agent):
    pass;
# Note that there are no rules preventing you to create agents inside agents.
class neko_abstract_async_agent(neko_abstract_agent):
    # sets
    def grab_nested(this, term, object_dict):
        if (type(term) is list):
            return [this.grab_nested(n, object_dict) for n in term];
        else:
            return object_dict[term];
    def remap(this,remapping_dict,source_dict):
        result={};
        for k in remapping_dict:
            result[k]=this.grab_nested(remapping_dict[k],source_dict);
        return result;

    def make_private_workspace(this, workspace, environment, cvt_dict):
        pworkspace = neko_workspace();
        if(workspace is not None):
            pworkspace.batch_idx=workspace.batch_idx;
            pworkspace.epoch_idx = workspace.epoch_idx;
        for ik in cvt_dict:
            pworkspace.inter_dict[ik] = environment.queue_dict[ik].get();
        return pworkspace;
    def make_private_enviroment(this,environment,remapdict):
        penvironment = neko_environment();
        penvironment.queue_dict = this.remap(
            remapdict["queues"], environment.queue_dict);
        penvironment.assets_dict = this.remap(
            remapdict["assets"], environment.assets_dict);
        return penvironment
    def feedback(this,privateworkspace,private_workspace,workspace,environment):

        pass;

    def take_action_private(this, workspace, environment):
        pworkspace=this.make_private_workspace(workspace,environment,this.inputs)
        pworkspace,penv = this.take_action(pworkspace, environment);
        this.feedback(pworkspace,penv,workspace, environment);
        return workspace,environment;

        # if you do need sharing memory between
        # Pipes, memory, scheduling, latching and locking....
        # It sounds like agents are process----yes you are CORRECT!
        # The dead knowledge comes back at us! ZOMBIE attack!
        # del workspace;
        pass;

    def setup(this,param):
        this.input_dict = {};
        this.output_dict = {};
        this.epoch_cnt=9;
        this.iter_cnt=39;
        pass;


    def action_loop(this):
        # workspace CAN be semi-permenent, however without promise.
        # please assume that they are NOT.
        # Eventually, the user need to do the bookkeeping.
        print("loop undefined");
        workspace=neko_workspace();
        for i in range(this.epoch_cnt):
            for j in range(this.iter_cnt):
                this.take_action_private(workspace,this.environment);
                workspace.batch_idx+=1;
            workspace.epoch_idx+=1;
            workspace.batch_idx=0;

    def mount_environment(this,param,environment):
        this.inputs=param["inputs"];
        this.outputs=param["outputs"];
        if("remapping" not in param):
            this.environment = environment;
        else:
            this.environment=this.make_private_enviroment(environment,param["remapping"])
    def __init__(this,param):
        super().__init__(param);
        this.worker=None;
        this.stop();
        pass;

    def start(this,mapping_param,environment,mode="fork"):
        this.stop();
        this.mount_environment(mapping_param,environment);
        this.status="running";
        if(mode=="fork"):
            this.worker=multiprocessing.Process(target=this.action_loop);
        else:
            this.worker = multiprocessing.get_context(mode).Process(target=this.action_loop);

        this.worker.start();
        pass;
    def start_sync(this,mapping_param,environment,mode="fork"):
        this.stop();
        this.mount_environment(mapping_param, environment);
        this.status = "running";
        this.action_loop();

    def wait(this):
        this.worker.join();
    def stop(this):
        this.input_dict = {};
        this.output_dict = {};
        this.module_dict={};
        if(this.worker is not None):
            this.worker.kill();
        # Well the worker can have children, so there is no need to make it a list here.
        this.worker=None;
        this.status="stopped";
        this.environment=neko_environment();
        this.worker = None;
        this.workspace = None;
    def stop_and_quit(this):
        this.stop();
        exit(0);

if __name__ == '__main__':
    a=neko_abstract_async_agent(
        {}
    )
    a.start({"inputs":[],
         "outputs":[]},neko_environment());
    a.wait();
