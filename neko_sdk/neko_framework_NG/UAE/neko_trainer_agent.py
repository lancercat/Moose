from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.UAE.neko_abstract_agent import neko_abstract_async_agent
from neko_sdk.neko_framework_NG.agents.neko_optim_agent import get_neko_optim_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment


# A trainer agent include all training routines and some testing routiones.
# Also holds stuffs to carry ALL these routines
#   neko_modulars(weights and optimizers).
#   bogo modulars (simply something callable).
#   other agents
#   raw dataloaders
#   data augmentation and batching facilities
#   data/module distribution system
# routines and testers need to do their own logging.
class neko_trainer_agent(neko_abstract_async_agent):
    enviroment: neko_environment

    def build_agent(this, adict):
        rad = {}
        for k in adict:
            rad[k] = adict[k]["agent"](adict[k]["params"]);
        return rad;

    def setup(this, param):
        this.devices = neko_get_arg("devices", param, ["cuda:0"]);
        this.routine_dict = this.build_agent(param["routine_dict"]);
        this.routine_names = param["routine_names"];

        this.optim_dict = this.build_agent(neko_get_arg("optim_dict", param, {"main_optim": get_neko_optim_agent()}));
        this.optim_names = neko_get_arg("optim_names", param, ["main_optim"])
        this.pre_test_dict = this.build_agent(param["pretest_dict"]);
        this.pre_test_names = param["pretest_names"];

        this.post_test_names = this.build_agent(param["posttest_names"]);
        this.post_test_dict = param["posttest_dict"];

        this.tester_names = param["tester_names"];
        this.tester_dict = this.build_agent(param["tester_dict"]);

        this.iter_logger_names = param["iter_logger_names"];
        this.iter_logger_dict = this.build_agent(param["iter_logger_dict"]);

        this.epoch_logger_names = param["epoch_logger_names"];
        this.epoch_logger_dict = this.build_agent(param["epoch_logger_dict"]);

        this.check_each = neko_get_arg("check_each", param, 20000);

        this.set_private_env = neko_get_arg("set_private_env", param, False);
        this.epoch_cnt = neko_get_arg("epoch_cnt", param);
        this.iter_cnt = neko_get_arg("iter_cnt", param);

    def mount_environment(this, param, environment: neko_environment):
        this.environment = environment;
        for k in this.routine_dict:
            if (k in param):
                this.routine_dict[k]["environment"] = \
                    this.make_private_enviroment(environment, param[k]["remap"]);

    ## Switchs into training or testing mode.
    ## There will be cache makings.

    def take_action(this, _, environment: neko_environment):
        # ws=neko_workspace(device=this.devices[0]);
        ws = neko_workspace(device=this.devices);
        ws.batch_idx = environment.batch_idx;
        ws.epoch_idx = environment.epoch_idx;
        for k in this.routine_names:
            this.routine_dict[k].take_action(ws, environment);
            # print(k,"done fpbp")
        for k in this.optim_names:
            this.optim_dict[k].take_action(ws, environment);
        for k in this.iter_logger_names:
            this.iter_logger_dict[k].take_action(ws, environment);
        environment.batch_idx += 1;
        # print(ws.logdict);
        pass;

    def check_and_save(this):
        this.environment.modset.eval_mode();
        try:
            this.environment.save_mods();
        except:
            this.environment.save_mods();
            print("saving failed, full disk?")
        for k in this.pre_test_dict:
            this.pre_test_dict[k].take_action(None, this.environment);
        for k in this.tester_dict:
            this.tester_dict[k].take_action(None, this.environment);
        for k in this.post_test_dict:
            this.post_test_dict[k].take_action(None, this.environment);

        this.environment.modset.train_mode();

    def action_loop(this):
        this.check_and_save();

        for i in range(this.epoch_cnt):
            for j in range(this.iter_cnt):
                if (j and j % this.check_each == 0):
                    for k in this.epoch_logger_names:
                        this.epoch_logger_dict[k].take_action({}, this.environment);
                    this.check_and_save();
                this.take_action(None, this.environment);
            this.environment.epoch_idx += 1;
            this.environment.batch_idx = 0;
            this.environment.modset.update_opt(this.epoch_cnt);
            this.check_and_save();
            print("epoch done");
