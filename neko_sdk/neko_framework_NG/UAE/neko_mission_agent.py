import numpy as np
import torch
from easydict import EasyDict

from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.UAE.neko_abstract_agent import neko_abstract_sync_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from osocrNG.data_utils.neko_lmdb_holder import neko_lmdb_holder

from neko_sdk.neko_framework_NG.names import default_variable_names as dvn
from neko_sdk.neko_framework_NG.names import default_param_names as dpn
class neko_pretest_agent(neko_abstract_sync_agent):
    def take_action(this, _: neko_workspace, environment: neko_environment):
        environment.modset.eval_mode();


class neko_posttest_agent(neko_abstract_sync_agent):
    def take_action(this, _: neko_workspace, environment: neko_environment):
        environment.modset.train_mode();

from osocrNG.data_utils.data_agents.testing_dataset_agents import testing_cached_proto_loading_agent,testing_cached_proto_loading_agent_multiple
class neko_abstract_mission_agent(neko_abstract_sync_agent):
    MOD_proto_mvn_name="proto_mvn_name";
    MOD_prototyper_name="prototyper_name";
    OUTPUT_raw_id_name=dvn.raw_id_name;
    OUTPUT_raw_image_name=dvn.raw_image_name;
    OUTPUT_raw_label_name = dvn.raw_label_name;
    def set_proto_io(this,iocvt_dict,modcvt_dict):
        pass;
    def set_proto_loader(this,param):
        pass;
    def arm_cached_protos(this,workspace,protodict,test):
        return workspace;
    def cache_protos(this, environment):
        with torch.no_grad():
            protodict = this.proto_loader.cache_prototypes(environment);
        return protodict;
    def setup(this, param):
        this.input_dict = EasyDict();
        this.output_dict = EasyDict();
        this.internal_dict = EasyDict();

        this.omods = EasyDict();
        this.mnames = EasyDict();
        this.set_mod_io(param[dpn.iocvt_dict], param[dpn.modcvt_dict]);
        this.set_proto_loader(param);
        this.tests = param["tests"]
        this.test_data_holders = {};
        dataparam = neko_get_arg("data", this.tests);
        for t in dataparam:
            this.test_data_holders[t] = dataparam[t];
        ecfg = neko_get_arg("tester", param);
        this.tester = ecfg["agent"](ecfg["params"]);
        rcfg = neko_get_arg("reporters", param);
        this.reporters = {};
        for r in rcfg:
            this.reporters[r] = rcfg[r]["agent"](rcfg[r]["params"]);
        pass;


    def take_action(this, _: neko_workspace, environment: neko_environment):
        protodict=this.cache_protos(environment);
        for test in this.tests["tests"]:
            for r in this.reporters:
                this.reporters[r].reset(test,this.tests["meta"][this.tests["tests"][test]["meta"]]["has_unk"]);
            d = this.test_data_holders[this.tests["tests"][test]["data"]];
            for i in range(len(d)):
                ret = d.fetch_item({"id": i + 1});
                workspace = neko_workspace();
                if (ret is None):
                    continue;
                workspace.inter_dict[this.image_name] = [np.array(ret["image"])];
                if (this.text_label_name in workspace.inter_dict):
                    workspace.inter_dict[this.id_name] = [i + 1];
                if("label" in ret):
                    workspace.inter_dict[this.text_label_name] = [ret["label"]];
                workspace=this.arm_cached_protos(workspace,protodict,test);

                this.tester.take_action(workspace, environment);
                for r in this.reporters:
                    this.reporters[r].take_action(workspace, environment);
            for r in this.reporters:
                this.reporters[r].report(environment);

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.proto_mvn_name=this.register(this.MOD_proto_mvn_name,modcvt_dict,this.mnames);
        this.prototyper_name=this.register(this.MOD_prototyper_name,modcvt_dict,this.mnames);
        this.id_name=this.register(this.OUTPUT_raw_id_name,iocvt_dict,this.output_dict);
        this.image_name=this.register(this.OUTPUT_raw_image_name,iocvt_dict,this.output_dict);
        this.text_label_name=this.register(this.OUTPUT_raw_label_name,iocvt_dict,this.output_dict);
        this.set_proto_io(iocvt_dict,modcvt_dict);


class neko_test_mission_agent(neko_abstract_mission_agent):
    OUTPUT_tensor_proto_vec_name = dvn.tensor_proto_vec_name;
    OUTPUT_tensor_proto_img_name = dvn.tensor_proto_img_name;
    OUTPUT_proto_label_name = dvn.proto_label_name;
    OUTPUT_tdict_name = dvn.tdict_name;



    def set_proto_io(this, iocvt_dict, modcvt_dict):
        this.proto_name = this.register(this.OUTPUT_tensor_proto_vec_name, iocvt_dict, this.output_dict);
        this.proto_img_name=this.register(this.OUTPUT_tensor_proto_img_name,iocvt_dict,this.output_dict);
        this.proto_label_name = this.register(this.OUTPUT_proto_label_name, iocvt_dict, this.output_dict);
        this.tdict_name = this.register(this.OUTPUT_tdict_name, iocvt_dict, this.output_dict);

    def arm_cached_protos(this, workspace, protodict, test):
        workspace.inter_dict[this.proto_name] = protodict[this.tests["tests"][test]["meta"]][
            this.proto_name];
        workspace.inter_dict[this.proto_img_name]=protodict[this.tests["tests"][test]["meta"]][this.proto_img_name]
        workspace.inter_dict[this.proto_label_name] = protodict[this.tests["tests"][test]["meta"]][
            this.proto_label_name];
        workspace.inter_dict[this.tdict_name] = protodict[this.tests["tests"][test]["meta"]][
            this.tdict_name];
        return workspace;

    def set_proto_loader(this, param):
        this.proto_loader = testing_cached_proto_loading_agent(
            {
                "meta": param["tests"]["meta"],
                "iocvt_dict": {
                    testing_cached_proto_loading_agent.INPUT_meta_key_name: "NEP_skipped_NEP",
                    testing_cached_proto_loading_agent.OUTPUT_proto_label_name: this.proto_label_name,
                    testing_cached_proto_loading_agent.OUTPUT_tensor_proto_vec_name: this.proto_name,
                    testing_cached_proto_loading_agent.OUTPUT_tensor_proto_img_name: this.proto_img_name,
                    testing_cached_proto_loading_agent.OUTPUT_tdict_name:this.tdict_name,
                },
                "modcvt_dict": {
                    "proto_mvn_name": this.proto_mvn_name,
                    "prototyper_name": this.prototyper_name
                },
            }
        )

class neko_test_mission_agent_vbd(neko_abstract_mission_agent):
    OUTPUT_tensor_proto_vec_name = dvn.tensor_proto_vec_name;
    OUTPUT_proto_label_name = dvn.proto_label_name;
    OUTPUT_tdict_name = dvn.tdict_name;
    OUTPUT_rotated_proto_tensor_name = dvn.rotated_tensor_proto_vec_name;



    def set_proto_io(this, iocvt_dict, modcvt_dict):
        this.rotated_proto_name = this.register(this.OUTPUT_rotated_proto_tensor_name, iocvt_dict, this.output_dict);
        this.proto_name=this.register(this.OUTPUT_tensor_proto_vec_name,iocvt_dict,this.output_dict);
        this.proto_label_name = this.register(this.OUTPUT_proto_label_name, iocvt_dict, this.output_dict);
        this.tdict_name = this.register(this.OUTPUT_tdict_name, iocvt_dict, this.output_dict);

    def arm_cached_protos(this, workspace, protodict, test):
        workspace.inter_dict[this.proto_name] = protodict[this.tests["tests"][test]["meta"]][
            this.proto_name];
        workspace.inter_dict[this.rotated_proto_name] = protodict[this.tests["tests"][test]["meta"]][
            this.rotated_proto_name];
        workspace.inter_dict[this.proto_label_name] = protodict[this.tests["tests"][test]["meta"]][
            this.proto_label_name];
        workspace.inter_dict[this.tdict_name] = protodict[this.tests["tests"][test]["meta"]][
            this.tdict_name];

        return workspace;

    def set_proto_loader(this, param):
        this.proto_loader = testing_cached_proto_loading_agent(
            {
                testing_cached_proto_loading_agent.PARAM_meta: param["tests"]["meta"],
                testing_cached_proto_loading_agent.PARAM_possible_rotation:[0,1,2,3],
                "iocvt_dict": {
                    testing_cached_proto_loading_agent.INPUT_meta_key_name: "NEP_skipped_NEP",
                    testing_cached_proto_loading_agent.OUTPUT_tensor_proto_vec_name: this.proto_name,
                    testing_cached_proto_loading_agent.OUTPUT_rotated_tensor_proto_vec_name:this.rotated_proto_name,
                    testing_cached_proto_loading_agent.OUTPUT_proto_label_name: this.proto_label_name,
                    testing_cached_proto_loading_agent.OUTPUT_tdict_name:this.tdict_name,
                },
                "modcvt_dict": {
                    "proto_mvn_name": this.proto_mvn_name,
                    "prototyper_name": this.prototyper_name
                },
            }
        )

#
#
# class neko_test_mission_agent_multi_sel_proto(neko_abstract_sync_agent):
#     def set_mod_io(this, iocvt_dict, modcvt_dict):
#         this.mnames.proto_mvn_name=modcvt_dict["proto_mvn_name"];
#         this.mnames.prototyper_name=modcvt_dict["prototyper_name"];
#         this.output_dict.image_name=iocvt_dict[dvn.raw_image_name];
#         this.output_dict.text_label_name=iocvt_dict[dvn.raw_label_name];
#         this.output_dict.proto_name=iocvt_dict[dvn.tensor_proto_vec_name];
#         this.output_dict.proto_label_name=iocvt_dict[dvn.proto_label_name];
#         this.output_dict.tdict_name=iocvt_dict[dvn.tdict_name];
#
#
#     def setup(this, param):
#         this.input_dict = EasyDict();
#         this.output_dict = EasyDict();
#         this.internal_dict=EasyDict();
#
#         this.omods = EasyDict();
#         this.mnames = EasyDict();
#         this.set_mod_io(param["iocvt_dict"], param["modcvt_dict"]);
#         test_data_holders:dict[neko_lmdb_holder];
#         this.proto_loader=testing_cached_proto_loading_agent_multiple(
#             {
#                 "meta":param["tests"]["meta"],
#                 "iocvt_dict":{
#                     "meta_key_name": "NEP_skipped_NEP",
#                     "tensor_proto_vec_name": this.output_dict.proto_name,
#                     "plabel_name": this.output_dict.proto_label_name,
#                     "tdict_name": this.output_dict.tdict_name,
#                 },
#                 "modcvt_dict":{
#                     "proto_mvn_name": this.mnames.proto_mvn_name,
#                     "prototyper_name":this.mnames.prototyper_names
#                 },
#             }
#         );
#         this.tests=param["tests"]
#         this.test_data_holders={};
#         dataparam=neko_get_arg("data", this.tests);
#         for t in dataparam:
#             this.test_data_holders[t]=dataparam[t];
#         ecfg=neko_get_arg("tester",param);
#         this.tester=ecfg["agent"](ecfg["params"]);
#         rcfg=neko_get_arg("reporters",param);
#         this.reporters={};
#         for r in rcfg:
#             this.reporters[r]=rcfg[r]["agent"](rcfg[r]["params"]);
#         pass;
#
#     def take_action(this, _: neko_workspace, environment: neko_environment):
#         protodict=this.proto_loader.cache_prototypes(environment);
#         for test in this.tests["tests"]:
#             for r in this.reporters:
#                 this.reporters[r].reset(test);
#             d=this.test_data_holders[ this.tests["tests"][test]["data"]];
#             for i in range(len(d)):
#                 ret=d.fetch_item({"id":i+1});
#                 workspace=neko_workspace();
#                 workspace.inter_dict[this.output_dict.image_name] = [np.array(ret["image"])];
#                 workspace.inter_dict[this.output_dict.text_label_name] = [ret["label"]];
#                 workspace.inter_dict[this.output_dict.proto_name]=protodict[ this.tests["tests"][test]["meta"]]["prototypes"];
#                 workspace.inter_dict[this.output_dict.proto_label_name]=protodict[ this.tests["tests"][test]["meta"]]["plabels"];
#                 workspace.inter_dict[this.output_dict.tdict_name] = protodict[this.tests["tests"][test]["meta"]]["tdict"];
#                 this.tester.take_action(workspace,environment);
#                 for r in this.reporters:
#                     this.reporters[r].take_action(workspace,environment);
#             for r in this.reporters:
#                 this.reporters[r].report(environment);
#
