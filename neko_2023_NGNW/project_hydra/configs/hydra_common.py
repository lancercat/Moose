from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_agent_wrapping_agent
from osocrNG.data_utils.data_agents.testing_dataset_agents import neko_single_image_dispatcher_padder


def get_tester_hydra(params):
    return {
        "agent":neko_agent_wrapping_agent,
        "params":{
            "agent_list":["dispatcher","core"],
            "dispatcher":{
                "agent":neko_single_image_dispatcher_padder,
                "params":{
                    "batch_size":1,
                    "anchors":params["anchors"],
                    "becaon_size":64,
                    "iocvt_dict":params["iocvt_dict"],
                    "modcvt_dict": params["modcvt_dict"],
                }
            },
            "core":params["core"],
        }

    }
