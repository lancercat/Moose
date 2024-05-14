from osocrNG.modular_agents_ocrNG.output_logging_subs.wandb_acr_fps_reporter import case_inv_acr_fps_reporter_wandb,case_inv_multi_orientation_acr_fps_reporter_wandb

def get_wandb_reporters(param):
    agent=case_inv_acr_fps_reporter_wandb
    return{
        "case_inv_acr_time":
            {
                "agent":agent,
                "params":{
                    agent.PARAM_wandb_run:param["wandb_run"],
                    "iocvt_dict":param["iocvt_dict"],
                    "modcvt_dict": param["modcvt_dict"],

                }
            }
        };
def get_wandb_mo_reporters(param):
    agent=case_inv_multi_orientation_acr_fps_reporter_wandb;
    return{
        "case_inv_acr_time":
            {
                "agent":agent,
                "params":{
                    agent.PARAM_wandb_run: param["wandb_run"],
                    "iocvt_dict":param["iocvt_dict"],
                    "modcvt_dict": param["modcvt_dict"],

                }
            }
        };