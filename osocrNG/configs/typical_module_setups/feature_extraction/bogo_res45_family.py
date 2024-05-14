from osocrNG.bogomods_g2.res45g2.res45_g2 import neko_res45_bogo_g2


def config_bogo_resbinorm_g2_core(conv_container,bn_container,engine):
    return {
        "bogo_mod": engine,
        "args":
        {
            "mod_cvt":
            {
                "conv":conv_container,
                "norm":bn_container,
            },
        }
    }
def config_bogo_resbinorm_g2(conv_container,bn_container):
    return config_bogo_resbinorm_g2_core(conv_container,bn_container,engine=neko_res45_bogo_g2);
