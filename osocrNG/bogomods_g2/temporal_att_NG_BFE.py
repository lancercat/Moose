from neko_sdk.neko_framework_NG.bogog2_modules.neko_abstract_bogo_g2 import neko_abstract_bogo_g2
class neko_BFE_att_NG(neko_abstract_bogo_g2):
    def registermods(this,moddict,mod_cvt_dict):
        this.fe = moddict[mod_cvt_dict["fe"]];
        this.att_engine= moddict[mod_cvt_dict["att_engine"]];
        pass;
    def check_cvt(this,mod_cvt_dict):
        mod_cvt_dict["fe"];
        mod_cvt_dict["att_engine"];


    def forward(this, tensorimg,situation,  feature_, img_size, mask=None):
        att_feats = this.fe(tensorimg);
        return this.att_engine( tensorimg,situation, att_feats, img_size);
def config_BFE_att(param,bogocfgdict):

    bogocfgdict[param["name"]]={
        "bogo_mod": neko_BFE_att_NG,
        "args":
            {
                "mod_cvt":
                    {
                        "fe": param["fe_name"],
                        "att_engine": param["att_engine_name"],
                    },
            }
    }
    return bogocfgdict;

