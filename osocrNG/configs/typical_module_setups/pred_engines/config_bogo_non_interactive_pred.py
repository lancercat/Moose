from osocrNG.deprecated.pred_engines.bogo_non_interactive_pred import bogo_non_interactive_pred

def config_bogo_non_interactive_pred(seq,preds,losses,weights,keys):
    return {
        "bogo_mod": bogo_non_interactive_pred,
        "args":
            {
                "seq": seq,
                "preds": preds,
                "losses": losses,
                "weights": weights,
                "keys": keys,

            }
    }