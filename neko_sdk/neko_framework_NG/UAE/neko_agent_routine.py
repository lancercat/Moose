# Do one iter of training or testing, with given list of agents
# it does not take care of implementation, no controls either.

#
class neko_agent_routine:
    def san_chk(this):
        # errors=[]
        have = set(this.inp_cvt_dict.keys());
        for r in this.subroutines:
            for i_n in r.input_dict:
                if (i_n not in have):
                    print("input", i_n, "of", r, "is missing");
            for o_n in r.output_dict:
                have.add(o_n);
            for m_n in r.mnames:
                if(m_n not in this.mod_cvt_dict):
                    print("input", m_n, "of", r, "is missing")
        pass;

    def __init__(this,args):
        mod_cvt_dicts, inp_cvt_dicts=\
        args["mod_cvt_dicts"],args["inp_cvt_dicts"]
        this.subroutines=args["modular_agents_ocrNG"];
        # tells which module is which in modular_dict.
        # we may have two identical routines(DAN_char and DAN_word sharing only part of modules)
        # we recommend aliasing with bogo modules, so the config can be automatically figured out.
        # Or you will need to figure the cvtdicts your self :-)
        this.mod_cvt_dict=mod_cvt_dicts;
        this.inp_cvt_dict=inp_cvt_dicts;
        # r1d2 (x
        # well it actually checks if the routine needs non-existing modules or data..
        this.san_chk();
        pass;

    def grab_nested(this,moduleterm,modular_dict):
        if (type(moduleterm) is list):
            return [this.grab_nested(n,modular_dict)for n in moduleterm]
        else:
            return modular_dict[moduleterm];
    def grab_modules(this,input_dict,modular_dict):
        mdict={};
        idict={}
        for k in this.mod_cvt_dict:
            mdict[k] = this.grab_nested(this.mod_cvt_dict[k],modular_dict);
        for k in this.inp_cvt_dict:
            idict[k] = input_dict[this.inp_cvt_dict[k]];
        return idict,mdict;

    def act_core(this,input_dict,modular_dict,nEpoch,batch_idx,device):
        idict, mdict = this.grab_modules(input_dict, modular_dict, ctrl_modular_dict);
        workspace = neko_workspace(idict,nEpoch,batch_idx,device);
        # Note the logger goes here now :-)
        for r in this.subroutines:
            workspace = r.act(mdict, workspace);
        return workspace;

    # Nah we know we have losses... Why don't you just push loss to the bp you ask?
    # Short answer is we can't, as we may want to drop intermediate results to save space.
    def act(this, input_dict, modular_dict, ctrl_modular_dict, nEpoch, batch_idx, device):
        workspace=this.act_core(input_dict,modular_dict,nEpoch,batch_idx,device);
        fls=[];
        # loss is 0
        # weight of loss is 1
        for k in workspace.objdict:
            fls.append(workspace.objdict[k][0]*workspace.objdict[k][1]);
        total_loss=torch.cat(fls).sum();
        return total_loss;

    def bp_impl(this, loss):
        loss.backward();
    def actbp(this, input_dict, modular_dict, ctrl_modular_dict, nEpoch, batch_idx, device="cuda"):
        loss=this.act(input_dict,modular_dict,nEpoch,batch_idx,device);
        # If you want debugging, make bogo modules or modular_agents_ocrNG to do that.
        this.bp_impl(loss);
        # as we said, the logs now goes to their own modules.
    def actbp_amp(this, input_dict, modular_dict, ctrl_modular_dict, nEpoch, batch_idx, device):
        with torch.cuda.amp.autocast():
            this.actbp(input_dict, modular_dict, ctrl_modular_dict, nEpoch, batch_idx, device);
        pass;
    # Sometime logger_lo needs to be flushed at the end of epoch or something---
    # so they need an extra flusher, not here--- but in the trainer or tester.
    # Putting them here makes no sense at all--- they need to be called by trainer after all.


    def __call__(this, input_dict, modular_dict, ctrl_modular_dict, nEpoch, batch_idx,device="cuda",amp=False):
        if(amp):
            this.actbp_amp(input_dict, modular_dict, ctrl_modular_dict, nEpoch, batch_idx,device);
        else:
            this.actbp(input_dict, modular_dict, ctrl_modular_dict, nEpoch, batch_idx,device);
        return None;

# When you do not have too fancy names, you can automatically figure the configs out.
def build_ng_routine(subroutines:list[neko_module_wrapping_agent], inputs:list[str], prfx:str):
    inp_cvt_dict={};
    for i in inputs:
        inp_cvt_dict[i]=prfx+i;
    mods=set();
    for r in subroutines:
        for m_n in r.mnames:
            mods.add(m_n);
    mod_cvt_dict={}
    for m_n in mods:
        mod_cvt_dict[m_n]=prfx+m_n;
    return neko_routine_agent(
        {
            "inp_cvt_dict":inp_cvt_dict,
            "modcvt_dict":mod_cvt_dict,
            "modular_agents_ocrNG":subroutines,
        }
    );



