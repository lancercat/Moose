import datetime
import time

from neko_sdk.ocr_modules.sptokens import tUNKREP
import editdistance

class reporter_core:
    def reset(this,name):
        this.name=name;
        this.start_time=time.time();
        this.tot=0;
        this.corr=0;
        this.lcorr=0;
        this.tned=0;

    def record(this,gt,pr):
        if (len(gt) == len(pr)):
            this.lcorr+=1;
        pr=pr.replace(tUNKREP,"")
        this.corr+=(gt.lower()==pr.lower());
        lgt=len(gt);
        if(len(gt)==0):
            lgt=1;
        this.tned += editdistance.eval(gt,pr)/lgt;
        this.tot+=1;
    def report(this,eidx,bidx):
        try:
            print(
                "Date:", datetime.datetime.now(),
                ",TEST:", this.name,
                ",Epoch:", eidx,
                ",Iter:", bidx,
                ",Total:", this.tot,
                ",ACR:", this.corr / this.tot,
                ",ASNED:", this.tned / this.tot,
                ",Lenpred_ACR:", this.lcorr / this.tot,
                ",FPS:", this.tot / (time.time() - this.start_time),
            );
        except:
            pass;

class reporter_core_ARPF:
    def reset(this, name):
        this.name = name;
        this.correct = 0
        this.tot = 0.
        this.total_C = 0.
        this.total_W = 0.
        this.total_U = 0.
        this.total_K = 0.
        this.Ucorr = 0.
        this.Kcorr = 0.
        this.KtU = 0.

    def record(this, gt, pr, UNK="⑨"):
        this.tot += 1;
        this.total_C += len(gt)
        this.total_W += len(pr)
        cflag = int(gt == pr);
        this.correct = this.correct + cflag;
        if (gt.find(UNK) != -1):
            this.total_U += 1.;
            this.Ucorr += (pr.find(UNK) != -1);
        else:
            this.total_K += 1.;
            this.Kcorr += cflag;
            this.KtU += (pr.find(UNK) != -1);

    def report(this, eidx, bidx):
        if this.tot == 0:
            pass
        R = this.Ucorr / max(this.total_U, 1);
        P = this.Ucorr / max(this.Ucorr + this.KtU, 1);
        F = 2 * (R * P) / max(R + P, 1.)
        print({"Date": datetime.datetime.now(),
               "TEST": this.name,
               "Epoch": eidx,
               "Iter": bidx,
               "Total": this.tot,
               "KACR": this.Kcorr / max(1, this.total_K),
               "R": R, "P": P, "F": F}
              );

    def __init__(this):
        pass;
