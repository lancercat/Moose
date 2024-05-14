import glob
import os

from neko_sdk.ocr_modules.collect_log import bootstrap_folders;
from neko_sdk.ocr_modules.collect_log import copylog;


def register_method_tr(log,eitr,vds):
    with open(log,"r") as fp:
        lines=[i.strip() for i in fp];
    itr=-1;
    epoc=-1;
    titr=-1;
    dsd={};
    ds="Meh";
    ttl="Meh";
    xoff=0;
    for i in lines:
        if (i.find("ACR") == -1):
            continue;
        if(i.find("TEST")==-1):
            continue; # wandb noise
        d = {}
        terms = i.split(" ,")
        for t in terms:
            k, v = t.split(": ", 1);
            d[k] = v;
        ds = d["TEST"];
        if i.find("starts") != -1:
            ds = i.rsplit(" ", 1)[0];
        titr = int(d["Epoch"]) * 200000 + int(d["Iter"]);
        if (titr < 0):
            continue;
        if (ds not in vds):
            continue;
        if (ds not in dsd):
            dsd[ds] = {};
        dsd[ds][titr + xoff] = [float(d["ACR"])];

    if(titr<20000):
        return None;
    return dsd;
def get_resd_tr(root_dirs,pats,bundles,eitr,vds):
    alog=[];
    for root_dir in root_dirs:
        for pat in pats:
            logs=glob.glob(os.path.join(root_dir,"*",pat,"PLAYDAN*.log"));
            xlogs=glob.glob(os.path.join(root_dir,"*",pat,"PLAYDAN_XL.log"));
            alog+=logs;
            alog+=xlogs
    keys=[os.path.basename(os.path.dirname(i)).rsplit("/",1)[0] for  i in alog];
    devs=[os.path.basename(os.path.dirname(os.path.dirname(i))) for  i in alog];

    adsd={}
    for log,dev,key in zip(alog,devs,keys):
        ddsd={key:register_method_tr(log,eitr,vds)};
        if(ddsd is None):
            continue;
        for key in ddsd:
            dsd=ddsd[key]
            if(dsd is None):
                print("omitting ",log);
                continue;
            for k in dsd:
                if(k not in adsd):
                    adsd[k]={};
                if(key not in adsd[k]):
                    adsd[k][key]={}
                if dev not in adsd[k][key]:
                    adsd[k][key][dev]={};
                adsd[k][key][dev]=dsd[k];
    return adsd;

def collect_resd(dev_metas,method_metas,bundles,vds,workingdir,projects,pats=["*long*"],eitr=200000):
    bootstrap_folders(workingdir, dev_metas)
    draw_meta = {};
    for k in method_metas:
        if (k.find("aalr") == -1):
            draw_meta[k] = method_metas[k];
    roots=[os.path.join(workingdir,p,"raw")for p in projects];
    for i in range(len(roots)):
        for d in dev_metas:
            for m in draw_meta:
                copylog(dev_metas[d], roots[i], m, projects[i]);
        os.system("rmdir " + os.path.join(roots[i], "*", "*"));
        os.system("rmdir " + os.path.join(roots[i], "*"));
    resd = get_resd_tr(roots, pats, bundles, eitr, vds);
    # resd=get_resd_tr("/home/lasercat/ssddata/standardbench2_candidates/",["*mk*e2*"])
    return resd;

