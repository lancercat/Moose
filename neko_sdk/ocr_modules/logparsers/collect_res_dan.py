import glob
import os

from neko_sdk.ocr_modules.collect_log import bootstrap_folders;
from neko_sdk.ocr_modules.collect_log import copylog;


def register_method_tr_bundle(log,bdict,eitr,vds):
    with open(log,"r") as fp:
        lines=[i.strip() for i in fp];
    itr=-1;
    epoc=-1;
    titr=-1;
    ddsd={};
    ds="Meh";
    ttl="Meh";
    xoff=0;
    dsd=None;
    for k in bdict:
        ddsd[bdict[k]["mname"]]={};
    for i in lines:
        if(i.find("[A]")!=-1 or i.find("[B]")!=-1 or i.find("[C]")!=-1or i.find("[D]")!=-1):
            continue;
        if i.find("ends") !=-1:
            ds="Meh"
        if i.find("starts") !=-1:
            ds=i.rsplit(" ",1)[0];
        elif (i.find("test_accr")!=-1):
            bname=i[1:].split("]")[0]
            if(bname not in bdict):
                print("error");
                exit(9);
            else:
                mname = bdict[bname]["mname"];
                dsd=ddsd[mname];
        elif(i[:8]=="Accuracy"):
            ttl = "Meh";
            if(titr<0):
                continue;
            if(ds not in vds):
                continue;
            if(ds not in dsd):
                dsd[ds]={};
            dsd[ds][titr+xoff]=[float(t.split(":")[1]) for t in i.split(",")];


        elif(i[:2]=="(0"):
            continue;
        elif (len(i.split(" ")) == 2 and i.find(".") == -1 and i.find("n") == -1   and i.find("p") == -1 and i.find("G") ==-1 and i.find("decay")==-1and i.find("itrtime")==-1):
            try:
                epoc, itr = [int(t) for t in i.split(" ")];
            except:
                continue;
            titr=epoc*eitr+itr
            print(epoc,eitr,titr)
        elif(len(i.split(" ")) == 2 and i.split(" ")[1]=="Final"):
            epoc=int(i.split(" ")[0]);
            itr = 200000;
            titr=epoc*eitr+itr
            print(epoc,eitr,titr,"Final")


    if(titr<20000):
        return None;
    return ddsd;
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
        if(i.find("[A]")!=-1 or i.find("[B]")!=-1 or i.find("[C]")!=-1or i.find("[D]")!=-1):
            continue;
        if i.find("ends") !=-1:
            ds="Meh"
        if i.find("starts") !=-1:
            ds=i.rsplit(" ",1)[0];
        elif(i[:8]=="Accuracy"):
            ttl = "Meh";
            if(titr<0):
                continue;
            if(ds not in vds):
                continue;
            if(ds not in dsd):
                dsd[ds]={};
            dsd[ds][titr+xoff]=[float(t.split(":")[1]) for t in i.split(",")];
        elif (i.find("test_accr")!=-1):
            if(i.find("base_chs_xl_close_set_benchmark")!=-1):
                xoff=0;
                ds+="XL"
            elif(i.find("base_mjst_xl_close_set_benchmark")!=-1):
                xoff=0;
                ds+="XL"
            else:
                xoff=0
        elif(i[:2]=="(0"):
            continue;
        elif (len(i.split(" ")) == 2 and i.find(".") == -1 and i.find("n") == -1   and i.find("p") == -1 and i.find("G") ==-1 and i.find("decay")==-1and i.find("itrtime")==-1):
            try:
                epoc, itr = [int(t) for t in i.split(" ")];
            except:
                continue;
            titr=epoc*eitr+itr
            print(epoc,eitr,titr)
        elif(len(i.split(" ")) == 2 and i.split(" ")[1]=="Final"):
            epoc=int(i.split(" ")[0]);
            itr = 200000;
            titr=epoc*eitr+itr
            print(epoc,eitr,titr,"Final")


    if(titr<20000):
        return None;
    return dsd;
def get_resd_tr(root_dirs,pats,bundles,eitr,vds):
    alog=[];
    for root_dir in root_dirs:
        for pat in pats:
            logs=glob.glob(os.path.join(root_dir,"*",pat,"PLAYDAN.log"));
            xlogs=glob.glob(os.path.join(root_dir,"*",pat,"PLAYDAN_XL.log"));
            alog+=logs;
            alog+=xlogs
    keys=[os.path.basename(os.path.dirname(i)).rsplit("/",1)[0] for  i in alog];
    devs=[os.path.basename(os.path.dirname(os.path.dirname(i))) for  i in alog];

    adsd={}
    for log,dev,key in zip(alog,devs,keys):
        if(key in  bundles):
            ddsd=register_method_tr_bundle(log,bundles[key],eitr,vds)
        else:
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

def collect_resd(dev_metas,method_metas,bundles,vds,workingdir,projects,pats=["open*","close*"],eitr=200000):
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

