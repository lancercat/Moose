import os.path
import shutil

import cv2
import numpy as np


def grab_image_and_recon(folder,id):
    imp=os.path.join(folder,str(id)+".jpg");
    recp=os.path.join(folder,str(id)+"prec.png");
    att=os.path.join(folder,str(id)+"att.png");
    if(not (os.path.exists(imp)and os.path.exists(recp))):
        return None;
    im=cv2.imread(imp);
    recon=cv2.imread(recp);
    att=cv2.imread(att);


    aim=att[:32];

    apadw = np.min(np.nonzero(aim.sum(0).sum(1)));
    corew=im.shape[1]-48;

    att_h,att_w=att.shape[0],att.shape[1];
    att_r=(corew/(att_w-apadw-apadw))
    att[:,:2,0]=255;
    att[:, -2:, 0] = 255;
    att[:2, :, 0] = 255;
    att[-2:, : , 0] = 255;
    att=cv2.resize(att,[int(att_r*att_w),int(att_r*att_h)]);

    apadw=int(apadw*att_r);


    lpad=48;
    lpadtot=max(lpad,apadw);
    rpadtot=apadw;

    lpadr=np.zeros([recon.shape[0],lpadtot,3],dtype=recon.dtype);
    rpadr=np.zeros([recon.shape[0],rpadtot,3],dtype=recon.dtype);
    reconpad=np.concatenate([lpadr,cv2.resize(recon,[im.shape[1]-48,recon.shape[0]]),rpadr],axis=1);


    iml=[];
    if(lpad<lpadtot):
        iml.append(np.zeros([im.shape[0],lpadtot-lpad,3],dtype=recon.dtype));
    iml.append(im);
    if (rpadtot > 0):
        iml.append(np.zeros([im.shape[0],rpadtot,3],dtype=recon.dtype))
    impad=np.concatenate(iml,axis=1);

    attl=[]
    if(apadw<lpadtot):
        attl.append(np.zeros([att.shape[0],lpadtot-apadw,3],dtype=recon.dtype));
    attl.append(att)
    attpad=np.concatenate(attl,axis=1);
    if(attpad.shape[1]!=impad.shape[1]):
        attpad=cv2.resize(attpad,[impad.shape[1],attpad.shape[0]])

    return np.concatenate([reconpad,impad,attpad],axis=0);

def compare(folders,id,ribbons=None):
    candidates=[]
    for i in range(len(folders)):
        folder =folders[i];
        ribbon=ribbons[i];
        cim=grab_image_and_recon(folder,id);
        if(cim is None):
            print(folder,id);
            return None;
        # cv2.imshow("a",ribbon);
        # cv2.waitKey(0);
        cim=np.concatenate([cv2.resize(ribbon,[cim.shape[1],64]),cim]);
        candidates.append(cim);
        candidates.append(np.zeros_like(cim[:,:16]));
    hsz=np.max([i.shape[0] for i in candidates]);
    wsz = np.sum([i.shape[1] for i in candidates]);
    canvas=np.zeros([hsz,wsz,3],dtype=candidates[0].dtype);
    woff=0;
    for c in candidates:
        dx=c.shape[1];
        canvas[:c.shape[0],woff:woff+dx]=c;
        woff+=dx;
    return canvas;
def get_logdir(root,method,tag="base_"):
    return os.path.join(root,method,tag+"prototyper/JAP_lang/");
def textribbon(text):
    im=cv2.putText(np.zeros([48, 1024, 3],dtype=np.uint8),text,org = (6, 30),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale = 1,color = (255, 255, 255),thickness=4);
    pad = np.max(np.nonzero(np.hstack(im.sum(0).sum(-1))));
    im=im[:,:pad+1];
    return im;

def compare_files(root,methods,dst,cnt,tags=None):
    if(tags is None):
        tags=["base_" for m in methods]
    ribbons=[textribbon(method.replace("open_","").replace("model","")) for method in methods];

    shutil.rmtree(dst,ignore_errors=True);
    os.makedirs(dst);
    folders=[get_logdir(root,method,tag)for tag,method in zip(tags,methods)];
    for i in range(cnt):
        try:
            crim=compare(folders,i,ribbons);
        except:
            print(folders,i);
            crim=None;
        if(crim is None):
            continue;
        cv2.imwrite(os.path.join(dst,str(i)+".jpg"),crim);
    return folders;

def compare_filesm(grp_folders,dst,cnt,tag="base_"):
    shutil.rmtree(dst,ignore_errors=True);
    os.makedirs(dst);
    for i in range(cnt):
        crims=[]
        flag=False;
        for folders in grp_folders:
            crim=compare(folders,i);
            crims.append(crim);
            if(crim is None):
                flag=True;
                break;
        if(flag):
            continue;
        for j in range(1,len(crims)):
            crims[j]=cv2.resize(crims[j],[crims[0].shape[1],crims[0].shape[0]]);
        crim=np.concatenate(crims,axis=0);
        cv2.imwrite(os.path.join(dst,str(i)+".jpg"),crim);
    return grp_folders;

def cmpmcsc():
    rootsc="/run/media/lasercat/ssddata/all_285/";
    methods_sc = [os.path.join(rootsc, i,"jtrmodels/closeset_benchmarks/base_chs_prototyper/JAP_lang/") for i in
                  ["open_basemodel_protorec","open_basemodel_protorec_dommixMVA","open_dtmodel_protorec_dommixMVA","open_basemodel_protorec_cycleNS","open_basemodel_protorec_cycleNS2"]
    ];
    rootmc="/run/media/lasercat/ssddata/mc-285/";
    methods_mc = [ os.path.join(rootmc,i,"jtrmodels/closeset_benchmarks/base_chs_prototyper/JAP_lang/") for i in
        ["open_mc_basemodel_protorec", "open_mc_basemodel_protorec_dommixMVA","open_mc_dtmodel_protorec_dommixMVA","open_mc_basemodel_protorec_cycleNS","open_mc_basemodel_protorec_cycleNS2"]
    ];
    dst="/run/media/lasercat/ssddata/soai_zero/";
    compare_filesm([methods_sc,methods_mc],dst,4009);


def cmp285():
    root="/run/media/lasercat/ssddata/all_285/";
    methods=["open_basemodel_protorec","open_basemodel_protorec_dommixMVA","open_dtmodel_protorec_dommixMVA","open_basemodel_protorec_cycleNS","open_basemodel_protorec_cycleNS2"];
    vdst=os.path.join(root,"rec_cmp");
    compare_files(root,methods,vdst,4009);
def visprotorec():
    root="/run/media/lasercat/ssddata/all_285/";
    methods=["open_basemodel_protorec"];
    vdst=os.path.join(root,"rec_cmp");
    compare_files(root,methods,vdst,4009);
def cmp285_mc():
    root = "/run/media/lasercat/ssddata/mc-285/";
    methods = ["open_mc_basemodel_protorec", "open_mc_basemodel_protorec_dommixMVA","open_mc_dtmodel_protorec_dommixMVA","open_mc_basemodel_protorec_cycleNS","open_mc_basemodel_protorec_cycleNS2"];
    vdst = os.path.join(root, "rec_cmp");
    compare_files(root, methods, vdst, 4009);
def cmp285_da_mc():
    root = "/run/media/lasercat/ssddata/mc-285/";
    methods = [ "open_mc_basemodel_protorec_dommixMVA","open_mc_dtmodel_protorec_dommixMVA","open_mc_dtmmodel_protorec_dommixMVA","open_mc_dt2model_protorec_dommixMVA"];
    vdst = os.path.join(root, "cmpda");
    compare_files(root, methods, vdst, 4009);
def cmp285da():
    root="/run/media/lasercat/ssddata/all_285/";
    methods=["open_basemodel_protorec","open_basemodel_protorec_dommixF","open_basemodel_protorec_dommixFS","open_basemodel_protorec_dommixMVA","open_basemodel_protorec_dommixFS_MVA"];
    vdst=os.path.join(root,"dacmp");
    compare_files(root,methods,vdst,4009);
def cmp285ns():
    root="/run/media/lasercat/ssddata/all_285/";
    methods=["open_basemodel_protorec","open_basemodel_protorec_dommixFS","open_basemodel_protorec_dommixMVA","open_basemodel_protorec_cycleNS2"];
    vdst=os.path.join(root,"nscmp");
    compare_files(root,methods,vdst,4009);
def cmp285pshufmc():
    root = "/run/media/lasercat/ssddata/mc-285/";
    methods = ["open_mc_basemodeldf_protorec_patchshuf", "open_mc_basemodeldf_protorec_patchshuf_FS","open_mc_basemodeldf_protorec_patchshuf_NS3","open_mc_basemodeldf_protorec_patchshuf_MVA"];
    vdst = os.path.join(root, "rec_cmp");
    compare_files(root, methods, vdst, 4009);
def cmpaaslsct():
    root = "/run/media/lasercat/ssddata/mc-285/";
    methods = ["open_mc_basemodeldf_protorec_patchshuf", "open_mc_basemodeldf_protorec_patchshuf_FS",
               "open_mc_basemodeldf_protorec_patchshuf_NS3", "open_mc_basemodeldf_protorec_patchshuf_MVA"];
    vdst = os.path.join(root, "rec_cmp");
    compare_files(root, methods, vdst, 4009);
if __name__ == '__main__':
    # cmpmcsc();
    # cmp285_mc();
    # cmp285();
    # visprotorec();
    # cmp285da();
    #cmp285ns();
    cmp285pshufmc();