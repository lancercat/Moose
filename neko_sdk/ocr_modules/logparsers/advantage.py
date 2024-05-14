import os.path
import shutil

import cv2
import editdistance as ed
import numpy as np


def quality(root,id,prefx="",postfx=".txt"):
    with open(root + prefx+str(id) + postfx, "r") as ifp:
        [gt, pr] = [i.strip() for i in ifp][:2];
    return 1-(ed.eval(pr,gt)/max(len(pr),len(gt))),pr==gt,[gt, pr];
def combine(ref_img_path,method_img_path):
    rim=cv2.imread(ref_img_path);
    mim=cv2.imread(method_img_path);
    mr=max(rim.shape[0],mim.shape[0]);
    mc=max(rim.shape[1],mim.shape[1]);
    return np.concatenate([cv2.resize(rim,[mc,mr]),cv2.resize(mim,[mc,mr])],axis=1);


def advantage(ref_path,ref_imgpath,method_path,method_img_path,dst_root,cnt=4009,resprefx="",respostfx=".txt",imgprefx="",imgpostfx=".jpg"):
    types=[os.path.join(dst_root,i) for i in ["as_good","worse","better","as_bad","fatal","critical"]];
    for i in types:
        shutil.rmtree(i,ignore_errors=True);
        os.makedirs(i);
    for i in range(cnt):
        try:
            refq,refc,ref_det=quality(ref_path,i,resprefx,respostfx);
            methodq,methodc,metdet = quality(method_path, i, resprefx, respostfx);
            cim=combine(
                os.path.join(ref_imgpath,imgprefx+str(i)+imgpostfx),
                os.path.join(method_img_path, imgprefx + str(i) + imgpostfx)
            );
            if(refc and methodc):
                did=0;
            elif(refq>methodq):
                did=1;
            elif(refq<methodq):
                did=2;
            else:
                did=3;
            cv2.imwrite(os.path.join(types[did],imgprefx+str(i)+imgpostfx),cim);
            if(refc and not methodc):
                cv2.imwrite(os.path.join(types[4], imgprefx + str(i) + imgpostfx), cim);
            if(not refc and methodc):
                cv2.imwrite(os.path.join(types[5], imgprefx + str(i) + imgpostfx), cim);

        except:
            print(i);



if __name__ == '__main__':
    advantage("/run/media/lasercat/ssddata/project_290_dump/318prirC/open_basemodelS_DT48_run3/GZSL/base_chs_prototyper/JAP_lang/",
              "/run/media/lasercat/ssddata/project_290_dump/318prirC/open_basemodelS_DT48_run3/GZSL/base_chs_prototyper/JAP_lang/",
              "/run/media/lasercat/ssddata/project_290_dump/318prirC/open_ssr_r3g3_mpfS_DT48_D10k_3sp_pm_run3/GZSL/base_chs_prototyper/JAP_lang/",
              "/run/media/lasercat/ssddata/project_290_dump/318prirC/open_ssr_r3g3_mpfS_DT48_D10k_3sp_pm_run3/GZSL/base_chs_prototyper/JAP_lang/",
              "/run/media/lasercat/ssddata/project_290_dump/318prirC/advantage_r3");
    advantage(
        "/run/media/lasercat/ssddata/project_290_dump/318prirC/open_basemodelS_DT48_run3/GZSL/base_chs_prototyper/JAP_lang/",
        "/run/media/lasercat/ssddata/project_290_dump/318prirC/open_basemodelS_DT48_run3/GZSL/base_chs_prototyper/JAP_lang/",
        "/run/media/lasercat/ssddata/project_290_dump/318prirC/open_ssr_r3g3_mpfS_DT48_D10k_3sp_pm_run3/GZSL/base_chs_prototyper/JAP_lang/",
        "/run/media/lasercat/ssddata/project_290_dump/318prirC/open_ssr_r3g3_mpfS_DT48_D10k_3sp_pm_run3/GZSL/base_chs_prototyper/JAP_lang/",
        "/run/media/lasercat/ssddata/project_290_dump/318prirC/advantage_r3");

    # advantage("/run/media/lasercat/ssddata/all_283/318prirC/open_basemodel_prec_shufIN3_g3kDPs/jtrmodels/closeset_benchmarks/shuf3_chs_prototyper/JAP_lang/",
    #           "/run/media/lasercat/ssddata/all_283/318prirC/open_basemodel_prec_shufIN3_g3kDPs/dashboard",
    #           "/run/media/lasercat/ssddata/all_283/318prirC/open_basemodel_prec_shufIN3_g3kDPs_dis_dmxdp/jtrmodels/closeset_benchmarks/shuf3_chs_prototyper/JAP_lang/",
    #           "/run/media/lasercat/ssddata/all_283/318prirC/open_basemodel_prec_shufIN3_g3kDPs_dis_dmxdp/dashboard",
    #           "/run/media/lasercat/ssddata/advantage/dis_dmxdp")