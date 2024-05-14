
import glob
import os

from neko_sdk.athena.common.quickptg1 import prepare_pt


def bootstrap_folder(root,dst,lang,pfix="*.jpg"):
    ptfile=prepare_pt(os.path.join(root,lang));
    sfolder=os.path.join(root,lang);
    dfolder=os.path.join(dst,os.path.basename(lang),"results");
    os.makedirs(dfolder,exist_ok=True);
    files = glob.glob(os.path.join(sfolder, pfix));
    return files,ptfile,sfolder,dfolder;

