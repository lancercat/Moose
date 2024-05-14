
import cv2
import glob
import os

from neko_sdk.MJT.lanuch_std_test import testready
from neko_sdk.athena.osocr.open_test import img_test;
from neko_sdk.ocr_modules.result_renderer import render_word
from neko_sdk.athena.common.analyze_folder import bootstrap_folder


def run_athena_folder(root,dst,lang,taskbuilder,argv):
    files,ptfile,sfolder,dfolder=bootstrap_folder(root,dst,lang);
    runner,globalcache,mdict=testready(argv,ptfile,taskbuilder);
    for i in range(len(files)):
        res = img_test(files[i],
                       runner,
                       globalcache);
        base = os.path.basename(files[i]);
        dstt = os.path.join(dfolder, base.replace("jpg", "txt"));
        dsti = os.path.join(dfolder, base);
        dim, _ = render_word(mdict, {}, cv2.imread(files[i]), None, res, 0);
        cv2.imwrite(dsti, dim);
        print(res);
        with open(dstt, "w+") as fp:
            fp.writelines(res);


def run_athena(root, dst,taskbuilder, argv):
    os.makedirs(dst, exist_ok=True)
    langs = glob.glob(os.path.join(root, "lang_*"));
    for lang in langs:
        run_athena_folder(root, dst, lang, taskbuilder, argv);
