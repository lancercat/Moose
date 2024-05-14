import random

import regex

from neko_sdk.lmdb_wrappers.im_lmdb_wrapper import im_lmdb_wrapper;
from neko_sdk.lmdb_wrappers.ocr_lmdb_reader import neko_ocr_lmdb_mgmt;


def hfilter(root,dst):
    from PIL import Image
    import six
    seendb = im_lmdb_wrapper(dst);
    db=neko_ocr_lmdb_mgmt(root,False,1000);
    charset=set();
    for i in range(len(db)):
        im,t=db.getitem_encoded_kv(i,["image"],["label", "lang"]);
        buf = six.BytesIO()
        buf.write(im[0])
        buf.seek(0)
        try:
            img = Image.open(buf)
        except IOError:
            print(f'Corrupted image for ', t);
            continue;
        if(img.size[0]<img.size[1]):
            continue;
        seendb.adddata_kv({},{"label":t[0],"lang":t[1]},{"image":im[0]});
    return charset;
def sfilter(root,chasets,seendst):
    from PIL import Image
    import six
    seendb = im_lmdb_wrapper(seendst);
    db=neko_ocr_lmdb_mgmt(root,False,1000);
    charset=set();
    for i in range(len(db)):
        nt="";
        im,t=db.getitem_encoded_kv(i,["image"],["label", "lang"]);
        buf = six.BytesIO()
        buf.write(im[0])
        buf.seek(0)
        try:
            img = Image.open(buf)
        except IOError:
            print(f'Corrupted image for ', t);
            continue;
     
        seen=True;
        ust=None;
        for c in regex.findall(r'\X', t[0], regex.U) :
            if c not in chasets:
                seen=False;
                ust=c;
            nt+=c;
        if(seen):
            seendb.adddata_kv({}, {"label": t[0], "lang": t[1]}, {"image": im[0]});
        else:
            print(t[0])
            print(ust)
            for c in regex.findall(r'\X', t[0], regex.U) :
                if c not in chasets:
                    print(c)
            pass;
    return charset;

def shfilter(root,chasets,seendst):
    from PIL import Image
    import six
    seendb = im_lmdb_wrapper(seendst);
    db=neko_ocr_lmdb_mgmt(root,False,1000);
    charset=set();
    for i in range(len(db)):
        nt="";
        im,t=db.getitem_encoded_kv(i,["image"],["label", "lang"]);
        buf = six.BytesIO()
        buf.write(im[0])
        buf.seek(0)
        try:
            img = Image.open(buf)
        except IOError:
            print(f'Corrupted image for ', t);
            continue;
        if(img.size[0]<img.size[1]):
            continue;

        seen=True;
        ust=None;
        for c in regex.findall(r'\X', t[0], regex.U) :
            if c not in chasets:
                seen=False;
                ust=c;
            nt+=c;
        if(seen):
            seendb.adddata_kv({}, {"label": t[0], "lang": t[1]}, {"image": im[0]});
        else:
            print(t[0])
            print(ust)
            for c in regex.findall(r'\X', t[0], regex.U) :
                if c not in chasets:
                    print(c)
            pass;
    return charset;

def subset(root,dst,cnt):
    from PIL import Image
    import six
    seendb = im_lmdb_wrapper(dst);
    db=neko_ocr_lmdb_mgmt(root,True,1000);
    charset=set();
    all=list(range(1,len(db)))
    random.shuffle(all);
    sub = all[:cnt];

    for i in sub:
        nt="";
        im,t=db.getitem_encoded_kv(i,["image"],["label"]);
        buf = six.BytesIO()
        buf.write(im[0])
        buf.seek(0)
        try:
            img = Image.open(buf)
        except IOError:
            print(f'Corrupted image for ', t);
            continue;
        seendb.adddata_kv({}, {"label": t[0]}, {"image": im[0]});
    # seendb.end_this();



def harvast_cs(root):

    db=neko_ocr_lmdb_mgmt(root,False,1000);
    charset=set();
    for i in range(len(db)):
        nt="";
        im,t=db.getitem_encoded_kv(i,["image"],["label", "lang"]);
        for c in regex.findall(r'\X', t[0], regex.U) :
            if c not in charset:
                charset.add(c);

    return charset;
