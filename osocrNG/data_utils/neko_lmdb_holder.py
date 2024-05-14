import sys

import lmdb
import six
from PIL import Image

from neko_sdk.neko_framework_NG.agents.neko_data_source_NG import neko_abstract_data_source_NG


class neko_lmdb_holder(neko_abstract_data_source_NG):

    def __getstate__(this):
        state = this.__dict__
        state["env"] = None;
        state["txn"] = None;
        return state

    def __setstate__(this, state):
        this.__dict__ = state
        env = lmdb.open(
            this.lmdb_root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False);
        this.env = env;
        this.reset_txn();

    def reset_txn(this):
        this.txn = this.env.begin(write=False)

    def init_etc(this, para):
        pass;

    def disarm(this):
        this.root = None;
        this.envs = None;
        this.nSamples = 0;

    def arm_lmdb(this, root):

        this.lmdb_root = root;
        env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)
        if not env:
            print('cannot creat lmdb from %s' % (root[i]))
            sys.exit(0)
        with env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            this.nSamples += nSamples
        this.env = env;
        this.reset_txn();

    def __len__(this):
        return this.nSamples

    def setup(this, para):
        this.disarm();
        this.arm_lmdb(para["root"]);
        this.vert_to_hori = -100000;
        if ("vert_to_hori" in para):
            this.vert_to_hori = para["vert_to_hori"]
        this.init_etc(para);

    def fetch_core(this, index):

        img_key = 'image-%09d' % index;
        label_key = 'label-%09d' % index;
        label = str(this.txn.get(label_key.encode()).decode('utf-8'));
        imgbuf = this.txn.get(img_key.encode())
        buf = six.BytesIO();
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf);
        if (img.width / img.height < this.vert_to_hori):
            img = img.transpose(Image.ROTATE_90);
        ret = {
            "image": img,
            "label": label,
        };
        return ret;

    def fetch_item(this, descp):
        index = (descp["id"]);
        ret = None;
        try:
            ret = this.fetch_core(index);
        except:
            try:
                this.reset_txn();
                ret = this.fetch_core(descp);
                print("bad_txn", "resetted");
            except:
                print('Corrupted image for %d' % index);
        return ret;

    def all_valid_indexes(this):
        return [{"id": i} for i in range(this.nSamples)];
