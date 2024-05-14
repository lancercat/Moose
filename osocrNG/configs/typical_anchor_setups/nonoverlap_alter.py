# main
# anchors describe knowledge. Whether or not to hornor is the call of the model
def get_v4_core(name,keys,ratios):
    ad={  "long":{
            "batch_size":48,
            "target_size":(32*5,32),
            "beacon_size":(64,64),
            "maxT": 30,
            "possible_rotation": [0]
        },
        "short":{
            "batch_size": 48,
            "target_size":(32*3,32*2),
            "beacon_size": (64, 64),
            "maxT":25,
            "possible_rotation": [0]
        },
        "vert":{
            "batch_size":48,
            "target_size":(32,32*5),
            "beacon_size": (64, 64),
            "maxT": 30,
            "possible_rotation":[0,3]
        }};
    ret={
        "names":keys,
    };
    for n,r in zip(keys,ratios):
        ret[n]=ad[n];
        ret[n]["ratio"]=r;
        if(r>0):
            name+="_"+str(r);
    ret["profile_name"] =name.replace(".","");
    return ret;

def get_hydra_v4_anchor_2h1v_3_05():
    return get_v4_core(
        "moose_mk2",
        ["long", "short", "vert"],
        [3,0.5,-100]
    );
def get_hydra_v4_anchor_2h1Tv_3_05():
    d=get_v4_core(
        "moose_mk2VT",
        ["long", "short", "vert"],
        [3,0.5,-100]
    )
    d["vert"]["force_transpose"]=1;
    return d;
def get_hydra_v4_anchor_2h0v_3_05():
    d=get_v4_core(
        "moose_mk2_hori",
        ["long", "short"],
        [3, 0.5]
    );
    return d;

def get_hydra_v4_anchor_1h1v_l_05():
    d=get_v4_core(
        "moose_mk2L",
        ["long", "vert"],
        [0.5, -100]
    );
    return d;

def get_hydra_v4_anchor_1h1v_s_05():
    d=get_v4_core(
        "moose_mk2S",
        ["short", "vert"],
        [0.5, -100]
    );
    return d;
