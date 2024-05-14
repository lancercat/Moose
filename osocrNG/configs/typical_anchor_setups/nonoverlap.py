# main
# anchors describe knowledge. Whether or not to hornor is the call of the model
def get_hydra_v3_anchor_2h1v_6_05():
    return {
        "profile_name":"hydra_6_05",
        "names":["long","short","vert"],
        "long":{
            "ratio":6,
            "batch_size":24,
            "target_size":(256,32),
            "beacon_size":(64,64),
            "maxT": 40,
            "possible_rotation": [0]
        },
        "short":{
            "ratio":0.5,
            "batch_size": 48,
            "target_size":(128,32),
            "beacon_size": (64, 64),
            "maxT":25,
            "possible_rotation": [0]
        },
        "vert":{
            "batch_size":24,
            "ratio":-100,
            "target_size":(32,256),
            "beacon_size": (64, 64),
            "maxT": 30,
            "possible_rotation":[0,3]
        },
    };
def get_hydra_v3_anchor_2h1Tv_6_05():
    return {
        "profile_name":"hydra_6_05Tv",
        "names":["long","short","vert"],
        "long":{
            "ratio":6,
            "batch_size":24,
            "target_size":(256,32),
            "beacon_size":(64,64),
            "maxT": 40,
            "possible_rotation": [0]
        },
        "short":{
            "ratio":0.5,
            "batch_size": 48,
            "target_size":(128,32),
            "beacon_size": (64, 64),
            "maxT":25,
            "possible_rotation": [0]
        },
        "vert":{
            "force_transpose":1,
            "batch_size":24,
            "ratio":-100,
            "target_size":(32,256),
            "beacon_size": (64, 64),
            "maxT": 30,
            "possible_rotation":[0,1]
        },
    };

def get_hydra_v3_anchor_2h0v_6():
    return {
        "profile_name":"hydra_6_novert",
        "names":["long","short"],
        "long":{
            "ratio":6,
            "batch_size":24,
            "target_size":(256,32),
            "beacon_size":(64,64),
            "maxT": 40,
            "possible_rotation": [0]
        },
        "short":{
            "ratio":0.5,
            "batch_size": 48,
            "target_size":(128,32),
            "beacon_size": (64, 64),
            "maxT":25,
            "possible_rotation": [0]
        },
    };



def get_hydra_v3_anchor_2h0v_6():
    return {
        "profile_name":"hydra_6",
        "names":["long","short"],
        "long":{
            "ratio":6,
            "force_rotate_to_hori": True,
            "batch_size":24,
            "target_size":(256,32),
            "beacon_size":(64,64),
            "maxT": 40,
            "possible_rotation": [0]
        },
        "short":{
            "ratio":0.5,
            "force_rotate_to_hori": True,
            "batch_size": 48,
            "target_size":(128,32),
            "beacon_size": (64, 64),
            "maxT":25,
            "possible_rotation": [0]
        },
    };
##### One horizontal
def get_hydra_v3_anchor_1h0v_05s():
    return {
        "profile_name":"hydra_05s",
        "names":["short"],
        "short":{
            "ratio":0.5,
            "force_rotate_to_hori": True,
            "batch_size": 48,
            "target_size":(128,32),
            "beacon_size": (64, 64),
            "maxT":25,
            "possible_rotation": [0]
        },
    };
def get_hydra_v3_anchor_1h0v_05l():
    return {
        "profile_name":"hydra_05l",
        "names":["long"],
        "long":{
            "ratio":0.5,
            "force_rotate_to_hori": True,
            "batch_size":24,
            "target_size":(256,32),
            "beacon_size":(64,64),
            "maxT": 40,
            "possible_rotation": [0]
        },
    };
def get_hydra_v3_anchor_1h1v_05(bss=48):
    return {
        "profile_name":"hydra_05",
        "names":["hori","vert"],
        "hori":{
            "ratio":0.5,
            "batch_size": bss,
            "target_size":(128,32),
            "beacon_size": (64, 64),
            "maxT":40
        },
        "vert":{
            "batch_size":bss,
            "ratio":-100,
            "target_size":(32,256),
            "beacon_size": (64, 64),
            "maxT": 30,
        },
    };
# another if we have vert thresh set to 1
def get_hydra_v3_anchor_2h1v_6_1():
    return {
        "profile_name":"hydra_6_1",
        "names":["long","short","vert"],
        "long":{
            "ratio":6,
            "batch_size":24,
            "target_size":(256,32),
            "beacon_size":(64,64),
            "maxT": 40,
            "possible_rotation": [0]
        },
        "short":{
            "ratio":1,
            "batch_size": 48,
            "target_size":(128,32),
            "beacon_size": (64, 64),
            "maxT":25,
            "possible_rotation": [0]
        },
        "vert":{
            "batch_size":24,
            "ratio":-100,
            "target_size":(32,256),
            "beacon_size": (64, 64),
            "maxT": 30,
            "possible_rotation":[0,3]
        },
    };

#####Alternative designs (try to keep same with horizontal setup)
def get_hydra_v3_anchor_1h1v_1(bss=48):
    return {
        "profile_name":"debug_1",
        "names":["long","vert"],
        "long":{
            "ratio":1,
            "batch_size": bss,
            "target_size":(128,32),
            "beacon_size": (64, 64),
            "maxT":30
        },
        "vert":{
            "batch_size":bss,
            "ratio":-100,
            "target_size":(32,256),
            "beacon_size": (64, 64),
            "maxT": 30,
        },
    };





####################History
def get_hydra_v1_anchor():
    return {
        "profile_name":"v1h",
        "names":["long","short","vert"],
        "long":{
            "ratio":6,
            "batch_size":48,
            "target_size":(320,32),
            "beacon_size":(64,64),
            "maxT": 40,
        },
        "short":{
            "ratio":1,
            "batch_size": 48,
            "target_size":(192,48),
            "beacon_size": (64, 64),
            "maxT":20
        },
        "vert":{
            "batch_size":48,
            "ratio":-100,
            "target_size":(48,192),
            "beacon_size": (64, 64),
            "maxT": 30,
        },
    };


def get_hydra_v1_anchor_smol():
    return {
        "profile_name":"v1h",
        "names":["long","short","vert"],
        "long":{
            "ratio":6,
            "batch_size":20,
            "target_size":(320,32),
            "beacon_size":(64,64),
            "maxT": 40,
        },
        "short":{
            "ratio":1,
            "batch_size": 48,
            "target_size":(128,32),
            "beacon_size": (64, 64),
            "maxT":20
        },
        "vert":{
            "batch_size":48,
            "ratio":-100,
            "target_size":(32,128),
            "beacon_size": (64, 64),
            "maxT": 30,        },
    };


def get_hydra_v2_anchor():
    return {
        "profile_name":"hydra_6_2_05",
        "names":["long","short","block","vert"],
        "long":{
            "ratio":6,
            "batch_size":20,
            "target_size":(320,32),
            "beacon_size":(64,64),
            "maxT": 40,
        },
        "short":{
            "ratio":2,
            "batch_size": 48,
            "target_size":(128,32),
            "beacon_size": (64, 64),
            "maxT":25
        },
        "block": {
            "ratio": 0.5,
            "batch_size": 48,
            "target_size": (64, 64),
            "beacon_size": (64, 64),
            "maxT": 15
        },
        "vert":{
            "batch_size":48,
            "ratio":-100,
            "target_size":(32,128),
            "beacon_size": (64, 64),
            "maxT": 30,
        },
    };


def get_fakehydra_anchor():
    return {
        "profile_name":"fakehydra",
        "names": ["main"],
        "main":{
            "batch_size": 48,
            "ratio":-100,
            "target_size":(192,48),
            "beacon_size": (48, 48),
        },
    };


def get_fakehydra_anchor_smol():
    return {
        "profile_name":"fakehydra",
        "names": ["main"],
        "main":{
            "batch_size": 48,
            "ratio":-100,
            "target_size":(128,32),
            "beacon_size": (48, 48),
        },
    };


def get_hydra_v1_anchor_debug_so():
    return {
        "names":["short"],
        "long":{
            "ratio":6,
            "target_size":(320,32)
        },
        "short":{
            "ratio":1,
            "target_size":(128,32)
        },
        "vert":{
            "ratio":-100,
            "target_size":(32,128)
        },
    };



def get_hydra_v1_anchor():
    return {
        "profile_name":"v1h",
        "names":["long","short","vert"],
        "long":{
            "ratio":6,
            "batch_size":48,
            "target_size":(320,32),
            "beacon_size":(64,64),
            "maxT": 40,
        },
        "short":{
            "ratio":1,
            "batch_size": 48,
            "target_size":(192,48),
            "beacon_size": (64, 64),
            "maxT":20
        },
        "vert":{
            "batch_size":48,
            "ratio":-100,
            "target_size":(48,192),
            "beacon_size": (64, 64),
            "maxT": 30,
        },
    };
def get_hydra_v1_anchor_smol():
    return {
        "profile_name":"v1h",
        "names":["long","short","vert"],
        "long":{
            "ratio":6,
            "batch_size":20,
            "target_size":(320,32),
            "beacon_size":(64,64),
            "maxT": 40,
        },
        "short":{
            "ratio":1,
            "batch_size": 48,
            "target_size":(128,32),
            "beacon_size": (64, 64),
            "maxT":20
        },
        "vert":{
            "batch_size":48,
            "ratio":-100,
            "target_size":(32,128),
            "beacon_size": (64, 64),
            "maxT": 30,        },
    };
def get_hydra_v2_anchor():
    return {
        "profile_name":"hydra_6_2_05",
        "names":["long","short","block","vert"],
        "long":{
            "ratio":6,
            "batch_size":20,
            "target_size":(320,32),
            "beacon_size":(64,64),
            "maxT": 40,
        },
        "short":{
            "ratio":2,
            "batch_size": 48,
            "target_size":(128,32),
            "beacon_size": (64, 64),
            "maxT":25
        },
        "block": {
            "ratio": 0.5,
            "batch_size": 48,
            "target_size": (64, 64),
            "beacon_size": (64, 64),
            "maxT": 15
        },
        "vert":{
            "batch_size":48,
            "ratio":-100,
            "target_size":(32,128),
            "beacon_size": (64, 64),
            "maxT": 30,
        },
    };

# there are no vertical samples
def get_hydra_v3_anchor_mjst():
    return {
        "profile_name":"hydra_6_2",
        "names":["long","short","block"],
        "long":{
            "ratio":6,
            "batch_size":20,
            "target_size":(256,32),
            "beacon_size":(64,64),
            "maxT": 40,
        },
        "short":{
            "ratio":2,
            "batch_size": 48,
            "target_size":(128,32),
            "beacon_size": (64, 64),
            "maxT":25
        },
        "block": {
            "ratio": -9999,
            "batch_size": 48,
            "target_size": (48, 64),
            "beacon_size": (64, 64),
            "maxT": 15
        },

    };

# there are no vertical samples

# there are vertical samples


def get_hydra_v3_anchor_1h1v_1_eval():
    return {
        "profile_name":"debug_1",
        "names":["long","vert"],
        "long":{
            "ratio":-100,
            "batch_size": 48,
            "target_size":(128,32),
            "beacon_size": (64, 64),
            "maxT":30
        },
        "vert":{
            "batch_size":24,
            "ratio":-100,
            "target_size":(32,256),
            "beacon_size": (64, 64),
            "maxT": 30,
        },
    };

