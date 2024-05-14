
# there are no vertical samples
def get_hydra_v3o_anchor_mjst():
    return {
        "profile_name":"hydra_5_05_o",
        "names":["long","short","block"],
        "long":{
            "ratio":5,
            "training_range":(2,9999),
            "batch_size":20,
            "target_size":(256,32),
            "beacon_size":(64,64),
            "maxT": 40,
        },
        "short":{
            "ratio":0.5,
            "training_range": (0.5, 6),
            "batch_size": 48,
            "target_size":(128,32),
            "beacon_size": (64, 64),
            "maxT":25
        },
        "block": {
            "ratio": -9999,
            "training_range":(-9999,2),
            "batch_size": 48,
            "target_size": (48, 64),
            "beacon_size": (64, 64),
            "maxT": 15
        },
    };
#here are no vertical samples
def get_hydra_v3o_anchor_multling():
    return {
        "profile_name":"hydra_5_05_o",
        "names":["long","short","vert"],
        "long":{
            "ratio":5,
            "training_range":(2,9999),
            "batch_size":24,
            "target_size":(256,32),
            "beacon_size":(64,64),
            "maxT": 40,
        },
        "short":{
            "ratio":0.5,
            "training_range": (0.5, 6),
            "batch_size": 48,
            "target_size":(128,32),
            "beacon_size": (64, 64),
            "maxT":25
        },
        "vert":{
            "training_range": (-9999, 1),
            "ratio": -9999,
            "batch_size": 24,
            "target_size": (32,256),
            "beacon_size": (64, 64),
            "maxT": 30
        }
    };
