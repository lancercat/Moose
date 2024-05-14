def update(opt):
    try:
        opt.step();
    except:
        print("Oops",opt);
        exit(9)
    return [];


def normgrad(mod):
    mod.normgrad();
    return [];