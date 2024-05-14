def neko_get_arg(key,args,default=None):
    if(args is None):
        return default;
    if(key in args):
        return args[key];
    elif(default is not None):
        if (default == "NEP_skipped_NEP"):
            return None;
        else:
            return default;
    else:
        return args[key];
def neko_set_arg_if_not_already(key,args,value):
    if(key not in args):
        args[key]=value;

def neko_get_arg_dict(args,default_dict=None):
    ret_args={}
    for k in default_dict:
        if(k not in args):
            ret_args[k]=default_dict[k];
        else:
            ret_args[k]=default_dict[k];
    return args

def neko_get_defarg(key,args,pfx="_name"):
    return neko_get_arg(key+pfx,args,key);
