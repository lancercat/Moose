import pylcs


def img_test(img,runner,args):
    res=runner.test_img(0,img,args);
    return  res;

def dicted_test(img,runner,globalcache,lex):
    res=runner.test_img(0,img,
                globalcache);
    mind=9999;
    p=res;
    lns=[]
    with open(lex,"r") as fp:
        lns=[i.strip() for i in fp];
    lex=lns[0].split(",");
    iss=[];
    for i in lex:
        if(len(i)==0):
            continue;
        e=pylcs.edit_distance(res.lower(),i.lower())
        if(e<mind):
            mind=e;
            p=i
            iss.append(i)
    if(p==""):
        print("???")
    # if(p.lower()!=res.lower()):
    #     print(res,"->",p)
    return p

