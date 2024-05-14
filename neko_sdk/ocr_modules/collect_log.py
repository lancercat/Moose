import os;

from neko_sdk.environment.copy import copylog


def bootstrap_folders(root,hosts):
    for k in hosts:
        os.makedirs(os.path.join(root,hosts[k]["name"]),exist_ok=True);


if __name__ == '__main__':
    from neko_2021_mjt.standardbench2_candidates.metas import METHOD_METAS, DEV_METAS;
    bootstrap_folders("/home/lasercat/ssddata",DEV_METAS)
    for d in DEV_METAS:
        for m in METHOD_METAS:
            copylog(DEV_METAS[d],"/home/lasercat/ssddata",m,"standardbench2_candidates")
            # copy_model(DEV_METAS[d],"/home/lasercat/ssddata",m,"standardbench2_candidates")
