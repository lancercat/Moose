import cv2
import numpy as np
def show_tims( tim, name, var=255,mean=0,timeout=0):
    ims = (tim*var+mean).detach().cpu().numpy();
    ims = ims.transpose([0, 2, 3, 1]);
    ims = ims.astype(np.uint8);
    iml=np.split(ims,ims.shape[0],axis=0);
    iml=[i[0] for i in iml];
    ims=np.concatenate(iml);
    # for i in range(len(ims)):
    cv2.imshow(name , ims);
    cv2.waitKey(timeout);