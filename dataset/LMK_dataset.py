from pathlib import Path
from os.path import splitext
from os import listdir
import cv2
from genLmks import genLandmarks
from read_write_pts import write_pts_file

masks_dir = Path("./train/masks")
lmks_dir  = Path("./train/lmks")

for file in listdir(masks_dir):
    ids = splitext(file)[0].replace('_mask', '')

    mask_file = list(masks_dir.glob(ids + "_mask" + '.*')) 
    print(mask_file[0])
    
    landmarks = genLandmarks(str(mask_file[0]),128)
    mask = cv2.imread(str(mask_file[0]),1)
    width = mask.shape[1]
    height = mask.shape[0]
    write_pts_file(str(lmks_dir)+"/"+ids+"_lmk.pts", landmarks,width,height)
