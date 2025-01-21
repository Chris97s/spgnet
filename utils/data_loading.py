import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils.read_write_pts import read_pts_file
import re

class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, lmks_dir : str ,scale: float = 1.0, eigen: list = [] ,mask_suffix: str = '_mask',lmk_suffix: str = '_lmk',shapes_id:list = []):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.lmks_dir = Path(lmks_dir)
        self.eigen = eigen
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.lmk_suffix = lmk_suffix
        self.shapes_id = shapes_id
        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)


    @staticmethod
    def preprocess(pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)  

        if not is_mask:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
                
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))   #第35行asarray会导致图片维度变化，所以需要转化回来

        img_ndarray = img_ndarray / 255
        return img_ndarray


    @staticmethod
    def load(filename):
        ext = splitext(filename)[1]
        if ext == '.npy':
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        elif ext == '.pts':
            return read_pts_file(filename)[0]
        else:
            return Image.open(filename)
        

    def __getitem__(self, idx):
        name = self.ids[idx]   #返回第idx个图片的name
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))   
        img_file = list(self.images_dir.glob(name + '.*'))
        lmk_file = list(self.lmks_dir.glob(name+self.lmk_suffix +'.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])
        lmk = self.load(lmk_file[0])
        lmk = np.asarray(lmk)
        
        
        height = img.size[1]
        width = img.size[0]
        img = img.resize((256, 256),Image.BILINEAR)
        mask = mask.resize((256, 256),Image.BILINEAR)
        
        #将lmk的坐标归一化
        lmk[:,0] = (1/width)*lmk[:,0]     #x轴  归一化
        lmk[:,1] = (1/height)*lmk[:,1]    #y轴  归一化
        
        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)

        
        pattern = r'\d+'
        matches = re.findall(pattern, name)
        number_part = ''.join(matches)
        number = int(number_part)
        
        return {
            'id':name,
            'id_number':number,
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous(),
            'lmk' : torch.as_tensor(lmk.copy()).float().contiguous(),
            'height': height,
            'width': width,
        }



