import nibabel as nib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import copy

class DataManager:
    def __init__(self):
        self.img_w = 137
        self.img_h = 113
        self.img_d = 113
        self.imgs = self._populate_images()
        self.pt_500, self.pt_500_a, self.pt_4000, self.pt_4000_a  = self._populate_PTS()

    def _populate_images(self):
        imgs = []
        for i in range(1,172):
            img = nib.load(f'./dataset/n171_smwp1/smwp10{i:03}_T1.nii').get_fdata()
            plt.show()
            imgs.append(img)
        imgArray = np.array([np.array([img],dtype=np.float32) for img in imgs])

        normalizedImgs = (imgArray-np.min(imgArray))/(np.max(imgArray)-np.min(imgArray))
        
        return normalizedImgs

    def _populate_PTS(self):
        df = pd.read_excel('./dataset/PTs_500_4k_blinded.xlsx')
        pt500 = df["PT500"].to_numpy(dtype=np.float32)
        pt4000 = df["PT4000"].to_numpy(dtype=np.float32)
        pt500t = torch.tensor(df["PT500"].to_numpy(dtype=np.float32), dtype=torch.float32).reshape(-1, 1)
        pt4000t = torch.tensor(df["PT4000"].to_numpy(dtype=np.float32), dtype=torch.float32).reshape(-1, 1)
        return pt500t, pt500, pt4000t, pt4000
    
    def get_img_spatial_features(self):
        height_step = self.img_h // 3
        width_step = self.img_w // 3
        depth_step = self.img_d // 3

        spatial_features = []
        block_sums = []
        for img in self.imgs:
            block_sums.clear()
            for y in range(0, 3):
                for x in range(0, 3):
                    for z in range(0, 3):
                        y_start = y*height_step
                        x_start = x*width_step
                        z_start = z*depth_step
                        block = img[:,y_start:y_start+height_step,x_start:x_start+width_step,z_start:z_start+depth_step]
                        block_sums.append(np.sum(block.flatten()))
            spatial_features.append(copy.deepcopy(block_sums))

        return np.array(spatial_features)


                        

