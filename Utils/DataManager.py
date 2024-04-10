import nibabel as nib
import numpy as np
import pandas as pd
import torch

class DataManager:
    def __init__(self):
        self.img_w = 137
        self.img_h = 113
        self.img_d = 113
        self.imgs = self._populate_images()
        self.pt_500, self.pt_4000 = self._populate_PTS()

    def _populate_images(self):
        imgs = []
        for i in range(1,172):
            img = nib.load(f'./dataset/n171_smwp1/smwp10{i:03}_T1.nii').get_fdata()
            imgs.append(img)
        return np.array([np.array([img],dtype=np.float32) for img in imgs])

    def _populate_PTS(self):
        df = pd.read_excel('./dataset/PTs_500_4k_blinded.xlsx')
        pt500 = torch.tensor(df["PT500"].to_numpy(dtype=np.float32), dtype=torch.float32).reshape(-1, 1)
        pt4000 = torch.tensor(df["PT4000"].to_numpy(dtype=np.float32), dtype=torch.float32).reshape(-1, 1)
        return pt500, pt4000
