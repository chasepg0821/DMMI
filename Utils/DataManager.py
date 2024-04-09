import nibabel as nib
import numpy as np
import pandas as pd

class DataManager:
    def __init__(self, path_to_dataset="./dataset", width=137, height=113, depth=113, num_images=171):
        self.img_w = width
        self.img_h = height
        self.img_d = depth
        self.imgs, self.pt_500, self.pt_4000 = self._get_data(path_to_dataset, num_images)
        print(self.pt_500,self.pt_4000)

    def _get_data(self, path_to_dataset, num_images):
        imgs = []
        for i in range(1,num_images+1):
            img = nib.load(f'{path_to_dataset}/n171_smwp1/smwp10{i:03}_T1.nii').get_fdata()
            imgs.append(img)

        df = pd.read_excel(f'{path_to_dataset}/PTs_500_4k_blinded.xlsx')
        
        return np.array(imgs), df["PT500"].to_numpy(), df["PT4000"].to_numpy()
