import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

        plt.imshow(imgs[0][:,:,imgs[0].shape[2]//2])
        plt.show()
        plt.imshow(imgs[5][:,:,imgs[5].shape[2]//2])
        plt.show()

        return np.array([np.array([img],dtype=np.float32) for img in imgs])

    def _populate_PTS(self):
        df = pd.read_excel('./dataset/PTs_500_4k_blinded.xlsx')
        return df["PT500"].to_numpy(dtype=np.float32), df["PT4000"].to_numpy(dtype=np.float32)
    
    def get_pt_tensors(self, dataset=500):
        if dataset == 500:
            pts = self.pt_500
        else:
            pts = self.pt_4000

        tensors = []
        for pt in pts:
            #normal range
            if pt < 20:
                tensors.append(np.array([1,0,0,0,0,0],dtype=np.float32))
            #mild hearing loss
            elif pt < 40:
                tensors.append(np.array([0,1,0,0,0,0],dtype=np.float32))
            #moderate hearing loss
            elif pt < 55:
                tensors.append(np.array([0,0,1,0,0,0],dtype=np.float32))
            #mod-severe
            elif pt < 70:
                tensors.append(np.array([0,0,0,1,0,0],dtype=np.float32))
            #severe
            elif pt < 90:
                tensors.append(np.array([0,0,0,0,1,0],dtype=np.float32))
            #profound
            else:
                tensors.append(np.array([0,0,0,0,0,1],dtype=np.float32))

        return np.array(tensors)