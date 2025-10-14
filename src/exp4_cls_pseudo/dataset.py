import cv2
import torch
import random
import numpy as np
import pandas as pd
import albumentations as albu
from torch.utils.data import Dataset

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

classes = [
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery',
    'Right Middle Cerebral Artery',
    'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation',
    'Aneurysm Present'
]

flip_classes = [
    'Right Infraclinoid Internal Carotid Artery',
    'Left Infraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Middle Cerebral Artery',
    'Left Middle Cerebral Artery',
    'Anterior Communicating Artery',
    'Right Anterior Cerebral Artery',
    'Left Anterior Cerebral Artery',
    'Right Posterior Communicating Artery',
    'Left Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation',
    'Aneurysm Present'
]

class AneurysmDataset(Dataset):
    def __init__(self, df=None, brain_box_df=None, pseudo_df=None, pseudo_brain_box_df=None, noise_df=None, image_size=None, mode=None, crop_ratio=1):
        self.df = df.reset_index(drop=True)
        
        self.image_size = image_size
        assert mode in  ['train', 'valid']
        self.mode = mode
        self.crop_ratio = crop_ratio

        self.brain_box_dict = {}
        for _, row in brain_box_df.iterrows():
            box = row['brain_box'].split(' ')
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            self.brain_box_dict[row['SeriesInstanceUID']] = [x1, y1, x2, y2]

        if self.mode == 'train':
            for _, row in pseudo_brain_box_df.iterrows():
                box = row['brain_box'].split(' ')
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                self.brain_box_dict[row['SeriesInstanceUID']] = [x1, y1, x2, y2]

            noise_df = noise_df.copy().loc[noise_df.SeriesInstanceUID.isin(df.SeriesInstanceUID.values.tolist())]
            neg_df = self.df.loc[self.df['Aneurysm Present'] == 0]
            neg_df = neg_df.loc[~neg_df.SeriesInstanceUID.isin(noise_df.SeriesInstanceUID.values.tolist())]
            pos_df = self.df.loc[(self.df['Aneurysm Present'] == 1)&(self.df['has_box'] == 1)]

            self.df = pd.concat([neg_df, pos_df, noise_df, pseudo_df], ignore_index=True)
            self.df = self.df.sample(frac=1).reset_index(drop=True)

            self.transform = albu.Compose([
                albu.RandomResizedCrop(size=(self.image_size, self.image_size), scale=(0.5, 1.0), ratio=(0.75, 1.3333), p=1),
                albu.ShiftScaleRotate(rotate_limit=15, border_mode=0, p=0.5),
                albu.OneOf([
                    albu.MotionBlur(blur_limit=5),
                    albu.MedianBlur(blur_limit=5),
                    albu.GaussianBlur(blur_limit=5),
                    albu.GaussNoise(var_limit=(5.0, 30.0)),
                ], p=0.5),
                albu.CLAHE(clip_limit=4.0, p=0.5),
                albu.HueSaturationValue(p=0.5),
                albu.RandomBrightnessContrast(p=0.5),
            ])
        else:
            self.transform = albu.Compose([
                albu.Resize(self.image_size, self.image_size),
            ])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_path = row['image_path']
        image = cv2.imread(image_path)
        brain_x1, brain_y1, brain_x2, brain_y2 = self.brain_box_dict[row['SeriesInstanceUID']]
        image = image[brain_y1:brain_y2, brain_x1:brain_x2, :]
        label = row[classes].values.tolist()
        if self.mode == 'train':
            if random.random() > 0.5:
                image = cv2.flip(image, 1)
                label = row[flip_classes].values.tolist()
        else:
            if self.crop_ratio != 1:
                height, width = image.shape[0:2]
                xc = width/2
                yc = height/2
                x1 = int(xc-0.5*self.crop_ratio*width)
                x2 = int(xc+0.5*self.crop_ratio*width)
                y1 = int(yc-0.5*self.crop_ratio*height)
                y2 = int(yc+0.5*self.crop_ratio*height)
                image = image[y1:y2,x1:x2,:]
                
        image = self.transform(image=image)['image']
        image = image.astype(np.float32)
        image /= 255
        image = np.transpose(image, (2,0,1))
        image = torch.from_numpy(image)
        label = torch.FloatTensor(label)
        return image, label, row['image_path']
