import albumentations as A
import pandas as pd
import numpy as np
from PIL import Image
import joblib
import torch
from torch.utils.data import Dataset

class BengaliTrainDataset(Dataset):
    def __init__(self, dataframe, folds, transforms = None):
        super().__init__()

        self.df = dataframe[dataframe.kfold.isin(folds)].reset_index(drop = True)
        df = dataframe[["image_id", "grapheme_root", "vowel_diacritic", "consonant_diacritic", "kfold"]]
        self.transforms = transforms

        df = df[df.kfold.isin(folds)].reset_index(drop = True)
        self.image_ids = df.image_id.values
        self.grapheme_root = df.grapheme_root.values
        self.vowel_diacritic = df.vowel_diacritic.values
        self.consonant_diacritic = df.consonant_diacritic.values

    def __getitem__(self, index: int):
        image = joblib.load(f"input/image_pickles/{self.image_ids[index]}.pkl")
        image = image.reshape(137,236).astype(float)
        image = Image.fromarray(image).convert("RGB")
        image = np.array(image)
        

        if self.transforms:
            sample = {
              'image': image
            }
            sample = self.transforms(**sample)
            image = sample['image']
        
        image = np.transpose(image, (2,0,1)).astype(np.float32)

        return {
            'image': torch.as_tensor(image, dtype = torch.float),
            'grapheme_root': torch.as_tensor(self.grapheme_root[index], dtype = torch.long),
            "vowel_diacritic": torch.as_tensor(self.vowel_diacritic[index], dtype = torch.long),
            "consonant_diacritic": torch.as_tensor(self.consonant_diacritic[index], dtype = torch.long)
        }



    def __len__(self):
        return len(self.image_ids)
        