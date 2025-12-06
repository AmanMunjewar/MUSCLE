import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from datasets.HAM10000.augmentation import train_transform, val_transform, center_crop

os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

base_dir = r'/home/datasets/HAM10000'

class HAM10000_DataSets(Dataset):
    def __init__(
            self,
            base_dir=base_dir,
            split='train',
            img_size=(256, 256),
            seed=1234
    ):
        super(HAM10000_DataSets, self).__init__()

        self.base_dir = base_dir
        self.split = split
        self.img_size = img_size
        self.metadata_path = os.path.join(self.base_dir, 'HAM10000_metadata.csv')

        # Define label mapping
        self.label_map = {'nv': 0, 'df': 1, 'bkl': 2, 'vasc': 3, 'akiec': 4, 'bcc': 5, 'mel': 6}

        if split == 'train':
            self.transform = train_transform(img_size=img_size)
        else:
            self.transform = val_transform(img_size=img_size)

        self.df = self._load_and_split_data(seed)
        print(f'split: {self.split}, total {len(self.df)} samples')

    def _load_and_split_data(self, seed):
        if not os.path.exists(self.metadata_path):
             # For local testing if path doesn't exist, we might want to fail or mock.
             # But per instructions, we assume the path exists.
             # If it fails, it fails.
             pass

        df = pd.read_csv(self.metadata_path)

        # Shuffle with seed
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

        n = len(df)
        train_end = int(n * 0.8)
        val_end = int(n * 0.9)

        if self.split == 'train':
            return df.iloc[:train_end]
        elif self.split == 'val':
            return df.iloc[train_end:val_end]
        elif self.split == 'test':
            return df.iloc[val_end:]
        else:
            raise ValueError(f"Unknown split: {self.split}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row['image_id']
        dx = row['dx']

        # Determine image path (check both folders)
        img_path = os.path.join(self.base_dir, 'HAM10000_images_part_1', image_id + '.jpg')
        if not os.path.exists(img_path):
            img_path = os.path.join(self.base_dir, 'HAM10000_images_part_2', image_id + '.jpg')
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image {image_id}.jpg not found in part_1 or part_2")

        cls_label = self.label_map[dx]

        image = cv2.imread(img_path)
        if image is None:
             raise ValueError(f"Failed to load image: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_shape = image.shape
        crop_size = image_shape[0] if image_shape[0] <= image_shape[1] else image_shape[1]  # min edge

        image = center_crop(image, crop_size=[crop_size, crop_size])

        transform = self.transform(image=image)
        image = transform['image']

        return {'image': image.float(),
                'cls_label': cls_label,
                'image_name': image_id}
