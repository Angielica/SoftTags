import os

import torch
from PIL import Image

import random
import pandas as pd

from torch.utils.data import DataLoader, Dataset

from sklearn.model_selection import train_test_split

class TripletDataset(Dataset):
    def __init__(self, df, image_folder, is_train=True, transform=None):
        self.df = df
        self.image_folder = image_folder
        self.is_train = is_train
        self.transform = transform

        self.image_names = self.df['image_names'].values
        self.labels = self.df['labels'].values
        self.index = self.df.index.values

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        anchor_label = self.labels.iloc[idx]

        anchor_image_path = os.path.join(self.image_folder, self.image_names.iloc[idx])
        anchor_image = Image.open(anchor_image_path).convert('RGB')

        if self.is_train:

            positive_list = self.index[self.index!=idx][self.labels[self.index!=idx]==anchor_label]

            if not positive_list:
                positive_list = self.index[self.index == idx][self.labels[self.index == idx] == anchor_label]

            positive_idx = random.choice(positive_list)
            positive_image_path = os.path.join(self.image_folder, self.image_names.iloc[positive_idx])
            positive_image = Image.open(positive_image_path).convert('RGB')

            negative_list = self.index[self.index!=idx][self.labels[self.index!=idx]!=anchor_label]
            negative_idx = random.choice(negative_list)
            negative_image_path = os.path.join(self.image_folder, self.image_names.iloc[negative_idx])
            negative_image = Image.open(negative_image_path).convert('RGB')

            if self.transform is not None:
                anchor_image = self.transform(anchor_image)
                positive_image = self.transform(positive_image)
                negative_image = self.transform(negative_image)

            return anchor_image, positive_image, negative_image, anchor_label

        else:

            other_list = self.index[self.index!=idx]
            other_idx = random.choice(other_list)
            other_image_path = os.path.join(self.image_folder, self.image_names.iloc[other_idx])
            other_image = Image.open(other_image_path).convert('RGB')

            other_label = self.labels.iloc[other_idx]

            if anchor_label == other_label:
                y = torch.ones(1)
            else:
                y = torch.zeros(1)


            if self.transform is not None:
                anchor_image = self.transform(anchor_image)
                other_image = self.transform(other_image)
            return anchor_image, other_image, y


def get_data(params):

    df = pd.read_csv(params['df_path'])

    train, test = train_test_split(df, test_size=params['test_size'], random_state=params['seed'])

    # Dividere in train, val and test?

    train_dataset = TripletDataset(train, params['image_folder'])
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)

    test_dataset = TripletDataset(test, params['image_folder'])
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'])
    return train_loader, test_loader


