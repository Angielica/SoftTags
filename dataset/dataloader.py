from PIL import Image
import os
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import numpy as np


class TripletDataset(Dataset):
    def __init__(self, df, image_folder, is_train=True, transform=None):
        self.df = df
        self.image_folder = image_folder
        self.is_train = is_train
        self.transform = transform

        self.image_names = self.df['img_name']
        self.labels = self.df['label']
        self.index = self.df.index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        anchor_label = self.labels[self.index[idx]]
        anchor_image_path = os.path.join(self.image_folder, self.image_names[self.index[idx]])
        anchor_image = Image.open(anchor_image_path).convert('RGB')

        if self.is_train:
            positive_list = self.index[self.index != idx][self.labels[self.index != idx] == anchor_label]

            if positive_list.empty:
                positive_idx = idx
            else:
                positive_idx = random.choice(positive_list)

            positive_image_path = os.path.join(self.image_folder,
                                               self.image_names[self.index == positive_idx].values[0])
            positive_image = Image.open(positive_image_path).convert('RGB')

            negative_list = self.index[self.index != idx][self.labels[self.index != idx] != anchor_label]
            negative_idx = random.choice(negative_list)
            negative_image_path = os.path.join(self.image_folder,
                                               self.image_names[self.index == negative_idx].values[0])
            negative_image = Image.open(negative_image_path).convert('RGB')

            if self.transform is not None:
                anchor_image = self.transform(anchor_image)
                positive_image = self.transform(positive_image)
                negative_image = self.transform(negative_image)

            return anchor_image, positive_image, negative_image, anchor_label
        else:
            other_list = self.index[self.index != idx]
            other_idx = random.choice(other_list)
            other_image_path = os.path.join(self.image_folder, self.image_names[self.index == other_idx].values[0])
            other_image = Image.open(other_image_path).convert('L')

            other_label = self.labels[self.index == other_idx].values[0]

            if anchor_label == other_label:
                y = torch.ones(1)
            else:
                y = torch.zeros(1)

            if self.transform is not None:
                anchor_image = self.transform(anchor_image)
                other_image = self.transform(other_image)

            return anchor_image, other_image, y


class TripletDatasetAll(Dataset):
    def __init__(self, df, is_train=True, transform=None, thresh=.5):
        self.df = df
        self.is_train = is_train
        self.transform = transform
        self.thresh = thresh

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = self.df.iloc[idx]

        anchor_label = data['anchor_label']
        anchor_image_path = data['anchor_path']
        anchor_image = Image.open(anchor_image_path).convert('RGB')

        positive_image_path = data['positive_path']
        positive_image = Image.open(positive_image_path).convert('RGB')

        negative_image_path = data['negative_path']
        negative_image = Image.open(negative_image_path).convert('RGB')

        if self.transform is not None:
            anchor_image = self.transform(anchor_image)
            positive_image = self.transform(positive_image)
            negative_image = self.transform(negative_image)

        if self.is_train:
            return anchor_image, positive_image, negative_image, anchor_label

        else:
            '''
            p = torch.rand(1)

            if p > self.thresh:
                return anchor_image, positive_image, torch.ones(1)
            else:
                return anchor_image, negative_image, torch.zeros(1)
            '''
            return anchor_image, positive_image, torch.ones(1), negative_image, torch.zeros(1)


def create_train_val_test_sets(params):
    if params['generate_triplets']:
        train_folder, val_folder, test_folder = params['train_folder'], params['val_folder'], params['test_folder']
        train_dataset = create_triplets(train_folder, params)
        train_dataset.to_csv(params['train_path'], index=False)

        val_dataset = create_triplets(val_folder, params)
        val_dataset.to_csv(params['val_path'], index=False)

        test_dataset = create_triplets(test_folder, params)
        test_dataset.to_csv(params['test_path'], index=False)
    else:
        train_dataset = pd.read_csv(params['train_path'])
        val_dataset = pd.read_csv(params['val_path'])
        test_dataset = pd.read_csv(params['test_path'])

    print(f'[Triplet] Train set size: ', len(train_dataset))
    print(f'[Triplet] Val set size: {len(val_dataset)}')
    print(f'[Triplet] Test set size: {len(test_dataset)}')

    return train_dataset, val_dataset, test_dataset


def create_triplets(folder, params):
    unique_tags_id = os.listdir(folder)

    triplet = []
    for tag_id in unique_tags_id:
        for elem in os.listdir(folder + '/' + tag_id):
            anchor_image_path = os.path.join(folder, tag_id, elem)
            anchor_image = elem
            anchor_label = tag_id

            # Create k triplets
            np.random.seed(params['seed'])

            positive_list = []

            for pos_elem in os.listdir(folder + '/' + tag_id):
                positive_list.append(os.path.join(folder, tag_id, pos_elem))

            positive_list = np.random.choice(positive_list, size=params['n_triples'], replace=False)

            tmp = os.listdir(folder)
            tmp.remove(tag_id)

            negative_list = []

            for neg_id in tmp:
                for neg_elem in os.listdir(folder + '/' + neg_id):
                    negative_list.append(os.path.join(folder, neg_id, neg_elem))

            negative_list = np.random.choice(negative_list, size=params['n_triples'], replace=False)

            while len(positive_list) != 0 and len(negative_list) != 0:
                positive_image_path = np.random.choice(positive_list, size=1)[0]
                negative_image_path = np.random.choice(negative_list, size=1)[0]

                positive_list.remove(positive_image_path)
                negative_list.remove(negative_image_path)

                positive_image = positive_image_path.split('/')[8]
                positive_label = positive_image_path.split('/')[7]

                negative_image = negative_image_path.split('/')[8]
                negative_label = negative_image_path.split('/')[7]

                triplet.append([anchor_image, anchor_label,
                                positive_image, positive_label,
                                negative_image, negative_label,
                                anchor_image_path, positive_image_path, negative_image_path])

    triplet = pd.DataFrame(triplet, columns=['anchor_image', 'anchor_label',
                                             'positive_image', 'positive_label',
                                             'negative_image', 'negative_label',
                                             'anchor_path', 'positive_path',
                                             'negative_path'])
    return triplet


def get_data(params):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((params['img_size'], params['img_size']))])

    train, val, test = create_train_val_test_sets(params)

    train_dataset = TripletDatasetAll(train, params['image_folder'], transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)

    val_dataset = TripletDatasetAll(val, params['image_folder'], transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=True)

    test_dataset = TripletDatasetAll(test, params['image_folder'], is_train=False,
                                     transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'])
    return train_loader, test_loader
