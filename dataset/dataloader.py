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


def create_train_test_sets(params):
    df = pd.read_csv(params['df_path'])
    labels = list(df.label.unique())

    train_tags, test_tags = train_test_split(labels, test_size=params['test_size'], random_state=params['seed'])

    train = df[df['label'].isin(train_tags)]
    test = df[df['label'].isin(test_tags)]

    train.reset_index(inplace=True)
    test.reset_index(inplace=True)

    print(f'Train set size: ', len(train))
    print(f'Test set size: {len(test)}')

    train = create_triplets_with_all_combinations(train, params)
    test = create_triplets_with_all_combinations(test, params)

    print(f'[Triplet] Train set size: ', len(train))
    print(f'[Triplet] Test set size: {len(test)}')

    with open(params['log_path'], 'a') as f:
        f.write(
            f'[Triplet] Train set size: {len(train)}; \n [Triplet] Test set size: {len(test)} \n')

    train.to_csv(params['train_path'], index=False)
    test.to_csv(params['test_path'], index=False)

    return train, test


def create_triplets_with_all_combinations(df, params):
    image_names = df['img_name']
    labels = df['label']
    index = df.index

    triplet = []
    for idx, name in enumerate(image_names):
        anchor_image = name
        anchor_label = labels[idx]

        positive_list = index[index == idx][labels[index == idx] == anchor_label]
        negative_list = index[index != idx][labels[index != idx] != anchor_label]

        if len(positive_list) > 0:
            for idxp in positive_list:
                positive_image = image_names.iloc[idxp]
                positive_label = labels.iloc[idxp]

                for idxn in negative_list:
                    negative_image = image_names.iloc[idxn]
                    negative_label = labels.iloc[idxn]

                    anchor_path = os.path.join(params['data_folder'], anchor_image)
                    positive_path = os.path.join(params['data_folder'], positive_image)
                    negative_path = os.path.join(params['data_folder'], negative_image)

                    triplet.append(
                        [anchor_image, anchor_label, positive_image, positive_label, negative_image, negative_label,
                         anchor_path, positive_path, negative_path])

    triplet = pd.DataFrame(triplet, columns=['anchor_image', 'anchor_label', 'positive_image', 'positive_label',
                                             'negative_image', 'negative_label', 'anchor_path', 'positive_path',
                                                             'negative_path'])

    return triplet


def create_train_val_test_sets(params):
    if params['generate_triplets']:
        train_folder, val_folder, test_folder = params['train_folder'], params['val_folder'], params['test_folder']
        train_dataset = create_triplets(train_folder, params)
        train_dataset.to_csv(params['train_path'], index=False)

        if params['angle_only_in_test']:
            val_dataset = create_triplets_test_and_val(val_folder, params)
            test_dataset = create_triplets_test_and_val(test_folder, params)

        else:
            val_dataset = create_triplets(val_folder, params)
            test_dataset = create_triplets(test_folder, params)

        val_dataset.to_csv(params['val_path'], index=False)
        test_dataset.to_csv(params['test_path'], index=False)
    else:
        train_dataset = pd.read_csv(params['train_path'])
        val_dataset = pd.read_csv(params['val_path'])
        test_dataset = pd.read_csv(params['test_path'])

    print(f'[Triplet] Train set size: ', len(train_dataset))
    print(f'[Triplet] Val set size: {len(val_dataset)}')
    print(f'[Triplet] Test set size: {len(test_dataset)}')

    with open(params['log_path'], 'a') as f:
        f.write(
            f'[Triplet] Train set size: {len(train_dataset)}; \n [Triplet] Val set size: {len(val_dataset)}; \n [Triplet] Test set size: {len(test_dataset)} \n')

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

            if len(positive_list) < params['n_triplet']:
                positive_list = list(np.random.choice(positive_list, size=params['n_triplet']))
            else:
                positive_list = list(np.random.choice(positive_list, size=params['n_triplet'], replace=False))

            tmp = os.listdir(folder)
            tmp.remove(tag_id)

            negative_list = []

            for neg_id in tmp:
                for neg_elem in os.listdir(folder + '/' + neg_id):
                    negative_list.append(os.path.join(folder, neg_id, neg_elem))

            negative_list = list(np.random.choice(negative_list, size=params['n_triplet'], replace=False))

            while len(positive_list) != 0 and len(negative_list) != 0:
                positive_image_path = np.random.choice(positive_list, size=1)[0]
                negative_image_path = np.random.choice(negative_list, size=1)[0]

                positive_list.remove(positive_image_path)
                negative_list.remove(negative_image_path)

                positive_image = positive_image_path.split('/')[-1]
                positive_label = positive_image_path.split('/')[-2]

                negative_image = negative_image_path.split('/')[-1]
                negative_label = negative_image_path.split('/')[-2]

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


def create_triplets_test_and_val(folder, params):
    unique_tags_id = os.listdir(folder)

    triplet = []
    for tag_id in unique_tags_id:
        elems = os.listdir(folder + '/' + tag_id)

        for elem in elems:
            if '0' in elem:
                anchor_image_path = os.path.join(folder, tag_id, elem)
                anchor_image = elem
                anchor_label = tag_id

                positive_list = [pos_elem for pos_elem in elems if '15' in pos_elem]

                for pos in positive_list:
                    positive_image_path = os.path.join(folder, tag_id, pos)
                    positive_image = pos
                    positive_label = tag_id

                    tmp = os.listdir(folder)
                    tmp.remove(tag_id)
                    tmp.remove('.DS_Store')

                    negative_list = []

                    for neg_id in tmp:
                        for neg_elem in os.listdir(folder + '/' + neg_id):
                            if '15' in neg_elem:
                                negative_list.append(os.path.join(folder, neg_id, neg_elem))

                    if len(negative_list) < params['n_triplet']:
                        negative_list = list(np.random.choice(negative_list, size=len(negative_list), replace=False))
                    else:
                        negative_list = list(np.random.choice(negative_list, size=params['n_triplet'], replace=False))

                    while len(negative_list) != 0:
                        negative_image_path = np.random.choice(negative_list, size=1)[0]

                        negative_list.remove(negative_image_path)

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
                                    transforms.Resize((params['dim1'], params['dim2']))])

    if 'df_path' in params:
        train, test = create_train_test_sets(params)
        train_dataset = TripletDatasetAll(train, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        test_dataset = TripletDatasetAll(test, is_train=False, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=params['batch_size'])

        return train_loader, None, test_loader

    else:
        train, val, test = create_train_val_test_sets(params)

        train_dataset = TripletDatasetAll(train, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        val_dataset = TripletDatasetAll(val, transform=transform)
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=True)
        test_dataset = TripletDatasetAll(test, is_train=False, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=params['batch_size'])
        return train_loader, val_loader, test_loader
