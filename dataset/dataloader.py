from PIL import Image
import os
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


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
    def __init__(self, df, image_folder, is_train=True, transform=None, thresh=.5):
        self.df = df
        self.image_folder = image_folder
        self.is_train = is_train
        self.transform = transform
        self.thresh = thresh

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = self.df.iloc[idx]

        anchor_label = data['anchor_label']
        anchor_image_path = os.path.join(self.image_folder, data['anchor_image'])
        anchor_image = Image.open(anchor_image_path).convert('RGB')

        positive_image_path = os.path.join(self.image_folder, data['positive_image'])
        positive_image = Image.open(positive_image_path).convert('RGB')

        negative_image_path = os.path.join(self.image_folder, data['negative_image'])
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

    train = create_triplets(train)
    test = create_triplets(test)

    print(f'[Triplet] Train set size: ', len(train))
    print(f'[Triplet] Test set size: {len(test)}')

    return train, test


def create_triplets(df):
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

                    triplet.append(
                        [anchor_image, anchor_label, positive_image, positive_label, negative_image, negative_label])

    triplet = pd.DataFrame(triplet, columns=['anchor_image', 'anchor_label', 'positive_image', 'positive_label',
                                             'negative_image', 'negative_label'])

    return triplet


def get_data(params):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((params['img_size'], params['img_size']))])

    train, test = create_train_test_sets(params)
    # train = pd.read_csv(params['df_path_train'])
    # test = pd.read_csv(params['df_path_test'])

    print(train.shape)
    print(test.shape)

    train_dataset = TripletDatasetAll(train, params['image_folder'], transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)

    test_dataset = TripletDatasetAll(test, params['image_folder'], is_train=False,
                                     transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'])
    return train_loader, test_loader
