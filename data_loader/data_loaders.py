from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data import Dataset
import pandas as pd
import torch
import numpy as np
import glob
import os

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)



class SpectrogramDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, csv_file, root_dir, batch_size, shuffle=True,
                 validation_split=0.0, num_workers=1, return_id=False):
        trsfm = transforms.Compose([
            transforms.ToTensor()
        ])
        self.data_dir = root_dir
        self.dataset = SimpleSpectrogramDataset(csv_file, self.data_dir,
                                          transform=trsfm, 
                                          return_id=return_id)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class RecurrenceDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, csv_file, root_dir, batch_size,
                shuffle=True,
                validation_split=0.0, 
                num_workers=1, 
                return_id=False,
                filter_type='rec_exists',
                target_type='rx_pose_carry'):
        trsfm = transforms.Compose([
            transforms.ToTensor()
        ])
        print('Inside dataloader')
        print(filter_type)
        print(target_type)
        self.data_dir = root_dir
        self.dataset = RecurrenceDataset(csv_file, self.data_dir,
                                          transform=trsfm, 
                                          return_id=return_id,
                                          filter_type=filter_type,
                                          target_type=target_type)
        super().__init__(self.dataset, batch_size,
                        shuffle, validation_split, 
                        num_workers)



class SpectrogramDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None, return_id=False):
        """
        Args:
            csv_file (string): Path to the csv file with distance values and simple statistics.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.table = pd.read_csv(csv_file)
        self.table = self.table[self.table['spec_exists']==True]
        self.root_dir = root_dir
        self.transform = transform
        self.return_id = return_id
    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        row = self.table.iloc[idx]
        # print([row['fileid'], row['chirp_number']])
        file_name = '{}_{}.npy'.format(row['fileid'], int(row['chirp_number']))
        file_name = os.path.join(self.root_dir, file_name)
        image = np.load(file_name)
        distance = row['distance_in_meters'].astype(np.float32)
        # sample = {'image': image, 'distance': distance}
        sample = [image,  distance]

        if self.transform:
            image = self.transform(image)

        sample = [image, distance] 
        if self.return_id:
            sample = [image, distance, row['fileid'], 
                      row['chirp_number']] 
        return sample


class SimpleSpectrogramDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None, return_id=False):
        """
        Args:
            csv_file (string): Path to the csv file with distance values and simple statistics.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        tx_extent = (7, 12)
        rx_extent = (-85, -32)
        self.table = pd.read_csv(csv_file)
        self.table = self.table[self.table['spec_exists']==True]
        self.root_dir = root_dir
        self.transform = transform
        self.return_id = return_id
        self.tx_extent = tx_extent
        self.rx_extent = rx_extent
    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        row = self.table.iloc[idx]
        # print([row['fileid'], row['chirp_number']])
#         file_name = '{}_{}.npy'.format(row['fileid'], int(row['chirp_number']))
#         file_name = os.path.join(self.root_dir, file_name)
#         image = np.load(file_name)


class RecurrenceDataset(SpectrogramDataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None,
                 filter_type='rec_multi_exists',
                 target_type='rx_pose_carry',
                 return_id=False,
                 data_limit=None):
        """
        Args:
            csv_file (string): Path to the csv file with distance values and simple statistics.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        print('types')
        print(filter_type)
        print(target_type)
        self.table = pd.read_csv(csv_file)
        self.table = self.table[self.table[filter_type]==True]
        self.root_dir = root_dir
        self.transform = transform
        self.return_id = return_id
        self.filter_type = filter_type
        self.target_type = target_type

        if data_limit:
            self.table = self.table.sample(data_limit)

    def get_target(self, row):
        pose_carry_dict = {'standing_pocket': 0,
                           'standing_hand': 1,
                           'sitting_pocket': 2,
                           'sitting_hand': 3}
        # print('sanityc hceck ', self.target_type)
        if self.target_type == 'distance_error':
            error = row['physical_distance'].astype(np.float32) - row['distance_in_meters'].astype(np.float32)
            return error
        return pose_carry_dict[row[self.target_type]]
    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        row = self.table.iloc[idx]
        # print([row['fileid'], row['chirp_number']])
        file_name = '{}_{}.npy'.format(row['fileid'], int(row['chirp_number']))
        file_name = os.path.join(self.root_dir, file_name)
        image = (np.load(file_name).astype(np.float32)) / 10.0
        target = self.get_target(row)
        # sample = {'image': image, 'distance': distance}
        sample = [image,  target]

        if self.transform:
            image = self.transform(image)

        sample = [image, target] 
        if self.return_id:
            sample = [image, target, row['fileid'], 
                      row['chirp_number']] 
        # print('inside dataset')
        # print(image.shape)
        # print(target)
        return sample


# class ErrorDataset(RecurrenceDataset):
#     def get_target(self, row):
#         if self.target_type == 'distance_error':
#             error = row['physical_distance'].astype(np.float32) - row['distance_in_meters'].astype(np.float32)
#         return error