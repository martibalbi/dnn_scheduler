"""
TASA Datasets
"""
import torchvision
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset
import h5py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np

import ai8x

class TASADataset(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.file_path = 'database.h5'
        self.transform = transform
        self.target_transform = target_transform
        self.hdf5_file = h5py.File(self.file_path, 'r')
        self.train = train
        
        with h5py.File(self.file_path, 'r') as hdf5_file:
            self.data_raw = np.array(hdf5_file['inputs'])
            self.targets_raw = np.array(hdf5_file['outputs'])

        self.train_indices, self.test_indices = train_test_split(np.arange(len(self.data_raw)), test_size=0.2, random_state=42)

        self.data, self.targets = self._load_data()

    def _load_data(self):
        if self.train:
            train_indices_sorted = sorted(self.train_indices)[:1000000]
            data = self.data_raw[train_indices_sorted]
            targets = self.targets_raw[train_indices_sorted]
            
        else:
            test_indices_sorted = sorted(self.test_indices)[:200000]
            data = self.data_raw[test_indices_sorted]
            targets = self.targets_raw[test_indices_sorted]

        return data, targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_data = np.array(self.data[idx]).reshape(-1, 1)
        output_data = np.array(self.targets[idx]).reshape(-1, 1)

        input_scaler = MinMaxScaler(feature_range=(0, 1))
        input_scaler.fit(input_data)
        input_data = input_scaler.transform(input_data)

        output_scaler = MinMaxScaler(feature_range=(0, 1))
        output_scaler.fit(output_data)
        output_data = output_scaler.transform(output_data)

        input_data = torch.tensor(input_data, dtype=torch.float32)
        output_data = torch.tensor(output_data, dtype=torch.float32)

        if self.transform is not None:
            input_data = self.transform(input_data)
        
        if self.target_transform is not None:
            output_data = self.target_transform(output_data)

        input_data = torch.tensor(input_data, dtype=torch.float32).squeeze(1)
        output_data = torch.tensor(output_data, dtype=torch.float32).squeeze(1)

        return input_data, output_data


def tasa_get_datasets(data, load_train=True, load_test=True):
    """
    Load the TASA dataset.
    """

    (data_dir, args) = data

    if load_train:
        train_transform = transforms.Compose([
            ai8x.normalize(args=args)
        ])

        train_dataset = TASADataset(root=data_dir, train=True, transform=train_transform)

    else:
        train_dataset = None

    if load_test:
        test_transform = transforms.Compose([
            ai8x.normalize(args=args)
        ])

        test_dataset = TASADataset(root=data_dir, train=False, transform=test_transform)

        if args.truncate_testset:
            test_dataset.data = test_dataset.data[:1]
    else:
        test_dataset = None

    return train_dataset, test_dataset

datasets = [
    {
        'name': 'TASA',
        'input': (1, 12),
        'output': (1, 420),
        'loader': tasa_get_datasets,
    },
]
