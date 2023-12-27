import numpy as np
import os
from tqdm import tqdm

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _get_cifar_transforms(augment=False):
    transform_augment = transforms.Compose([
        transforms.Pad(padding=4, fill=(125,123,113)),
        transforms.RandomCrop(32, padding=0),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_train = transform_augment if augment else transform_test

    return transform_train, transform_test

def load_cifar10(batch_size):
    transform_train, transform_test = _get_cifar_transforms(augment=False)

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader

if __name__ == "__main__":
    npz_fn = 'estimates_results.npz'
    if os.path.exists(npz_fn):
        estimates = np.load(npz_fn)
    else:
        raise AttributeError("estimation file does not exist", npz_fn)


    loaded_results = np.load('estimates_results.npz')
    memorizations = loaded_results['memorization']
    influences = loaded_results['influence']

    N = len(memorizations)
    M = len(influences)

    # Filter train data points with values > 0.25
    valid_train_indices = np.where(memorizations > 0.25)[0]

    # Randomly select 10 train data points from the filtered indices
    selected_train_indices = np.random.choice(valid_train_indices, 10, replace=False)

    # For each selected train data point, find 10 test data points with the highest influence
    selected_test_indices = []
    for train_index in selected_train_indices:
        # Get influence scores for the current train data point
        scores = influences[:, train_index]

        # Find indices of top 10 test data points with highest influence
        top_test_indices = np.argsort(scores)[-10:]

        # Print train index, memorization score, and sorted list of selected test influences
        print(f"{train_index}, {memorizations[train_index]:.2f},", "\t", f"{', '.join(map(lambda x: f'{scores[x]:.2f}', top_test_indices))}")


        # Append the selected test indices to the list
        selected_test_indices.append(top_test_indices)

    # Convert the list to a NumPy array
    selected_test_indices = np.array(selected_test_indices)

    print (np.max(influences[:, 28845]), np.min(influences[:,28845]))