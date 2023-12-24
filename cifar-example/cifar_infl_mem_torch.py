import numpy as np
import os
from tqdm import tqdm

import matplotlib.pyplot as plt
import torchvision.transforms as transforms

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def one_hot(x, k):
    return torch.eye(k)[x]

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class SubsetSampler(SubsetRandomSampler):
    def __init__(self, indices):
        super(SubsetSampler, self).__init__(indices)

def loss(output, target):
    return -torch.mean(torch.sum(output * target, dim=1))

def batch_correctness(model, data_loader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return correct / total


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

def train_model(model, train_loader, num_epochs, criterion, optimizer, scheduler, device):
    model.train()
    for epoch in tqdm(range(num_epochs)):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()

def subset_train(seed, subset_ratio, batch_size, num_epochs):
    torch.manual_seed(seed)
    np.random.seed(seed)


    train_loader, test_loader = load_cifar10(batch_size)

    model = models.resnet18()
    model.fc = nn.Linear(512, 10)  # CIFAR-100 has 100 classes

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=5e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    num_train_total = len(train_loader.dataset)
    num_train = int(num_train_total * subset_ratio)
    num_batches = int(np.ceil(num_train / batch_size))

    subset_sampler = SubsetSampler(np.random.choice(num_train_total, size=num_train, replace=False))
    train_loader = DataLoader(train_loader.dataset, batch_size=batch_size, sampler=subset_sampler)

    train_model(model, train_loader, num_epochs, criterion, optimizer, scheduler, device)
    model.eval()

    trainset_correctness = batch_correctness(model, train_loader, device)
    testset_correctness = batch_correctness(model, test_loader, device)

    trainset_mask = np.zeros(num_train_total, dtype=np.bool)
    trainset_mask[subset_sampler.indices] = True

    # Compute accuracy on the selected subsample
    selected_subset_correctness = batch_correctness(model, train_loader, device)

    # Create a DataLoader for the left-out subsample
    left_out_indices = np.setdiff1d(np.arange(num_train_total), subset_sampler.indices)
    left_out_sampler = SubsetSampler(left_out_indices)
    left_out_loader = DataLoader(train_loader.dataset, batch_size=batch_size, sampler=left_out_sampler)

    # Compute accuracy on the left-out subsample
    left_out_subset_correctness = batch_correctness(model, left_out_loader, device)

    # Compute accuracy on the test data
    testset_correctness = batch_correctness(model, test_loader, device)

    # Print accuracies
    print(f"Selected Subset Train Accuracy: {selected_subset_correctness:.4f}")
    print(f"Left-Out Subset Train Accuracy: {left_out_subset_correctness:.4f}")
    print(f"Test Accuracy: {testset_correctness:.4f}")



    return trainset_mask, trainset_correctness, testset_correctness

def estimate_infl_mem(num_runs, subset_ratio, batch_size, num_epochs):
    results = []

    for i_run in range(num_runs):
        results.append(subset_train(i_run, subset_ratio, batch_size, num_epochs))

    trainset_mask = np.vstack([ret[0] for ret in results])
    inv_mask = np.logical_not(trainset_mask)
    trainset_correctness = np.vstack([ret[1] for ret in results])
    testset_correctness = np.vstack([ret[2] for ret in results])

    print(f'Avg test acc = {np.mean(testset_correctness):.4f}')

    def _masked_avg(x, mask, axis=0, esp=1e-10):
        return (np.sum(x * mask, axis=axis) / np.maximum(np.sum(mask, axis=axis), esp)).astype(np.float32)

    def _masked_dot(x, mask, esp=1e-10):
        x = x.T.astype(np.float32)
        return (np.matmul(x, mask) / np.maximum(np.sum(mask, axis=0, keepdims=True), esp)).astype(np.float32)

    mem_est = _masked_avg(trainset_correctness, trainset_mask) - _masked_avg(trainset_correctness, inv_mask)
    infl_est = _masked_dot(testset_correctness, trainset_mask) - _masked_dot(testset_correctness, inv_mask)

    return dict(memorization=mem_est, influence=infl_est)

def show_cifar100_examples(estimates, n_show=10):
    def show_tensor_images(images, title, nrow=10, cmin=0, cmax=1, figsize=(15, 3)):
        images = make_grid(images, nrow=nrow, normalize=True, scale_each=True)
        plt.figure(figsize=figsize)
        plt.imshow(images.permute(1, 2, 0).clamp(cmin, cmax))
        plt.title(title)
        plt.axis('off')
        plt.show()

    n_context1 = 4
    n_context2 = 5

    idx_sorted = np.argsort(np.max(estimates['influence'], axis=1))[::-1]
    for i in range(n_show):
        idx_tt = idx_sorted[i]
        label_tt = test_loader.dataset.targets[idx_tt]
        show_tensor_images(test_loader.dataset.data[idx_tt], f'Test, Label={label_tt}')

        def _show_contexts(idx_list, title, ax_offset):
            images = [train_loader.dataset.data[idx] for idx in idx_list]
            show_tensor_images(images, title, nrow=len(images), figsize=(15, 3), ax_offset=ax_offset)

        idx_sorted_tr = np.argsort(estimates['influence'][idx_tt])[::-1]
        _show_contexts(idx_sorted_tr[:n_context1], 'Train, High Influence', n_context1 + 1)

        idx_class = np.nonzero(np.array(train_loader.dataset.targets) == label_tt)[0]
        idx_random = np.random.choice(idx_class, size=n_context2, replace=False)
        _show_contexts(idx_random, 'Train, Random Class', n_context1 + n_context2 + 1)


if __name__ == '__main__':
    num_runs = 1
    subset_ratio = 0.7
    batch_size = 128
    num_epochs = 30

    estimates = estimate_infl_mem(num_runs, subset_ratio, batch_size, num_epochs)
    # You can use the estimates dictionary for further analysis or visualization

    np.savez('estimates_results.npz', **estimates)

    #show_examples(estimates)

    #loaded_results = np.load('estimates_results.npz')
    #loaded_memorization = loaded_results['memorization']
    #loaded_influence = loaded_results['influence']