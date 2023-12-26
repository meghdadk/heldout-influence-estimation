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

def adjust_learning_rate(epoch, lr_dict, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(lr_dict['lr_decay_epochs']))
    new_lr = lr_dict['learning_rate']
    if steps > 0:
        new_lr = lr_dict['learning_rate'] * (lr_dict['lr_decay_rate'] ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    return new_lr

def batch_correctness(model, data_loader, device):
    correctness_list = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correctness_list.append(predicted == targets)

    return torch.cat(correctness_list).detach().cpu().numpy()


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

def train_model(model, train_loader, device):
    num_epochs = 30
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    lr_dict = {'learning_rate': 0.01, 'lr_decay_epochs': [15,20,25], 'lr_decay_rate': 0.1}
    model.train()
    for epoch in tqdm(range(num_epochs)):
        adjust_learning_rate(epoch, lr_dict, optimizer)
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    
def subset_load(seed, subset_ratio, batch_size, save_dir="checkpoints"):
    torch.manual_seed(seed)
    np.random.seed(seed)


    train_loader, test_loader = load_cifar10(batch_size)

    model = models.resnet18()
    model.fc = nn.Linear(512, 10)  # CIFAR-100 has 100 classes
    model.to(device)

    #scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    num_train_total = len(train_loader.dataset)
    num_train = int(num_train_total * subset_ratio)
    num_batches = int(np.ceil(num_train / batch_size))

    subset_sampler = SubsetSampler(np.random.choice(num_train_total, size=num_train, replace=False))
    sub_train_loader = DataLoader(train_loader.dataset, batch_size=batch_size, sampler=subset_sampler)


    chkpt_path = os.path.join(save_dir, 'resnet18_cifar10_model{}.pt'.format(seed))
    model.load_state_dict(torch.load(chkpt_path))

    model.eval()


    trainset_correctness = batch_correctness(model, train_loader, device)
    testset_correctness = batch_correctness(model, test_loader, device)

    trainset_mask = np.zeros(num_train_total, dtype=np.bool)
    trainset_mask[subset_sampler.indices] = True

    # Compute accuracy on the selected subsample
    selected_subset_correctness = batch_correctness(model, sub_train_loader, device)

    # Create a DataLoader for the left-out subsample
    left_out_indices = np.setdiff1d(np.arange(num_train_total), subset_sampler.indices)
    left_out_sampler = SubsetSampler(left_out_indices)
    left_out_loader = DataLoader(train_loader.dataset, batch_size=batch_size, sampler=left_out_sampler)

    # Compute accuracy on the left-out subsample
    left_out_subset_correctness = batch_correctness(model, left_out_loader, device)


    # Print accuracies
    print(f"Selected Subset Train Accuracy: {np.mean(selected_subset_correctness):.4f}")
    print(f"Left-Out Subset Train Accuracy: {np.mean(left_out_subset_correctness):.4f}")
    print(f"Test Accuracy: {np.mean(testset_correctness):.4f}")


    return trainset_mask, trainset_correctness, testset_correctness

def subset_train(seed, subset_ratio, batch_size, save_dir="checkpoints", load_trained=True):
    torch.manual_seed(seed)
    np.random.seed(seed)


    train_loader, test_loader = load_cifar10(batch_size)

    model = models.resnet18()
    model.fc = nn.Linear(512, 10)  # CIFAR-100 has 100 classes
    model.to(device)

    #scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    num_train_total = len(train_loader.dataset)
    num_train = int(num_train_total * subset_ratio)
    num_batches = int(np.ceil(num_train / batch_size))

    subset_sampler = SubsetSampler(np.random.choice(num_train_total, size=num_train, replace=False))
    sub_train_loader = DataLoader(train_loader.dataset, batch_size=batch_size, sampler=subset_sampler)


    train_model(model, sub_train_loader, device)
    model.eval()


    trainset_correctness = batch_correctness(model, train_loader, device)
    testset_correctness = batch_correctness(model, test_loader, device)

    trainset_mask = np.zeros(num_train_total, dtype=np.bool)
    trainset_mask[subset_sampler.indices] = True

    # Compute accuracy on the selected subsample
    selected_subset_correctness = batch_correctness(model, sub_train_loader, device)

    # Create a DataLoader for the left-out subsample
    left_out_indices = np.setdiff1d(np.arange(num_train_total), subset_sampler.indices)
    left_out_sampler = SubsetSampler(left_out_indices)
    left_out_loader = DataLoader(train_loader.dataset, batch_size=batch_size, sampler=left_out_sampler)

    # Compute accuracy on the left-out subsample
    left_out_subset_correctness = batch_correctness(model, left_out_loader, device)


    # Print accuracies
    print(f"Selected Subset Train Accuracy: {np.mean(selected_subset_correctness):.4f}")
    print(f"Left-Out Subset Train Accuracy: {np.mean(left_out_subset_correctness):.4f}")
    print(f"Test Accuracy: {np.mean(testset_correctness):.4f}")


    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    chkpt_path = os.path.join(save_dir, 'resnet18_cifar10_model{}.pt'.format(seed))
    torch.save(model.state_dict(), chkpt_path)

    return trainset_mask, trainset_correctness, testset_correctness

def estimate_infl_mem(num_runs, subset_ratio, batch_size):
    results = []

    for i_run in range(0, 4, 1):
        #results.append(subset_train(i_run, subset_ratio, batch_size))
        results.append(subset_load(i_run, subset_ratio, batch_size))

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

    print(mem_est.shape, infl_est.shape)

    return dict(memorization=mem_est, influence=infl_est)

def show_examples(estimates, n_show=10):
    def show_image(ax, image, title=None):
        image = F.to_pil_image(image)
        ax.axis('off')
        ax.imshow(image)
        if title is not None:
            ax.set_title(title, fontsize='x-small')

    n_context1 = 4
    n_context2 = 5

    fig, axs = plt.subplots(nrows=n_show, ncols=n_context1 + n_context2 + 1,
                            figsize=(n_context1 + n_context2 + 1, n_show))
    idx_sorted = np.argsort(np.max(estimates['influence'], axis=1))[::-1]
    for i in range(n_show):
        # show test example
        idx_tt = idx_sorted[i]
        label_tt = mnist_data['test_int_labels'][idx_tt]
        show_image(axs[i, 0], mnist_data['test_byte_images'][idx_tt],
                   title=f'test, L={label_tt}')

        def _show_contexts(idx_list, ax_offset):
            for j, idx_tr in enumerate(idx_list):
                label_tr = mnist_data['train_int_labels'][idx_tr]
                infl = estimates['influence'][idx_tt, idx_tr]
                show_image(axs[i, j + ax_offset], mnist_data['train_byte_images'][idx_tr],
                           title=f'tr, L={label_tr}, infl={infl:.3f}')

        # show training examples with highest influence
        idx_sorted_tr = np.argsort(estimates['influence'][idx_tt])[::-1]
        _show_contexts(idx_sorted_tr[:n_context1], 1)

        # show random training examples from the same class
        idx_class = np.nonzero(mnist_data['train_int_labels'] == label_tt)[0]
        idx_random = np.random.choice(idx_class, size=n_context2, replace=False)
        _show_contexts(idx_random, n_context1 + 1)

    plt.tight_layout()
    plt.savefig("cifar10-examples.png")
    plt.show()

if __name__ == '__main__':
    num_runs = 2
    subset_ratio = 0.7
    batch_size = 128

    estimates = estimate_infl_mem(num_runs, subset_ratio, batch_size)
    # You can use the estimates dictionary for further analysis or visualization

    np.savez('estimates_results.npz', **estimates)

    #show_examples(estimates)

    #loaded_results = np.load('estimates_results.npz')
    #loaded_memorization = loaded_results['memorization']
    #loaded_influence = loaded_results['influence']