import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler

# Example data
data_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
labels_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

# Number of samples to select for the subset
num_samples = 5

# Create a random subset of indices
subset_indices = np.random.choice(len(data_list), size=num_samples, replace=False)

# Custom sampler for the subset
subset_sampler = SubsetRandomSampler(subset_indices)

# Create DataLoader for the subset
subset_loader = DataLoader(list(zip(data_list, labels_list)), batch_size=20, sampler=subset_sampler)

# Iterate over the subset loader
for i in range(5):
    for batch in subset_loader:
        data_batch, labels_batch = batch
        print("Data Batch:", sorted(data_batch))
