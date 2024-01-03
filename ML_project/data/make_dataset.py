import os

import torch

if __name__ == "__main__":
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Define relative paths from the current directory
    raw_data_dir = os.path.join(current_dir[:-16], "data/raw/corruptmnist/")

    train_data, train_labels = [], []
    for i in range(5):
        train_data.append(torch.load(os.path.join(raw_data_dir, f"train_images_{i}.pt")))
        train_labels.append(torch.load(os.path.join(raw_data_dir, f"train_target_{i}.pt")))

    train_data = torch.cat(train_data, dim=0)
    train_labels = torch.cat(train_labels, dim=0)

    test_data = torch.load(os.path.join(raw_data_dir, "test_images.pt"))
    test_labels = torch.load(os.path.join(raw_data_dir, "test_target.pt"))

    # Normalize the data
    train_mean = train_data.mean(dim=(0, 2, 3))  # Calculate mean per batch along 28x28 dimensions
    train_std = train_data.std(dim=(0, 2, 3))  # Calculate std per batch along 28x28 dimensions
    test_mean = test_data.mean(dim=(0, 2, 3))
    test_std = test_data.std(dim=(0, 2, 3))

    # Normalize train and test data using calculated mean and std
    train_data = (train_data - train_mean.view(1, -1, 1, 1)) / train_std.view(1, -1, 1, 1)
    test_data = (test_data - test_mean.view(1, -1, 1, 1)) / test_std.view(1, -1, 1, 1)

    print(train_data.shape)
    print(train_labels.shape)
    print(test_data.shape)
    print(test_labels.shape)

    print(train_std.shape)
