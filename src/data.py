"""Fashion-MNIST loading, stratified 80/10/10 split and DataLoaders."""
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision import datasets, transforms

from .utils import DATA_DIR

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

def get_datasets(root=DATA_DIR, seed=42):
    tf = transforms.Compose([
        transforms.ToTensor(),  # rescale to float tensor [0,1]. Shape (1, H, W)
    ])
    train = datasets.FashionMNIST(root=root, train=True, download=True, transform=tf)
    test = datasets.FashionMNIST(root=root, train=False, download=True, transform=tf)
    data = ConcatDataset([train, test])

    labels = torch.cat([train.targets, test.targets]).numpy()
    idx = list(range(len(data)))

    train_idx, temp_idx = train_test_split(idx, test_size=0.2, stratify=labels, random_state=seed)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=labels[temp_idx], random_state=seed)

    return Subset(data, train_idx), Subset(data, val_idx), Subset(data, test_idx)

def get_dataloaders(batch_size=256, root=DATA_DIR, seed=42):
    """shuffle=True for training, shuffle=False for validation/test (as the assignment requires)."""
    train_data, val_data, test_data = get_datasets(root, seed)

    train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_dl, val_dl, test_dl
