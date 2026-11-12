import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, ConcatDataset

def get_dataloaders(batch_size=128, seed=42):
    torch.manual_seed(seed)
    
    transform = transforms.ToTensor()
    
    train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    full_dataset = ConcatDataset([train_data, test_data])
    total = len(full_dataset)
    
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)
    test_size = total - train_size - val_size
    
    train_ds, val_ds, test_ds = random_split(full_dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()
    print(f"Train: {len(train_loader.dataset)} citra, {len(train_loader)} batch")
    print(f"Val: {len(val_loader.dataset)} citra, {len(val_loader)} batch")
    print(f"Test: {len(test_loader.dataset)} citra, {len(test_loader)} batch")
