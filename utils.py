import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_image_grid(images, filename, nrow=8, title=None):
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)
    n_imgs = images.size(0)
    ncol = nrow
    nrows = (n_imgs + ncol - 1) // ncol
    
    fig, axes = plt.subplots(nrows, ncol, figsize=(ncol * 1.5, nrows * 1.5))
    axes = np.array(axes).reshape(-1)
    
    for idx in range(len(axes)):
        axes[idx].axis("off")
        if idx < n_imgs:
            axes[idx].imshow(images[idx, 0].detach().cpu().numpy(), cmap="gray")
            
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")
        
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close(fig)
