import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from torchvision.models import inception_v3, Inception_V3_Weights
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
from dataset import get_dataloaders
from models import (
    VanillaAutoencoder,
    SparseAutoencoder,
    VariationalAutoencoder,
    VanillaGAN_Generator,
    DCGAN_Generator,
    DDPM_UNet
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def sample_ddpm(model, n_samples=16, num_steps=1000):
    model.eval()
    betas = torch.linspace(1e-4, 0.02, num_steps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    x = torch.randn(n_samples, 1, 28, 28, device=device)
    for t in reversed(range(num_steps)):
        t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
        pred_noise = model(x, t_batch)
        
        alpha_t = alphas[t]
        alpha_hat_t = alphas_cumprod[t]
        beta_t = betas[t]
        
        z = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        x = (1.0 / alpha_t.sqrt()) * (x - ((1.0 - alpha_t) / (1.0 - alpha_hat_t).sqrt()) * pred_noise) + beta_t.sqrt() * z
        
    return (x.clamp(-1.0, 1.0) + 1.0) / 2.0

@torch.no_grad()
def evaluate_reconstruction(models_dict, test_loader):
    results = {}
    for name, model in models_dict.items():
        model.eval()
        mse_list, psnr_list, ssim_list = [], [], []
        
        for x, _ in test_loader:
            x = x.to(device)
            out = model(x)
            x_recon = out[0]
            
            mse_val = F.mse_loss(x_recon, x).item()
            mse_list.append(mse_val)
            
            orig_np = x.cpu().numpy()
            recon_np = x_recon.cpu().numpy()
            for i in range(orig_np.shape[0]):
                o = orig_np[i, 0]
                r = recon_np[i, 0]
                psnr_list.append(psnr_metric(o, r, data_range=1.0))
                ssim_list.append(ssim_metric(o, r, data_range=1.0))
                
        results[name] = {
            "MSE": np.mean(mse_list),
            "PSNR": np.mean(psnr_list),
            "SSIM": np.mean(ssim_list)
        }
    return results

@torch.no_grad()
def extract_inception_features(images, inception_model, batch_size=64):
    features = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size].to(device)
        batch = batch.repeat(1, 3, 1, 1)
        batch = F.interpolate(batch, size=(299, 299), mode="bilinear", align_corners=False)
        batch = batch * 2.0 - 1.0
        feat = inception_model(batch)
        features.append(feat.cpu().numpy())
    return np.concatenate(features, axis=0)

def calculate_fid(real_features, fake_features):
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
        
    return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean))

@torch.no_grad()
def run_full_evaluation(num_fid_samples=512):
    _, _, test_loader = get_dataloaders()
    
    vanilla_ae = VanillaAutoencoder().to(device)
    vanilla_ae.load_state_dict(torch.load("vanilla_ae.pth", weights_only=True))
    
    sparse_ae = SparseAutoencoder().to(device)
    sparse_ae.load_state_dict(torch.load("sparse_ae.pth", weights_only=True))
    
    vae = VariationalAutoencoder().to(device)
    vae.load_state_dict(torch.load("vae.pth", weights_only=True))
    
    vanilla_gan = VanillaGAN_Generator().to(device)
    vanilla_gan.load_state_dict(torch.load("vanilla_gan_g.pth", weights_only=True))
    
    dcgan = DCGAN_Generator().to(device)
    dcgan.load_state_dict(torch.load("dcgan_g.pth", weights_only=True))
    
    ddpm = DDPM_UNet(num_steps=1000).to(device)
    ddpm.load_state_dict(torch.load("ddpm.pth", weights_only=True))
    
    recon_models = {
        "Vanilla AE": vanilla_ae,
        "Sparse AE": sparse_ae,
        "VAE": vae
    }
    recon_metrics = evaluate_reconstruction(recon_models, test_loader)
    
    print("=== Reconstruction Metrics ===")
    for name, m in recon_metrics.items():
        print(f"{name:12s} | MSE: {m['MSE']:.4f} | PSNR: {m['PSNR']:.2f} dB | SSIM: {m['SSIM']:.4f}")
        
    inception = inception_v3(weights=Inception_V3_Weights.DEFAULT).to(device)
    inception.fc = nn.Identity()
    inception.eval()
    
    real_imgs = []
    for x, _ in test_loader:
        real_imgs.append(x)
        if sum(img.size(0) for img in real_imgs) >= num_fid_samples:
            break
    real_imgs = torch.cat(real_imgs, dim=0)[:num_fid_samples]
    real_feats = extract_inception_features(real_imgs, inception)
    
    vae.eval()
    vanilla_gan.eval()
    dcgan.eval()
    
    vae_samples = vae.decoder(torch.randn(num_fid_samples, 64, device=device)).view(-1, 1, 28, 28)
    vgan_samples = vanilla_gan(torch.randn(num_fid_samples, 100, device=device))
    dcgan_samples = dcgan(torch.randn(num_fid_samples, 100, device=device))
    ddpm_samples = sample_ddpm(ddpm, n_samples=num_fid_samples, num_steps=1000)
    
    gen_dict = {
        "VAE": vae_samples,
        "Vanilla GAN": vgan_samples,
        "DCGAN": dcgan_samples,
        "DDPM": ddpm_samples
    }
    
    print("\n=== Generative FID Score (Lower is Better) ===")
    fid_scores = {}
    for name, samples in gen_dict.items():
        fake_feats = extract_inception_features(samples, inception)
        score = calculate_fid(real_feats, fake_feats)
        fid_scores[name] = score
        print(f"{name:12s} | FID: {score:.2f}")
        
    fig, axes = plt.subplots(4, 8, figsize=(12, 6))
    for row, (name, samples) in enumerate(gen_dict.items()):
        for col in range(8):
            axes[row, col].imshow(samples[col, 0].cpu().numpy(), cmap="gray")
            axes[row, col].axis("off")
            if col == 0:
                axes[row, col].set_title(name, loc="left", fontsize=10, fontweight="bold")
                
    plt.tight_layout()
    plt.savefig("generative_comparison.png", dpi=200)
    plt.show()
    
    return recon_metrics, fid_scores
