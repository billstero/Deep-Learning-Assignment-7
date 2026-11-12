import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from models import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_mlp(epochs=15):
    train_loader, _, _ = get_dataloaders()
    model = MLPClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            correct += logits.argmax(1).eq(y).sum().item()
            total += y.size(0)
            
        acc = 100. * correct / total
        print(f"MLP Epoch {epoch+1}/{epochs} | Loss: {train_loss/len(train_loader):.4f} | Acc: {acc:.2f}%")
        
    torch.save(model.state_dict(), "mlp_classifier.pth")
    return model

def train_ae(model_class, name, epochs=10, is_sparse=False, lambda_l1=1e-3):
    train_loader, _, _ = get_dataloaders()
    model = model_class().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            x_recon, z = model(x)
            loss = criterion(x_recon, x)
            
            if is_sparse:
                loss += lambda_l1 * torch.norm(z, 1)
                
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        print(f"{name} Epoch {epoch+1}/{epochs} | Loss: {train_loss/len(train_loader):.4f}")
        
    torch.save(model.state_dict(), f"{name.replace(' ', '_').lower()}.pth")
    return model

def train_vae(epochs=10):
    train_loader, _, _ = get_dataloaders()
    model = VariationalAutoencoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            x_recon, z, mu, logvar = model(x)
            
            mse = nn.functional.mse_loss(x_recon, x, reduction='sum')
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = mse + kld
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        print(f"VAE Epoch {epoch+1}/{epochs} | Loss: {train_loss/len(train_loader.dataset):.4f}")
        
    torch.save(model.state_dict(), "vae.pth")
    return model

def train_gan(G_class, D_class, name, epochs=20):
    train_loader, _, _ = get_dataloaders()
    G = G_class().to(device)
    D = D_class().to(device)
    
    criterion = nn.BCELoss()
    opt_G = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    
    for epoch in range(epochs):
        g_loss_avg, d_loss_avg = 0, 0
        for x, _ in train_loader:
            batch_size = x.size(0)
            real_imgs = x.to(device)
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            
            opt_D.zero_grad()
            d_real = criterion(D(real_imgs), real_labels)
            z = torch.randn(batch_size, 100).to(device)
            fake_imgs = G(z)
            d_fake = criterion(D(fake_imgs.detach()), fake_labels)
            d_loss = d_real + d_fake
            d_loss.backward()
            opt_D.step()
            
            opt_G.zero_grad()
            g_loss = criterion(D(fake_imgs), real_labels)
            g_loss.backward()
            opt_G.step()
            
            d_loss_avg += d_loss.item()
            g_loss_avg += g_loss.item()
            
        print(f"{name} Epoch {epoch+1}/{epochs} | D Loss: {d_loss_avg/len(train_loader):.4f} | G Loss: {g_loss_avg/len(train_loader):.4f}")
        
    torch.save(G.state_dict(), f"{name.replace(' ', '_').lower()}_g.pth")
    torch.save(D.state_dict(), f"{name.replace(' ', '_').lower()}_d.pth")
