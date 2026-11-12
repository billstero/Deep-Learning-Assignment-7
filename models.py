import torch
import torch.nn as nn

class MLPClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        h1 = self.relu(self.fc1(x.view(-1, 784)))
        z = self.relu(self.fc2(h1))
        return self.fc3(z), z

class VanillaAutoencoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, latent_dim), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 784), nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x.view(-1, 784))
        return self.decoder(z).view(-1, 1, 28, 28), z

class SparseAutoencoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, latent_dim), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 784), nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x.view(-1, 784))
        return self.decoder(z).view(-1, 1, 28, 28), z

class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        self.relu = nn.ReLU()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 784), nn.Sigmoid()
        )

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc_mu(h1), self.fc_logvar(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decoder(z).view(-1, 1, 28, 28), z, mu, logvar

class VanillaGAN_Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 784), nn.Sigmoid()
        )
    def forward(self, z):
        return self.model(z).view(-1, 1, 28, 28)

class VanillaGAN_Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x.view(-1, 784))

class DCGAN_Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 128, 7, 1, 0, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, z):
        return self.model(z.view(-1, z.size(1), 1, 1))

class DCGAN_Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True)
        )
        self.classifier = nn.Sequential(nn.Linear(128 * 7 * 7, 1), nn.Sigmoid())
    def forward(self, x):
        out = self.features(x).view(x.size(0), -1)
        return self.classifier(out)

class DDPM_UNet(nn.Module):
    def __init__(self, num_steps=1000):
        super().__init__()
        self.time_emb = nn.Embedding(num_steps, 128)
        self.time_mlp = nn.Sequential(nn.Linear(128, 128), nn.ReLU())
        
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.te1 = nn.Linear(128, 64)
        
        self.down1 = nn.Conv2d(64, 64, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.te2 = nn.Linear(128, 128)
        
        self.down2 = nn.Conv2d(128, 128, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)
        self.te3 = nn.Linear(128, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 64, 3, padding=1)
        self.te4 = nn.Linear(128, 64)
        
        self.up2 = nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(128, 64, 3, padding=1)
        self.te5 = nn.Linear(128, 64)
        
        self.out = nn.Conv2d(64, 1, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        t_emb = self.time_mlp(self.time_emb(t))
        
        x1 = self.relu(self.conv1(x) + self.te1(t_emb).view(-1, 64, 1, 1))
        x2 = self.relu(self.conv2(self.down1(x1)) + self.te2(t_emb).view(-1, 128, 1, 1))
        x3 = self.relu(self.conv3(self.down2(x2)) + self.te3(t_emb).view(-1, 128, 1, 1))
        
        x4 = self.up1(x3)
        x4 = torch.cat([x4, x2], dim=1)
        x4 = self.relu(self.conv4(x4) + self.te4(t_emb).view(-1, 64, 1, 1))
        
        x5 = self.up2(x4)
        x5 = torch.cat([x5, x1], dim=1)
        x5 = self.relu(self.conv5(x5) + self.te5(t_emb).view(-1, 64, 1, 1))
        
        return self.out(x5)
