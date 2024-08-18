import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_keys(num_keys, key_length, mean=0, std=1):
    return np.random.normal(mean, std, size=(num_keys, key_length))


def add_noise(keys, noise_sigma):
    noise = np.random.normal(0, noise_sigma, size=keys.shape)
    return keys + noise


class VAE(nn.Module):
    def __init__(
        self, input_size=10, hidden_size1=64, hidden_size2=32, bottleneck_size=16
    ):
        super(VAE, self).__init__()

        # Encoder: Instead of outputting a single bottleneck layer, output mean and log variance
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size1),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size2),
        )

        # Separate layers for mean and log variance
        self.fc_mu = nn.Linear(hidden_size2, bottleneck_size)
        self.fc_logvar = nn.Linear(hidden_size2, bottleneck_size)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_size, hidden_size2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size2),
            nn.Linear(hidden_size2, hidden_size1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size1),
            nn.Linear(hidden_size1, input_size),
            nn.Sigmoid(),  # Assuming the output is normalized
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # Calculate standard deviation from log variance
        eps = torch.randn_like(std)  # Sample epsilon from standard normal distribution
        return mu + eps * std  # Reparameterization trick

    def forward(self, x):
        # Encode input
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        # Sample latent vector using reparameterization trick
        z = self.reparameterize(mu, logvar)

        # Decode latent vector
        x_reconstructed = self.decoder(z)

        return (
            x_reconstructed,
            mu,
            logvar,
        )


def vae_loss_function(recon_x, x, mu, logvar):
    # Reconstruction loss
    recon_loss = F.mse_loss(
        recon_x, x, reduction="sum"
    )  # or F.binary_cross_entropy(recon_x, x, reduction='sum')

    # KL divergence
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Total loss
    return recon_loss + kl_div


def train_model(noisy_keys, clean_keys, epochs, lr, batch_size=32):
    model = VAE(
        input_size=noisy_keys.shape[1],
        hidden_size1=64,
        hidden_size2=32,
        bottleneck_size=16,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for i in range(0, len(noisy_keys), batch_size):
            batch_noisy = noisy_keys[i : i + batch_size].to(device)
            batch_clean = clean_keys[i : i + batch_size].to(device)

            optimizer.zero_grad()
            recon_batch, mu, logvar = model(batch_noisy)

            # Compute loss
            recon_loss = F.mse_loss(recon_batch, batch_clean, reduction="sum")
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_div

            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    return model


num_keys = 100000
key_length = 10
sigma = 1
noise_sigma = sigma / 7
clean_keys = generate_keys(num_keys, key_length)
noisy_keys = add_noise(clean_keys, noise_sigma)


batch_size = 128
epochs = 2
lr = 0.01
noisy_keys = torch.tensor(noisy_keys, dtype=torch.float32)
clean_keys = torch.tensor(clean_keys, dtype=torch.float32)
model = train_model(noisy_keys, clean_keys, epochs, lr).to(device)


test_clean_keys = generate_keys(5, key_length)
test_noisy_keys = add_noise(test_clean_keys, noise_sigma)
test_noisy_keys = torch.tensor(test_noisy_keys, dtype=torch.float32).to(device)
test_denoised_keys = model(test_noisy_keys)
test_noisy_keys = test_noisy_keys.cpu().numpy()
test_denoised_keys = test_denoised_keys[0].detach().cpu().numpy()
x = test_noisy_keys - test_clean_keys
x = np.mean(x, axis=0)
y = test_denoised_keys - test_clean_keys
y = np.mean(y, axis=0)
plt.plot(x, label="Noisy Keys")
plt.plot(y, label="Denoised Keys")
plt.legend()
plt.show()
