import numpy as np
from scipy.signal import wiener
import matplotlib.pylab as plt


# Parameters
num_keys = 10000
key_size = 10
signal_mean = 0  # Mean of the original Gaussian distribution
signal_std = 1  # Standard deviation of the original Gaussian distribution
noise_mean = 0  # Mean of the noise Gaussian distribution
noise_std = 1 / 3  # Standard deviation of the noise Gaussian distribution

# Step 1: Generate random keys
keys = np.random.normal(signal_mean, signal_std, (num_keys, key_size))

# Step 2: Add noise to the keys
noise = np.random.normal(noise_mean, noise_std, (num_keys, key_size))
noisy_keys = keys + noise

# Step 3: Apply Wiener filter to denoise the keys
normalized_noisy_keys = (noisy_keys - np.mean(noisy_keys)) / np.std(noisy_keys)
denoised_keys = np.array(
    [
        wiener(normalized_noisy_key, mysize=9)
        for normalized_noisy_key in normalized_noisy_keys
    ]
)

# Step 4: rms error noisy and denoised keys
rms_error = np.sqrt(np.mean((noisy_keys - keys) ** 2, axis=1))
denoised_keys_error = np.sqrt(np.mean((denoised_keys - keys) ** 2, axis=1))
print("RMS error:", rms_error.mean())
print("RMS error of denoised keys:", denoised_keys_error.mean())
