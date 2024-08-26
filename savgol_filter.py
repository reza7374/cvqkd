from scipy.signal import savgol_filter
import numpy as np
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
# Step 3: Apply Savitzky-Golay filter to denoise the keys
# Parameters: window_length and polyorder
window_length = 10  # Must be odd and >= polyorder + 2
polyorder = 2  # The order of the polynomial used in filtering

denoised_keys_sg = np.array(
    [savgol_filter(noisy_key, window_length, polyorder) for noisy_key in noisy_keys]
)

# Step 4: Calculate RMS error for noisy and denoised keys
rms_error = np.sqrt(np.mean((noisy_keys - keys) ** 2, axis=1))
denoised_keys_error_sg = np.sqrt(np.mean((denoised_keys_sg - keys) ** 2, axis=1))

print("RMS error:", rms_error.mean())
print("RMS error of Savitzky-Golay denoised keys:", denoised_keys_error_sg.mean())
