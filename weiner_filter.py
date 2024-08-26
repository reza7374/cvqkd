import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_keys = 10000
key_size = 10
signal_mean = 0
signal_std = 1
noise_mean = 0
noise_std = 1 / 3

# Step 1: Generate random keys
keys = np.random.normal(signal_mean, signal_std, (num_keys, key_size))

# Step 2: Add noise to the keys
noise = np.random.normal(noise_mean, noise_std, (num_keys, key_size))
noisy_keys = keys + noise


# Step 3: Denoise the keys
def denoise(noisy_signal, noise_var):
    signal_var = np.var(noisy_signal)
    alpha = signal_var / (signal_var + noise_var)
    return alpha * noisy_signal


noise_var = noise_std**2
denoised_keys = denoise(noisy_keys, noise_var)

# Step 4: Calculate RMS error for noisy and denoised keys
rms_error_noisy = np.sqrt(np.mean((noisy_keys - keys) ** 2, axis=1))
rms_error_denoised = np.sqrt(np.mean((denoised_keys - keys) ** 2, axis=1))

print("Mean RMS error of noisy keys:", rms_error_noisy.mean())
print("Mean RMS error of denoised keys:", rms_error_denoised.mean())

# Step 5: Plot histogram of RMS errors
plt.figure(figsize=(10, 6))
plt.hist(rms_error_noisy, bins=50, alpha=0.5, label="Noisy Keys")
plt.hist(rms_error_denoised, bins=50, alpha=0.5, label="Denoised Keys")
plt.xlabel("RMS Error")
plt.ylabel("Frequency")
plt.title("Distribution of RMS Errors")
plt.legend()
plt.show()
