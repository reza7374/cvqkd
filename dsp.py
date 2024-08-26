import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
import pywt

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


# Original denoising method
def denoise(noisy_signal, noise_var):
    signal_var = np.var(noisy_signal)
    alpha = signal_var / (signal_var + noise_var)
    return alpha * noisy_signal


# Adaptive denoising
def adaptive_denoise(noisy_signal, estimated_noise_var):
    signal_var = np.var(noisy_signal, axis=1, keepdims=True)
    alpha = np.maximum(1 - estimated_noise_var / signal_var, 0)
    return alpha * noisy_signal


# Wiener filtering in frequency domain
def wiener_filter_freq(noisy_signal, noise_var):
    freq_signal = fft(noisy_signal, axis=1)
    power_spectrum = np.abs(freq_signal) ** 2
    wiener_filter = np.maximum(1 - noise_var / power_spectrum, 0)
    return np.real(ifft(freq_signal * wiener_filter, axis=1))


# Wavelet denoising
def wavelet_denoise(noisy_signal, wavelet="db4", level=2):
    coeffs = pywt.wavedec(noisy_signal, wavelet, level=level, axis=1)
    threshold = (
        np.sqrt(2 * np.log(noisy_signal.shape[1]))
        * np.median(np.abs(coeffs[-1]))
        / 0.6745
    )
    denoised_coeffs = [pywt.threshold(c, threshold, mode="soft") for c in coeffs]
    return pywt.waverec(denoised_coeffs, wavelet, axis=1)


# Apply all methods
noise_var = noise_std**2
denoised_keys = denoise(noisy_keys, noise_var)
estimated_noise_var = np.var(noisy_keys - keys, axis=1, keepdims=True)
adaptive_denoised_keys = adaptive_denoise(noisy_keys, estimated_noise_var)
wiener_denoised_keys = wiener_filter_freq(noisy_keys, noise_var)
wavelet_denoised_keys = wavelet_denoise(noisy_keys)
ensemble_denoised_keys = (
    denoised_keys
    + adaptive_denoised_keys
    + wiener_denoised_keys
    + wavelet_denoised_keys
) / 4

# Calculate RMS errors
methods = {
    "Noisy": noisy_keys,
    "Original Denoised": denoised_keys,
    "Adaptive Denoised": adaptive_denoised_keys,
    "Wiener Denoised": wiener_denoised_keys,
    "Wavelet Denoised": wavelet_denoised_keys,
    "Ensemble Denoised": ensemble_denoised_keys,
}

for name, method_keys in methods.items():
    rms_error = np.sqrt(np.mean((method_keys - keys) ** 2, axis=1))
    print(f"Mean RMS error of {name} keys:", rms_error.mean())

# Plot histogram of RMS errors
plt.figure(figsize=(12, 6))
for name, method_keys in methods.items():
    rms_error = np.sqrt(np.mean((method_keys - keys) ** 2, axis=1))
    plt.hist(rms_error, bins=50, alpha=0.5, label=name)
plt.xlabel("RMS Error")
plt.ylabel("Frequency")
plt.title("Distribution of RMS Errors")
plt.legend()
plt.show()
