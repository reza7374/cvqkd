import numpy as np
import matplotlib.pyplot as plt
import pywt


# Parameters
num_keys = 10000
key_size = 10
signal_mean = 0
signal_std = 1
noise_mean = 0
noise_std = 1 / 3

# Generate keys and add noise
keys = np.random.normal(signal_mean, signal_std, (num_keys, key_size))
noise = np.random.normal(noise_mean, noise_std, (num_keys, key_size))
noisy_keys = keys + noise


def adaptive_denoise(noisy_signal, estimated_noise_var, window_size=1, threshold=0):
    """
    Adaptive denoising function with configurable parameters.

    :param noisy_signal: The noisy input signal
    :param estimated_noise_var: Estimated noise variance
    :param window_size: Size of the window for local variance estimation
    :param threshold: Minimum value for alpha to prevent over-smoothing
    :return: Denoised signal
    """
    if window_size > 1:
        # Use a sliding window to estimate local signal variance
        pad_width = window_size // 2
        padded_signal = np.pad(
            noisy_signal, ((0, 0), (pad_width, pad_width)), mode="edge"
        )
        local_var = np.array(
            [
                np.var(padded_signal[:, i : i + window_size], axis=1)
                for i in range(key_size)
            ]
        ).T
    else:
        # Use global variance if window_size is 1
        local_var = np.var(noisy_signal, axis=1, keepdims=True)

    alpha = np.maximum(1 - estimated_noise_var / local_var, threshold)
    return alpha * noisy_signal


# Estimate noise variance
estimated_noise_var = np.var(noisy_keys - keys, axis=1, keepdims=True)

# Try different configurations
configs = [
    {"window_size": 1, "threshold": 0},
    {"window_size": 2, "threshold": 0},
    {"window_size": 1, "threshold": 0.01},
    {"window_size": 1, "threshold": 0.05},
    {"window_size": 2, "threshold": 0.01},
]


# def estimate_noise_variance(noisy_signal):
#     return (
#         np.median(np.abs(noisy_signal[:, 1:] - noisy_signal[:, :-1])) ** 2 / 0.9545**2
#     )


# estimated_noise_var = estimate_noise_variance(noisy_keys)

# results = {}

# for config in configs:
#     denoised_keys = adaptive_denoise(noisy_keys, estimated_noise_var, **config)
#     rms_error = np.sqrt(np.mean((denoised_keys - keys) ** 2, axis=1))
#     results[f"Adaptive (w={config['window_size']}, t={config['threshold']})"] = (
#         rms_error
#     )

# # Calculate RMS error for noisy keys as baseline
# noisy_rms_error = np.sqrt(np.mean((noisy_keys - keys) ** 2, axis=1))
# results["Noisy"] = noisy_rms_error

# # Print results
# for name, errors in results.items():
#     print(f"Mean RMS error of {name} keys: {errors.mean()}")

# # Plot histogram of RMS errors
# plt.figure(figsize=(12, 6))
# for name, errors in results.items():
#     plt.hist(errors, bins=50, alpha=0.5, label=name)
# plt.xlabel("RMS Error")
# plt.ylabel("Frequency")
# plt.title("Distribution of RMS Errors")
# plt.legend()
# plt.show()
import numpy as np
import matplotlib.pyplot as plt

# Your existing setup code here (num_keys, key_size, generating keys and noise, etc.)


# def estimate_noise_variance(noisy_signal):
#     return (
#         np.median(np.abs(noisy_signal[:, 1:] - noisy_signal[:, :-1])) ** 2 / 0.9545**2
#     )


# def adaptive_denoise(noisy_signal, estimated_noise_var, window_size=1, threshold=0):
#     if window_size > 1:
#         pad_width = window_size // 2
#         padded_signal = np.pad(
#             noisy_signal, ((0, 0), (pad_width, pad_width)), mode="edge"
#         )
#         local_var = np.array(
#             [
#                 np.var(padded_signal[:, i : i + window_size], axis=1)
#                 for i in range(noisy_signal.shape[1])
#             ]
#         ).T
#     else:
#         local_var = np.var(noisy_signal, axis=1, keepdims=True)

#     alpha = np.maximum(1 - estimated_noise_var / local_var, threshold)
#     return alpha * noisy_signal


# def adaptive_denoise_mad(noisy_signal, estimated_noise_var, threshold=0):
#     signal_mad = np.median(
#         np.abs(noisy_signal - np.median(noisy_signal, axis=1, keepdims=True)),
#         axis=1,
#         keepdims=True,
#     )
#     alpha = np.maximum(1 - estimated_noise_var / (signal_mad**2 * 1.4826**2), threshold)
#     return alpha * noisy_signal


# def wiener_filter(noisy_signal, estimated_noise_var):
#     freq_signal = np.fft.fft(noisy_signal, axis=1)
#     power = np.abs(freq_signal) ** 2
#     wiener_filter = np.maximum(1 - estimated_noise_var / power, 0)
#     return np.real(np.fft.ifft(freq_signal * wiener_filter, axis=1))


# def combined_denoise(noisy_signal, estimated_noise_var):
#     adaptive = adaptive_denoise(noisy_signal, estimated_noise_var)
#     wiener = wiener_filter(noisy_signal, estimated_noise_var)
#     return (adaptive + wiener) / 2


# estimated_noise_var = estimate_noise_variance(noisy_keys)

# methods = {
#     "Noisy": noisy_keys,
#     "Adaptive (w=1, t=0)": adaptive_denoise(
#         noisy_keys, estimated_noise_var, window_size=1, threshold=0
#     ),
#     "Adaptive (w=2, t=0)": adaptive_denoise(
#         noisy_keys, estimated_noise_var, window_size=2, threshold=0
#     ),
#     "Adaptive (w=1, t=0.01)": adaptive_denoise(
#         noisy_keys, estimated_noise_var, window_size=1, threshold=0.01
#     ),
#     "Adaptive MAD": adaptive_denoise_mad(noisy_keys, estimated_noise_var),
#     "Wiener": wiener_filter(noisy_keys, estimated_noise_var),
#     "Combined": combined_denoise(noisy_keys, estimated_noise_var),
# }

# results = {}
# for name, denoised_keys in methods.items():
#     rms_error = np.sqrt(np.mean((denoised_keys - keys) ** 2, axis=1))
#     results[name] = rms_error
#     print(f"Mean RMS error of {name} keys: {rms_error.mean()}")


# plt.figure(figsize=(12, 6))
# for name, errors in results.items():
#     plt.hist(errors, bins=50, alpha=0.5, label=name)
# plt.xlabel("RMS Error")
# plt.ylabel("Frequency")
# plt.title("Distribution of RMS Errors")
# plt.legend()
# plt.show()
def mild_denoise(noisy_signal, factor=0.1):
    signal_mean = np.mean(noisy_signal, axis=1, keepdims=True)
    return signal_mean + (noisy_signal - signal_mean) * (1 - factor)


def estimate_noise_wavelet(noisy_signal):
    coeffs = pywt.wavedec(noisy_signal, "db1", level=1, axis=1)
    detail = coeffs[-1]
    sigma = np.median(np.abs(detail)) / 0.6745
    return sigma**2


def wiener_filter_wavelet(noisy_signal, estimated_noise_var):
    freq_signal = np.fft.fft(noisy_signal, axis=1)
    power = np.abs(freq_signal) ** 2
    wiener_filter = np.maximum(1 - estimated_noise_var / power, 0)
    return np.real(np.fft.ifft(freq_signal * wiener_filter, axis=1))


methods = {
    "Noisy": noisy_keys,
    "Mild Denoised (0.1)": mild_denoise(noisy_keys, factor=0.1),
    "Mild Denoised (0.05)": mild_denoise(noisy_keys, factor=0.05),
    "Wiener (wavelet)": wiener_filter_wavelet(
        noisy_keys, estimate_noise_wavelet(noisy_keys)
    ),
}

results = {}
for name, denoised_keys in methods.items():
    rms_error = np.sqrt(np.mean((denoised_keys - keys) ** 2, axis=1))
    results[name] = rms_error
    print(f"Mean RMS error of {name} keys: {rms_error.mean()}")

plt.figure(figsize=(12, 6))
for name, errors in results.items():
    plt.hist(errors, bins=50, alpha=0.5, label=name)
plt.xlabel("RMS Error")
plt.ylabel("Frequency")
plt.title("Distribution of RMS Errors")
plt.legend()
plt.show()

# Analyze signal and noise characteristics
print(f"Signal mean: {np.mean(keys)}, Signal std: {np.std(keys)}")
print(
    f"Noise mean: {np.mean(noisy_keys - keys)}, Noise std: {np.std(noisy_keys - keys)}"
)
print(f"Signal-to-Noise Ratio: {np.var(keys) / np.var(noisy_keys - keys)}")
