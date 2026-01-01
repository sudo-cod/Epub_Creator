import numpy as np
import matplotlib.pyplot as plt
import time

# Given signal
y = np.array([-1, 2, 3, 0, -2, 1, 4, -3, 0, -2])
L = len(y)
n = np.arange(L)

print(f"Signal length L = {L}")
print(f"y[n] = {y}")

# a) Plot y[n]
plt.figure(figsize=(10, 6))
plt.stem(n, y, linefmt='b-', markerfmt='bo', basefmt='r-')
plt.title('a) Original Signal y[n]')
plt.xlabel('n')
plt.ylabel('y[n]')
plt.grid(True)
plt.show()


# b) Calculate and plot DTFT Y(e^jω) in Nyquist interval [-π, π]
def dtft(x, omega):
    """Compute DTFT of signal x at frequencies omega"""
    N = len(x)
    n = np.arange(N)
    X = np.zeros(len(omega), dtype=complex)
    for k, w in enumerate(omega):
        X[k] = np.sum(x * np.exp(-1j * w * n))
    return X


# Create frequency vector
omega = np.linspace(-np.pi, np.pi, 1024)
Y_dtft = dtft(y, omega)

# Modulus and phase
Y_dtft_modulus = np.abs(Y_dtft)
Y_dtft_phase = np.angle(Y_dtft)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(omega, Y_dtft_modulus)
plt.title('b) DTFT Modulus |Y(e^jω)|')
plt.xlabel('ω (rad/sample)')
plt.ylabel('|Y(e^jω)|')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(omega, Y_dtft_phase)
plt.title('b) DTFT Phase ∠Y(e^jω)')
plt.xlabel('ω (rad/sample)')
plt.ylabel('Phase (rad)')
plt.grid(True)
plt.tight_layout()
plt.show()

# c) Calculate N-point DFT (N=10) and compare with DTFT
N_dft = 10
Y_dft = np.fft.fft(y, N_dft)
omega_k = 2 * np.pi * np.arange(N_dft) / N_dft
omega_k_centered = np.where(omega_k > np.pi, omega_k - 2 * np.pi, omega_k)

Y_dft_modulus = np.abs(Y_dft)
Y_dft_phase = np.angle(Y_dft)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(omega, Y_dtft_modulus, 'b-', label='DTFT', alpha=0.7)
plt.stem(omega_k_centered, Y_dft_modulus, linefmt='r-', markerfmt='ro', basefmt='r-', label='DFT')
plt.title('c) DFT vs DTFT Modulus')
plt.xlabel('ω (rad/sample)')
plt.ylabel('|Y(ω)|')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(omega, Y_dtft_phase, 'b-', label='DTFT', alpha=0.7)
plt.stem(omega_k_centered, Y_dft_phase, linefmt='r-', markerfmt='ro', basefmt='r-', label='DFT')
plt.title('c) DFT vs DTFT Phase')
plt.xlabel('ω (rad/sample)')
plt.ylabel('Phase (rad)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# d) Calculate inverse DFT and compare with y[n]
y_reconstructed = np.fft.ifft(Y_dft)

plt.figure(figsize=(10, 6))
plt.stem(n, y, linefmt='b-', markerfmt='bo', basefmt='r-', label='Original y[n]')
plt.stem(n, np.real(y_reconstructed), linefmt='g--', markerfmt='gx', basefmt='r-', label='Reconstructed')
plt.title('d) Inverse DFT Comparison')
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()

print("\n" + "=" * 50)
print("d) Inverse DFT Comparison:")
print("Original signal:", y)
print("Reconstructed:  ", np.real(y_reconstructed))
print("Maximum error:   ", np.max(np.abs(y - np.real(y_reconstructed))))

# e) Zero-padding to L=16 and calculate 16-point FFT
L_zero = 16
y_zero_padded = np.zeros(L_zero)
y_zero_padded[:L] = y

Y_fft_16 = np.fft.fft(y_zero_padded)
omega_16 = 2 * np.pi * np.arange(L_zero) / L_zero
omega_16_centered = np.where(omega_16 > np.pi, omega_16 - 2 * np.pi, omega_16)

Y_fft_16_modulus = np.abs(Y_fft_16)
Y_fft_16_phase = np.angle(Y_fft_16)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(omega, Y_dtft_modulus, 'b-', label='DTFT', alpha=0.7, linewidth=2)
plt.stem(omega_16_centered, Y_fft_16_modulus, linefmt='g-', markerfmt='go', basefmt='g-', label='16-point FFT')
plt.title('e) 16-point FFT vs DTFT Modulus')
plt.xlabel('ω (rad/sample)')
plt.ylabel('|Y(ω)|')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(omega, Y_dtft_phase, 'b-', label='DTFT', alpha=0.7, linewidth=2)
plt.stem(omega_16_centered, Y_fft_16_phase, linefmt='g-', markerfmt='go', basefmt='g-', label='16-point FFT')
plt.title('e) 16-point FFT vs DTFT Phase')
plt.xlabel('ω (rad/sample)')
plt.ylabel('Phase (rad)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"\nZero-padded signal length: {L_zero}")
print("FFT result matches DTFT with better frequency resolution")


# f) Computational time comparison for DFT vs FFT
def dft_direct(x):
    """Direct DFT implementation (O(N^2))"""
    N = len(x)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    return X


L_values = np.arange(1000, 10001, 1000)
dft_times = []
fft_times = []

print("\n" + "=" * 50)
print("f) Computational Time Analysis:")
print("L\tDFT Time (s)\tFFT Time (s)\tSpeedup")

for L_val in L_values:
    # Generate random test signal
    x_test = np.random.randn(L_val)

    # Time DFT
    start_time = time.time()
    X_dft = dft_direct(x_test)
    dft_time = time.time() - start_time

    # Time FFT
    start_time = time.time()
    X_fft = np.fft.fft(x_test)
    fft_time = time.time() - start_time

    dft_times.append(dft_time)
    fft_times.append(fft_time)

    print(f"{L_val}\t{dft_time:.4f}\t\t{fft_time:.6f}\t\t{dft_time / fft_time:.1f}x")

# Plot computational time comparison
plt.figure(figsize=(10, 6))
plt.plot(L_values, dft_times, 'ro-', linewidth=2, markersize=6, label='Direct DFT (O(N²))')
plt.plot(L_values, fft_times, 'bo-', linewidth=2, markersize=6, label='FFT (O(N log N))')
plt.xlabel('Signal Length L')
plt.ylabel('Computational Time (seconds)')
plt.title('f) Computational Time: DFT vs FFT')
plt.legend()
plt.grid(True)
plt.yscale('log')
plt.show()

# Additional analysis: Show the complexity curves
plt.figure(figsize=(10, 6))

# Theoretical complexity curves
L_theoretical = np.linspace(100, 10000, 100)
dft_complexity = L_theoretical ** 2 * 1e-7  # Scaled for visualization
fft_complexity = L_theoretical * np.log2(L_theoretical) * 1e-6  # Scaled for visualization

plt.plot(L_theoretical, dft_complexity, 'r--', alpha=0.7, label='Theoretical O(N²)')
plt.plot(L_theoretical, fft_complexity, 'b--', alpha=0.7, label='Theoretical O(N log N)')
plt.plot(L_values, dft_times, 'ro-', linewidth=2, markersize=6, label='Measured DFT')
plt.plot(L_values, fft_times, 'bo-', linewidth=2, markersize=6, label='Measured FFT')

plt.xlabel('Signal Length L')
plt.ylabel('Relative Computational Time')
plt.title('f) Theoretical vs Measured Computational Complexity')
plt.legend()
plt.grid(True)
plt.yscale('log')
plt.show()

print("\n" + "=" * 50)
print("SUMMARY:")
print("1. The 10-point DFT perfectly samples the DTFT at discrete frequencies")
print("2. Inverse DFT perfectly reconstructs the original signal")
print("3. 16-point FFT with zero-padding provides better frequency resolution")
print("4. FFT shows significant computational advantage over direct DFT for large L")