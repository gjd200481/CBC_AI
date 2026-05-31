import numpy as np
import matplotlib.pyplot as plt

# ==========================
# Parameters
# ==========================

N = 512          # sampling points
L = 10e-3        # simulation window (m)

w0 = 1e-3        # beam waist (m)

x = np.linspace(    -L/2,    L/2,    N)
X,Y=np.meshgrid(    x,    x)

# ==========================
# Gaussian beam
# ==========================

E=np.exp(-(X**2+Y**2)/w0**2)

# ==========================
# FFT propagation
# ==========================

Farfield=np.fft.fftshift(np.fft.fft2(E))
Intensity=np.abs(Farfield)**2
Intensity/=np.max(Intensity)

# ==========================
# Plot
# ==========================

plt.figure(figsize=(10,4))
plt.subplot(121)
plt.imshow(E,cmap='jet')
plt.title("Near field")
plt.colorbar()
plt.subplot(122)
plt.imshow(Intensity,cmap='jet')
plt.title("Far field")
plt.colorbar()
plt.tight_layout()
plt.show()