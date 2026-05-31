import numpy as np
import matplotlib.pyplot as plt


# ==========================
# Parameters
# ==========================

N = 512
L = 10e-3
w0 = 0.5e-3
d = 1.5e-3      # beam spacing
phi = np.pi     # phase difference


x = np.linspace(
    -L/2,
    L/2,
    N
)

X,Y=np.meshgrid(
    x,
    x
)


# ==========================
# Beam1
# ==========================
E1=np.exp(-((X+d/2)**2+Y**2)/w0**2)
# ==========================
# Beam2
# ==========================

E2=np.exp(-((X-d/2)**2+Y**2)/w0**2)*np.exp(1j*phi)


# ==========================
# Coherent sum
# ==========================

E=E1+E2


# ==========================
# FFT
# ==========================

Farfield=np.fft.fftshift(
np.fft.fft2(
E
)
)

Intensity=np.abs(
Farfield
)**2

Intensity/=np.max(
Intensity
)


# ==========================
# Plot
# ==========================

plt.figure(
figsize=(10,4)
)

plt.subplot(
121
)

plt.imshow(
np.abs(E),
cmap='jet'
)

plt.title(
"Near field"
)

plt.colorbar()
plt.subplot(122)
zoom = 10       # 调整放大倍数，可试 50~120
center = N // 2
plt.imshow(
    Intensity[
        center-zoom:center+zoom,
        center-zoom:center+zoom
    ],
    cmap='jet'
)
plt.title(
    f"Far field, phi={phi}"
)
plt.colorbar()
plt.tight_layout()
plt.show()
