import numpy as np
import matplotlib.pyplot as plt


# =====================
# 参数设置
# =====================


N = 512
L = 10e-3
w0 = 0.5e-3
d = 1e-3
phi = np.pi / 2


# =====================
# 坐标网格
# =====================


x = np.linspace(-L / 2, L / 2, N)
y = np.linspace(-L / 2, L / 2, N)
X, Y = np.meshgrid(x, y)


# =====================
# 两束高斯光
# =====================


E1 = np.exp(-((X + d / 2) ** 2 + Y**2) / w0**2)

E2 = np.exp(-((X - d / 2) ** 2 + Y**2) / w0**2)
E2 = E2 * np.exp(1j * phi)


# =====================
# 近场叠加与远场计算
# =====================


E = E1 + E2
far_field = np.fft.fftshift(np.fft.fft2(E))
intensity = np.abs(far_field) ** 2
intensity = intensity / np.max(intensity)


# =====================
# 结果可视化
# =====================


plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(np.abs(E), cmap="hot")
plt.title("Near Field")

plt.subplot(1, 2, 2)
plt.imshow(intensity, cmap="hot")
plt.title("Far Field")

plt.tight_layout()
plt.show()
