import numpy as np
import matplotlib.pyplot as plt


# ==========================
# 参数设置
# ==========================


N = 512
L = 10e-3
w0 = 0.5e-3
d = 1.5e-3
phi = np.pi


# ==========================
# 坐标网格
# ==========================


x = np.linspace(-L / 2, L / 2, N)
X, Y = np.meshgrid(x, x)


# ==========================
# 两束高斯光
# ==========================


# 第一束光位于 x = -d/2，作为相位参考光束。
E1 = np.exp(-((X + d / 2) ** 2 + Y**2) / w0**2)

# 第二束光位于 x = +d/2，并叠加相位差 phi。
E2 = np.exp(-((X - d / 2) ** 2 + Y**2) / w0**2)
E2 = E2 * np.exp(1j * phi)


# ==========================
# 近场叠加
# ==========================


E = E1 + E2


# ==========================
# 远场计算
# ==========================


far_field = np.fft.fftshift(np.fft.fft2(E))
intensity = np.abs(far_field) ** 2
intensity = intensity / np.max(intensity)


# ==========================
# 结果可视化
# ==========================


plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(np.abs(E), cmap="jet")
plt.title("Near field")
plt.colorbar()

# 只显示远场中心区域，方便观察干涉条纹细节。
zoom = 20
center = N // 2
center_intensity = intensity[
    center - zoom:center + zoom,
    center - zoom:center + zoom,
]

plt.subplot(1, 2, 2)
plt.imshow(center_intensity, cmap="jet")
plt.title(f"Far field, phi={phi}")
plt.colorbar()

plt.tight_layout()
plt.show()
