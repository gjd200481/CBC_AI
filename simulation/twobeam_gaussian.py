import numpy as np
import matplotlib.pyplot as plt


# =====================
# 参数设置
# =====================

# 网格采样点数。
N = 512

# 计算窗口尺寸，单位为米。
L = 10e-3

# 高斯光束腰斑半径，单位为米。
w0 = 0.5e-3

# 两束光中心间距，单位为米。
d = 1e-3

# 两束光之间的相位差。
phi = np.pi / 2


# =====================
# 坐标网格
# =====================

x = np.linspace(-L / 2, L / 2, N)
y = np.linspace(-L / 2, L / 2, N)
X, Y = np.meshgrid(x, y)


# =====================
# 第一束高斯光
# =====================

# 第一束光位于 x = -d/2，作为相位参考光束。
E1 = np.exp(-((X + d / 2) ** 2 + Y**2) / w0**2)


# =====================
# 第二束高斯光
# =====================

# 第二束光位于 x = +d/2，并叠加相位差 phi。
E2 = np.exp(-((X - d / 2) ** 2 + Y**2) / w0**2)
E2 = E2 * np.exp(1j * phi)


# =====================
# 近场叠加
# =====================

# 两束光相干叠加，得到总复电场。
E = E1 + E2


# =====================
# 远场计算
# =====================

# 用二维傅里叶变换模拟远场衍射图样。
far_field = np.fft.fftshift(np.fft.fft2(E))

# 探测器记录的是光强，即复电场模长平方。
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
