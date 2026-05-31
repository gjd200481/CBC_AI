import numpy as np
import matplotlib.pyplot as plt

# =====================
# 基本参数
# =====================
N = 512
L = 10e-3
dx = L / N

x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, y)

# =====================
# 远场计算函数
# =====================
def far_field(E):
    F = np.fft.fftshift(np.fft.fft2(E))
    I = np.abs(F) ** 2
    I = I / np.max(I)
    return I

# =====================
# 1. 单缝
# =====================
slit_width = 0.5e-4
single_slit = np.abs(X) < slit_width / 2

# =====================
# 2. 双缝
# =====================
slit_width = 0.2e-4
slit_distance = 1.0e-3

double_slit = (
    (np.abs(X - slit_distance/2) < slit_width/2) |
    (np.abs(X + slit_distance/2) < slit_width/2)
)

# =====================
# 3. 圆孔
# =====================
radius = 0.8e-3
circle = X**2 + Y**2 < radius**2

# =====================
# 4. 高斯光束
# =====================
w0 = 0.8e-3
gaussian = np.exp(-(X**2 + Y**2) / w0**2)

# =====================
# 统一画图
# =====================
objects = [
    ("Single Slit", single_slit),
    ("Double Slit", double_slit),
    ("Circular Aperture", circle),
    ("Gaussian Beam", gaussian),
]

plt.figure(figsize=(10, 12))

for i, (name, E) in enumerate(objects):
    I = far_field(E)

    plt.subplot(4, 2, 2*i + 1)
    plt.imshow(np.abs(E), cmap="hot")
    plt.title(name + " Near Field")
    plt.axis("off")

    plt.subplot(4, 2, 2*i + 2)
    plt.imshow(I, cmap="hot")
    plt.title(name + " Far Field")
    plt.axis("off")

plt.tight_layout()
plt.show()