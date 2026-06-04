import numpy as np
import matplotlib.pyplot as plt
import os


# ======================
# 基本参数设置
# ======================


N = 256
L = 10e-3
w0 = 0.5e-3
d = 1.5e-3
num_samples = 1000
noise_sigma = 0.05

save_dir = "dataset/two_beam"
os.makedirs(save_dir, exist_ok=True)

x = np.linspace(-L / 2, L / 2, N)
X, Y = np.meshgrid(x, x)


# ======================
# 固定两束高斯光的位置
# ======================


# 第一束光作为相位参考，默认相位为 0。
E1 = np.exp(-((X + d / 2) ** 2 + Y ** 2) / w0**2)

# 第二束光的基础振幅分布，后续每个样本乘以 exp(1j * phi)。
E2_base = np.exp(-((X - d / 2) ** 2 + Y ** 2) / w0**2)

images = []
labels = []


# ======================
# 生成数据集
# ======================


for i in range(num_samples):
    # 在 [-pi, pi] 中随机采样两束光之间的相位差。
    phi = np.random.uniform(-np.pi, np.pi)

    E2 = E2_base * np.exp(1j * phi)
    E = E1 + E2

    # 用 FFT 模拟夫琅禾费远场衍射。
    far_field = np.fft.fftshift(np.fft.fft2(E))
    intensity = np.abs(far_field) ** 2
    intensity = intensity / np.max(intensity)

    # 加入零均值高斯噪声，模拟探测器测量噪声。
    noise = np.random.normal(0, noise_sigma, intensity.shape)
    intensity = intensity + noise
    intensity = np.clip(intensity, 0, 1)

    # 截取远场中心 160x160 区域，保留主要干涉条纹信息。
    zoom = 80
    center = N // 2
    crop = intensity[
        center - zoom:center + zoom,
        center - zoom:center + zoom,
    ]

    images.append(crop.astype(np.float32))

    # 用 sin/cos 编码相位，避免 -pi/pi 边界处的不连续问题。
    labels.append([np.sin(phi), np.cos(phi)])

images = np.array(images)
labels = np.array(labels, dtype=np.float32)

np.save(os.path.join(save_dir, "images_noise_0.05.npy"), images)
np.save(os.path.join(save_dir, "labels_noise_0.05.npy"), labels)

print("Dataset generated successfully!")
print("Images shape:", images.shape)
print("Labels shape:", labels.shape)


# ======================
# 显示一个样本
# ======================


plt.imshow(images[0], cmap="jet")
plt.title("Sample far field")
plt.colorbar()
plt.show()
