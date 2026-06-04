import numpy as np
import matplotlib.pyplot as plt
import os


# ======================
# Parameters
# ======================


N = 256
L = 10e-3
w0 = 0.5e-3
d = 1.5e-3
num_samples = 1000

save_dir = "dataset/two_beam"
os.makedirs(save_dir, exist_ok=True)

x = np.linspace(-L / 2, L / 2, N)
X, Y = np.meshgrid(x, x)


# ======================
# Fixed beam positions
# ======================


E1 = np.exp(-((X + d / 2) ** 2 + Y ** 2) / w0**2)
E2_base = np.exp(-((X - d / 2) ** 2 + Y ** 2) / w0**2)

images = []
labels = []


# ======================
# Generate dataset
# ======================


for i in range(num_samples):
    phi = np.random.uniform(-np.pi, np.pi)

    E2 = E2_base * np.exp(1j * phi)
    E = E1 + E2

    far_field = np.fft.fftshift(np.fft.fft2(E))
    intensity = np.abs(far_field) ** 2
    intensity = intensity / np.max(intensity)

    # 截取中心 160x160 区域，避免保存完整远场图带来的额外开销。
    zoom = 80
    center = N // 2
    crop = intensity[
        center - zoom:center + zoom,
        center - zoom:center + zoom,
    ]

    images.append(crop.astype(np.float32))

    # 标签使用 [sin(phi), cos(phi)]，用于处理相位周期性。
    labels.append([
        np.sin(phi),
        np.cos(phi),
    ])

images = np.array(images)
labels = np.array(labels, dtype=np.float32)

np.save(os.path.join(save_dir, "images.npy"), images)
np.save(os.path.join(save_dir, "labels.npy"), labels)

print("Dataset generated successfully!")
print("Images shape:", images.shape)
print("Labels shape:", labels.shape)


# ======================
# Show one sample
# ======================


plt.imshow(images[0], cmap="jet")
plt.title("Sample far field")
plt.colorbar()
plt.show()
