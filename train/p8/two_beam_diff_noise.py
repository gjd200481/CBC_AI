import numpy as np
import matplotlib.pyplot as plt
import os

# ======================
# 基本参数设置
# ======================
# 采样点数：最终会生成 N x N 的近场复振幅网格。
N = 256

# 近场计算窗口的物理尺寸，单位为米。
# 这里表示横向和纵向都在 [-L/2, L/2] 范围内采样。
L = 10e-3

# 单束高斯光的腰斑半径，单位为米。
# w0 越小，近场光斑越窄，远场衍射范围越宽。
w0 = 0.5e-3

# 两束光在 x 方向上的中心间距，单位为米。
# 第一束位于 x = -d/2，第二束位于 x = +d/2。
d = 1.5e-3

# 数据集样本数量。每个样本对应一个随机相位差 phi。
num_samples = 1000

# 保存生成数据的目录。如果目录不存在，则自动创建。
save_dir = "dataset/two_beam"
os.makedirs(save_dir, exist_ok=True)

# 构造一维坐标轴，并用 meshgrid 扩展成二维坐标矩阵。
# X 和 Y 的形状都是 (N, N)，分别表示每个采样点的横纵坐标。
x = np.linspace(-L / 2, L / 2, N)
X, Y = np.meshgrid(x, x)

# ======================
# 固定两束高斯光的位置
# ======================
# 第一束高斯光的近场复振幅。
# 这里没有显式相位项，因此默认相位为 0，作为相位参考光束。
E1 = np.exp(-((X + d / 2) ** 2 + Y ** 2) / w0**2)

# 第二束高斯光的基础振幅分布。
# 后续每个样本会给它乘上 exp(1j * phi)，从而引入随机相位差。
E2_base = np.exp(-((X - d / 2) ** 2 + Y ** 2) / w0**2)

# 用列表临时保存所有图像和标签，最后再统一转换为 numpy 数组。
images = []
labels = []

# ======================
# 生成数据集
# ======================
for i in range(num_samples):
    # 在 [-pi, pi] 中均匀随机采样两束光之间的相位差。
    phi = np.random.uniform(-np.pi, np.pi)

    # 给第二束光叠加相位因子。
    # 复数形式的电场可以同时表示振幅和相位。
    E2 = E2_base * np.exp(1j * phi)

    # 两束光在近场中相干叠加，得到总复电场。
    E = E1 + E2

    # 使用二维傅里叶变换模拟远场衍射/夫琅禾费衍射图样。
    # fft2 计算频域分布，fftshift 将零频分量移动到图像中心，便于观察和裁剪。
    Farfield = np.fft.fftshift(np.fft.fft2(E))

    # 相机或探测器测到的是光强，不是复振幅。
    # 光强等于复电场模长的平方。
    Intensity = np.abs(Farfield) ** 2

    # 将每张图归一化到最大值为 1，减少总能量尺度对训练的影响。
    Intensity = Intensity / np.max(Intensity)

    # 噪声标准差。这里加入简单的高斯噪声来模拟测量噪声。
    noise_sigma = 0.05

    # 生成与图像同尺寸的零均值高斯噪声。
    noise = np.random.normal(
        0,
        noise_sigma,
        Intensity.shape
    )

    # 将噪声叠加到归一化后的远场光强图上。
    Intensity = Intensity + noise

    # 加噪后可能出现负值或超过 1 的值。
    # clip 将数据限制在 [0, 1]，符合归一化图像的常见输入范围。
    Intensity = np.clip(
        Intensity,
        0,
        1
    )

    # 只截取远场中心区域。
    # 完整远场图为 256 x 256，而中心区域通常包含主要干涉条纹信息。
    # crop 的尺寸为 (2 * zoom) x (2 * zoom)，这里即 160 x 160。
    zoom = 80
    center = N // 2
    crop = Intensity[
        center - zoom:center + zoom,
        center - zoom:center + zoom
    ]

    # CNN 通常使用 float32 输入，既节省内存，也匹配深度学习框架默认精度。
    images.append(crop.astype(np.float32))

    # 标签不直接使用 phi，而使用 sin(phi), cos(phi)。
    # 原因是相位具有周期性：-pi 和 pi 表示同一个相位边界附近的状态。
    # 用 sin/cos 可以避免直接回归 phi 时在边界处出现不连续问题。
    labels.append([
        np.sin(phi),
        np.cos(phi)
    ])

# 将列表转换为数组：
# images 形状为 (num_samples, 160, 160)
# labels 形状为 (num_samples, 2)，第二维依次为 [sin(phi), cos(phi)]
images = np.array(images)
labels = np.array(labels, dtype=np.float32)

# 分别保存图像数据和标签数据，供训练脚本读取。
np.save(os.path.join(save_dir, "images.npy"), images)
np.save(os.path.join(save_dir, "labels.npy"), labels)

# 打印生成结果，方便确认数据维度是否符合预期。
print("Dataset generated successfully!")
print("Images shape:", images.shape)
print("Labels shape:", labels.shape)

# ======================
# 显示一个样本
# ======================
# 可视化第一张远场中心裁剪图，快速检查干涉条纹和噪声是否正常。
plt.imshow(images[0], cmap="jet")
plt.title("Sample far field")
plt.colorbar()
plt.show()
