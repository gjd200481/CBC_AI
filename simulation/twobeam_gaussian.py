import numpy as np
import matplotlib.pyplot as plt


# =====================
# 参数
# =====================

N = 512
L = 10e-3

w0 = 0.5e-3

d = 1e-3

phi = np.pi/2
# 相位差


x = np.linspace(
    -L/2,
    L/2,
    N
)

y = np.linspace(
    -L/2,
    L/2,
    N
)

X,Y = np.meshgrid(
    x,
    y
)


# =====================
# 第一束
# =====================

E1 = np.exp(
    -
    (
    (X+d/2)**2
    +
    Y**2
    )
    /
    w0**2
)


# =====================
# 第二束
# =====================

E2 = np.exp(
    -
    (
    (X-d/2)**2
    +
    Y**2
    )
    /
    w0**2
)


# 加相位

E2 =E2*np.exp(1j*phi)

# =====================
# 总场
# =====================

E =E1+E2


# =====================
# FFT
# =====================

Far =np.fft.fftshift(np.fft.fft2(E))

I =np.abs(Far)**2
I /=np.max(I)



# =====================
# 画图
# =====================

plt.figure(
figsize=(10,4)
)

plt.subplot(
1,2,1
)

plt.imshow(
np.abs(E),
cmap='hot'
)

plt.title(
"Near Field"
)



plt.subplot(
1,2,2
)

plt.imshow(
I,
cmap='hot'
)

plt.title(
"Far Field"
)


plt.show()