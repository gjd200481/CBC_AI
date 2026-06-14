# 基于双分支融合网络的多平面相位反演方法研究
## 用于七光束相干合成系统

---

## 摘要

相干合成(CBC)技术通过多束激光的相干叠加突破单孔径功率限制，但精确的相位感知仍是实现高合成效率的关键挑战。传统迭代优化方法收敛缓慢且噪声鲁棒性有限，难以满足实时控制需求。本文提出一种双分支融合网络架构，利用焦平面和焦前平面的远场光强观测实现七光束六边形阵列的单次相位反演。该架构为两个观测平面分别设计独立编码器，并通过门控融合机制自适应结合平面特定特征。模型采用物理引导的远场一致性损失和动态噪声增强训练(σ ∈ [0, 0.005])，在干净数据上达到0.683的Strehl比和0.796的合成效率，相比补偿前分别提升67%和49%。在中等噪声条件下(σ=0.02)，噪声增强模型(Cycle 44)保持0.616的Strehl比，相比基线模型鲁棒性提升30%，仅付出5%的干净数据性能代价。通过Integrated Gradients和Grad-CAM的高级归因分析验证了模型学习到与七光束几何结构一致的物理特征，焦平面和焦前平面的能量分布接近均衡(48.4% vs 51.6%)。消融实验表明，双分支架构(5.77M参数)优于更深的单体网络(11.34M参数)，凸显了物理启发式架构设计相对暴力扩展的优势。模型单样本推理时间为15毫秒，使得基于学习的CBC相位感知向高功率激光系统、自由空间光通信和定向能应用的实际部署迈进了重要一步。

**关键词**: 相干合成；相位反演；多平面成像；深度学习；噪声鲁棒性；可解释人工智能

---

## 1. 引言

### 1.1 研究背景

相干合成(Coherent Beam Combining, CBC)技术通过将多束激光相干叠加，能够突破单孔径激光器的功率和亮度限制，在工业加工、激光武器、自由空间光通信等领域具有广泛应用前景。然而，实现高效的相干合成需要精确控制各光束之间的相对相位，使其在远场形成相干叠加。相位误差的存在会导致远场光斑分裂、主瓣能量下降，严重影响合成效率和光束质量。

### 1.2 传统方法的局限性

传统相位感知方法主要包括：

1. **随机并行梯度下降(SPGD)**: 通过迭代优化寻找最优相位，但收敛速度慢(数百次迭代)，难以应对动态扰动

2. **Gerchberg-Saxton算法**: 基于傅里叶变换的迭代重建，但对噪声敏感，容易陷入局部最优

3. **干涉测量**: 需要额外的参考光路，系统复杂度高

这些方法的共同局限是速度慢、鲁棒性差，无法满足千赫兹级反馈控制的实时需求。

### 1.3 深度学习方法的兴起

近年来，基于深度学习的相位反演方法开始出现：

- **Hou等人(2019)**: 首次将CNN应用于CBC相位反演，但仅使用单个远场平面，RMSE约1.2 rad
- **Mills等人(2022)**: 提出单步相位优化方法，但仍需多次前向传播，推理速度慢
- **Xie等人(2024)**: 单平面CNN识别5光束系统相位，报告Strehl比约0.55

现有方法的不足：
1. 仅利用单平面信息，未充分挖掘多视角观测的互补性
2. 缺乏对噪声鲁棒性的系统研究
3. 缺少对模型学习机制的可解释性验证

### 1.4 本文贡献

针对上述问题，本文提出：

1. **双分支融合架构**: 首次在CBC领域利用焦平面+焦前平面的多平面观测，通过独立编码器和门控融合实现信息互补

2. **噪声增强训练策略**: 通过动态噪声注入(σ~Uniform(0, 0.005))，在σ=0.02噪声下实现+30%的鲁棒性提升，仅付出5%的干净数据性能代价

3. **高级归因分析**: 首次在CBC领域应用Integrated Gradients和Grad-CAM，验证模型学习到物理一致的特征表示

4. **完整的实验验证**: 包括基线对比、噪声鲁棒性测试、消融实验和负结果报告，为领域提供系统的实证研究

**论文组织结构**: 第2节综述相关工作；第3节详述方法；第4节展示实验结果；第5节讨论发现并分析局限性；第6节总结全文。

---

## 2. 相关工作

### 2.1 传统相位反演方法

**基于优化的方法**:
- SPGD算法通过随机扰动各光束相位，根据远场功率反馈迭代优化
- 优点：实现简单，无需精确建模
- 缺点：收敛慢(>100次迭代)，易受局部最优困扰

**基于干涉的方法**:
- 利用参考光束与信号光束干涉测量相对相位
- 优点：直接测量，精度高
- 缺点：需要稳定的参考光路，系统复杂

### 2.2 深度学习在CBC中的应用

**Hou等人(2019)**:
- 方法：单平面远场图像 → CNN → 相位预测
- 结果：RMSE约1.2 rad
- 局限：单平面信息有限

**Mills等人(2022)**:
- 方法：单步相位优化，基于可微分物理模型
- 优点：结合物理约束
- 局限：每个样本需要多次前向传播，速度慢

**Xie等人(2024)**:
- 方法：单平面CNN识别5光束相位
- 结果：Strehl比约0.55
- 局限：单平面，较简单阵列配置

### 2.3 多视图深度学习

在其他领域，多视图融合已被证明有效：
- **医学成像**: CT多角度投影重建
- **3D重建**: 多视角立体匹配
- **物体识别**: RGB-D融合

**启示**: 不同视角提供互补信息，融合策略至关重要

### 2.4 可解释AI在物理系统中的应用

- **气候模拟**: 验证深度学习模型是否学习物理规律
- **材料科学**: 确保预测基于合理物理机制
- **医学诊断**: 确保临床可接受性

**关键问题**: 模型是学习了物理还是数据集偏差？

**本文方法**: 使用IG和Grad-CAM验证物理一致性

---

## 3. 方法

### 3.1 问题建模

#### 3.1.1 七光束六边形阵列

考虑7束激光呈六边形排列，中心光束beam_0相位固定为0作为参考，外围6束beam_1到beam_6的相对相位为待估计量 φ₁, φ₂, ..., φ₆ ∈ [-π, π]。

物理参数：
- 波长: λ = 1064 nm (典型光纤激光器)
- 光束腰斑: w₀ = 0.5 mm
- 光束间距: d = 1.5 mm
- 远场观测距离: z = 10 m

#### 3.1.2 多平面观测模型

我们采集两个平面的远场光强：

1. **焦平面** (z = 0): I_focal(x, y)
2. **焦前平面** (z = -0.07 m): I_befocal(x, y)

物理直觉：
- 焦平面提供最佳聚焦信息，主瓣结构清晰
- 焦前平面引入相位调制信息，对小相位误差敏感
- 两者互补，共同约束相位解空间

#### 3.1.3 Sin/Cos相位编码

为避免相位周期性带来的不连续性，我们采用sin/cos编码：

**输出向量**: y = [sin φ₁, cos φ₁, sin φ₂, cos φ₂, ..., sin φ₆, cos φ₆] ∈ ℝ¹²

**优点**:
1. 自然处理 φ ∼ φ + 2π 的周期性
2. 避免 ±π 边界的不连续
3. 保留完整的相位信息

**相位重建**: φᵢ = atan2(sin φᵢ, cos φᵢ)

### 3.2 双分支融合网络架构

#### 3.2.1 整体设计哲学

**关键洞察**: 焦平面和焦前平面是物理上不同的观测，应由独立的特征提取器处理，再通过学习的融合机制结合。

**架构对比**:
- **简单堆叠** (Cycle 41): 两平面作为2通道输入 → 单个编码器
- **双分支融合** (Cycle 42, 本文): 两平面 → 两个独立编码器 → 门控融合

#### 3.2.2 焦平面编码器

```
Input: I_focal ∈ ℝ^(160×160)

Encoder_focal:
  Conv2D(1→32, 3×3) → ReLU → MaxPool(2×2)
  Conv2D(32→64, 3×3) → ReLU → MaxPool(2×2)
  Conv2D(64→128, 3×3) → ReLU → MaxPool(2×2)
  Conv2D(128→256, 3×3) → ReLU → AdaptiveAvgPool
  
Output: f_focal ∈ ℝ^256
```

#### 3.2.3 焦前平面编码器

```
Input: I_befocal ∈ ℝ^(160×160)

Encoder_befocal:
  (同样的卷积结构，但参数独立)
  
Output: f_befocal ∈ ℝ^256
```

#### 3.2.4 门控融合机制

```python
# 计算门控权重
gate = Sigmoid(Linear(concat(f_focal, f_befocal), 2))
w_focal, w_befocal = Softmax(gate)

# 加权融合
f_fused = w_focal * f_focal + w_befocal * f_befocal

# 相位预测头
y_pred = Linear(f_fused, 12)  # [sin φ₁, cos φ₁, ..., sin φ₆, cos φ₆]
```

**设计优点**:
1. 门控权重 w_focal, w_befocal 根据输入自适应调整
2. 不同样本可能依赖不同平面(高方差 → 动态融合)
3. 端到端可学习，无需手工设计权重

**参数量**: 5.77M (相比单体深度网络11.34M减少49%)

### 3.3 物理引导的训练

#### 3.3.1 相位损失

基础MSE损失在sin/cos空间：

$$
\mathcal{L}_{\text{phase}} = \frac{1}{12} \sum_{i=1}^{6} \left[ (\sin\hat{\phi}_i - \sin\phi_i)^2 + (\cos\hat{\phi}_i - \cos\phi_i)^2 \right]
$$

其中 $\hat{\phi}_i$ 是预测相位，$\phi_i$ 是真实相位。

#### 3.3.2 远场一致性损失

利用可微分的七光束傅里叶光学模型：

```
SevenBeamFourierOptics:
  Input: [sin φ₁, cos φ₁, ..., sin φ₆, cos φ₆]
  
  Step 1: 重建相位 φ_i = atan2(sin φ_i, cos φ_i)
  
  Step 2: 生成近场复振幅
    E_near(x, y) = Σ A_i exp(iφ_i) exp(-(x-x_i)²+(y-y_i)²/w₀²)
  
  Step 3: 傅里叶变换到远场
    E_far = FFT2D(E_near)
  
  Step 4: 计算光强
    I_reconstructed = |E_far|²
  
  Output: I_reconstructed ∈ ℝ^(160×160)
```

远场一致性损失：

$$
\mathcal{L}_{\text{farfield}} = \text{MSE}(I_{\text{reconstructed}}, I_{\text{focal}})
$$

**物理意义**: 预测的相位应能通过物理模型重建出观测到的远场图样

#### 3.3.3 补偿质量损失

评估补偿后的光束质量：

$$
\mathcal{L}_{\text{comp}} = -\text{Strehl}(\phi_{\text{predicted}})
$$

其中Strehl比近似为：

$$
\text{Strehl} \approx \exp(-\sigma_\phi^2)
$$

$\sigma_\phi^2$ 是相位方差。

#### 3.3.4 总损失函数

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{phase}} + \lambda_{\text{phy}} \mathcal{L}_{\text{farfield}} + \lambda_{\text{comp}} \mathcal{L}_{\text{comp}}
$$

**超参数**: $\lambda_{\text{phy}} = 0.05$, $\lambda_{\text{comp}} = 0.5$

**训练细节**:
- 优化器: Adam, lr=1e-3, weight_decay=1e-5
- 学习率调度: CosineAnnealing, T_max=30, eta_min=1e-6
- Batch size: 32
- Epochs: 30 (约12分钟 on RTX 3060)

### 3.4 动态噪声增强策略

#### 3.4.1 动机

Cycle 42基线模型在σ=0.002处出现局部性能下降（Strehl从0.683降至0.625），尽管噪声进一步增大时性能单调下降。

**假设**: 模型过拟合到干净训练数据(σ=0)，对轻微扰动(σ=0.002)脆弱。在中等噪声(σ=0.005)时反而激活更鲁棒的低频特征。

#### 3.4.2 训练时噪声注入

在每个训练迭代中：

```python
for epoch in range(30):
    for images, labels in train_loader:
        # 动态采样噪声强度
        sigma = np.random.uniform(0.0, 0.005)
        
        # 添加高斯噪声
        noise = torch.randn_like(images) * sigma
        noisy_images = images + noise
        
        # 前向传播
        outputs = model(noisy_images)
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
```

**设计要点**:
- 噪声范围 σ ∈ [0, 0.005] 覆盖问题区域 σ=0.002
- 每次迭代独立采样，避免模型记忆特定噪声模式
- 对验证集和测试集不添加噪声（评估真实泛化能力）

#### 3.4.3 预期效果

1. **σ=0.002退化消除**: 平滑性能曲线
2. **整体鲁棒性提升**: 正则化效应推广到更高噪声
3. **干净数据代价可控**: 轻微性能下降(<10%)

### 3.5 高级归因分析方法

#### 3.5.1 Integrated Gradients (IG)

**简单梯度的问题**:
- 梯度饱和区域(ReLU为0, Sigmoid/Tanh饱和)显示零梯度 ≠ 不重要
- 高频噪声掩盖真实结构

**IG解决方案**: 沿从基线(零图像)到输入的路径积分梯度

$$
\text{IG}_i(x) = (x_i - x'_i) \int_{\alpha=0}^{1} \frac{\partial f(x' + \alpha(x - x'))}{\partial x_i} d\alpha
$$

**数值实现**:
```python
def integrated_gradients(model, input, target_channel, steps=50):
    baseline = torch.zeros_like(input)
    alphas = torch.linspace(0, 1, steps+1)
    
    integrated_grads = 0
    for alpha in alphas:
        interpolated = baseline + alpha * (input - baseline)
        interpolated.requires_grad_(True)
        
        output = model(interpolated)
        target_output = output[0, target_channel]
        
        model.zero_grad()
        target_output.backward(retain_graph=True)
        
        integrated_grads += interpolated.grad / (steps + 1)
    
    integrated_grads *= (input - baseline)
    return integrated_grads.detach()
```

**优点**:
1. 满足敏感性公理: 不同输入 → 非零归因
2. 实现不变性: 功能等价的模型产生相同归因
3. 理论保证: 可靠性高于简单梯度

#### 3.5.2 Grad-CAM

**目标**: 可视化卷积层学到的空间特征

**方法**: 对目标通道的梯度在卷积特征图上加权平均

$$
\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A_{ij}^k}
$$

$$
L^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)
$$

其中 $A^k$ 是第k个卷积特征图，$\alpha_k^c$ 是其对目标输出通道c的重要性权重。

**实现**:
```python
class GradCAM:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.gradients = None
        self.activations = None
        self._register_hooks(target_layer_name)
    
    def compute(self, input, target_channel):
        # 前向传播
        output = self.model(input)
        target_output = output[0, target_channel]
        
        # 反向传播
        self.model.zero_grad()
        target_output.backward()
        
        # 计算权重（全局平均池化梯度）
        weights = torch.mean(self.gradients, dim=(2,3), keepdim=True)
        
        # 加权组合激活
        cam = torch.sum(weights * self.activations, dim=1)
        cam = F.relu(cam)
        
        # 上采样到输入尺寸
        cam = F.interpolate(cam, size=input.shape[2:], mode='bilinear')
        
        # 归一化
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam[0, 0]
```

**优点**:
1. 空间定位: 显示哪些图像区域最重要
2. 层级特定: 可视化不同深度的特征
3. 直观易懂: 热图叠加在原图上

#### 3.5.3 联合使用IG和Grad-CAM

**互补性**:
- IG: 像素级归因，全局能量分布
- Grad-CAM: 空间区域定位，层级特征

**验证策略**:
1. IG能量差异 → 通道重要性排序
2. Grad-CAM热图 → 空间注意力是否对齐物理结构
3. 双平面IG能量对比 → 验证融合机制有效性

---

*[由于长度限制，Results和Discussion章节将在下一部分继续]*
