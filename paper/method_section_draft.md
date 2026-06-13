# Method Section Draft

## 3. Methodology

### 3.1 Problem Formulation

We consider a seven-beam coherent beam combining (CBC) system arranged in a hexagonal array configuration. The central beam (beam_0) serves as the phase reference with fixed phase φ₀ = 0, while the six outer beams (beam_1 to beam_6) are positioned in a hexagonal pattern with unknown phase errors φ₁, ..., φ₆.

The far-field intensity distribution I(x, y) measured at the focal plane contains information about the relative phase errors. Our goal is to inverse these phase errors from a single far-field intensity image:

```
I(x, y) → CNN → {φ₁, φ₂, φ₃, φ₄, φ₅, φ₆}
```

**Phase Encoding**: To handle the periodic nature of phase (φ ∈ [-π, π]), we encode each phase as a pair of sin/cos values:

```
Label = [sin(φ₁), cos(φ₁), sin(φ₂), cos(φ₂), ..., sin(φ₆), cos(φ₆)]
```

This 12-dimensional encoding avoids discontinuities at phase boundaries and maintains smooth gradients during training.

### 3.2 Seven-Beam Optical Simulation

**Near-field Configuration**: Each beam is modeled as a Gaussian beam with identical waist radius w₀ = 0.5 mm. The central beam is positioned at the origin, while the six outer beams form a regular hexagon with inter-beam distance d = 1.5 mm.

**Near-field Complex Amplitude**: The total near-field complex amplitude is given by:

```
U_near(x, y) = Σ(k=0 to 6) A_k · exp(-(x - x_k)² + (y - y_k)²) / w₀²) · exp(i·φ_k)
```

where (x_k, y_k) are the beam centers and A_k are the amplitudes (normalized to 1 for ideal case).

**Far-field Propagation**: The far-field intensity distribution is computed via 2D Fast Fourier Transform (FFT):

```
U_far(u, v) = FFT{U_near(x, y)}
I(u, v) = |U_far(u, v)|²
```

The far-field image is cropped to a central 160×160 pixel region and normalized to [0, 1] for network input.

**Dataset Generation**: We generated a 10,000-sample training dataset with:
- Phase errors uniformly sampled from [-π, π]
- Grid size: 256×256 points
- Window size: 10 mm
- Random seed: 20260612 for reproducibility

### 3.3 Network Architecture

We propose a **Dual-Plane Fusion Phase CNN** that explicitly leverages both focal-plane and defocused-plane images for phase estimation. The architecture consists of:

**Input Branch**:
- Two separate encoders for focal plane (z = 0) and defocused plane (z = -7 cm)
- Each encoder: 3 convolutional blocks with residual connections
- Feature dimensions: 32 → 64 → 128 channels

**Fusion Module**:
- Gated fusion mechanism to dynamically weight focal/defocused features
- Gate values computed via: g = σ(Conv(concat(f_focal, f_defocus)))
- Fused features: f = g ⊙ f_focal + (1-g) ⊙ f_defocus

**Regression Head**:
- Adaptive average pooling to fixed size
- Fully connected layers: 2048 → 512 → 12
- Output: 12-dimensional sin/cos phase encoding

**Model Complexity**:
- Total parameters: 5.77M (smaller than simple dual-channel stack with 11.34M)
- Input: [B, 2, 160, 160] (batch, planes, height, width)
- Output: [B, 12] (batch, 6×2 sin/cos pairs)

### 3.4 Loss Functions

Our training objective combines four complementary loss terms:

**1. Phase Supervision Loss** (L_phase):
```
L_phase = MSE(y_pred, y_true)
```
where y_pred and y_true are the predicted and ground-truth sin/cos encodings.

**2. Far-field Consistency Loss** (L_farfield):

We decode the predicted phases and reconstruct the far-field intensity using differentiable FFT:

```
φ_pred = atan2(sin_pred, cos_pred)
U_recon = FFT{Σ A_k · G_k(x, y) · exp(i·φ_pred,k)}
L_farfield = MSE(I_recon, I_input)
```

This physics-based constraint ensures that predicted phases are consistent with the input far-field pattern.

**3. Compensation Quality Loss** (L_comp):

To directly optimize downstream compensation performance, we include:

```
L_comp = -Strehl_ratio - α · Main_lobe_energy
```

where Strehl ratio and main lobe energy are computed from the residual far-field pattern after phase compensation.

**4. Unit Circle Constraint** (L_unit):

To enforce the geometric constraint sin²φ + cos²φ = 1:

```
L_unit = MSE(sin²_pred + cos²_pred, 1)
```

**Total Loss with Warmup**:
```
L_total = L_phase + λ_phy · L_farfield + λ_comp(t) · L_comp + λ_unit · L_unit
```

where λ_comp(t) = 0 for t < t_warmup, then λ_comp(t) = 0.5 to stabilize early training.

**Hyperparameters** (determined by ablation studies in Cycle 32-34):
- λ_phy = 0.05 (Fourier physics weight)
- λ_comp = 0.5 (compensation quality weight)
- λ_unit = 0.01 (unit circle weight)
- Warmup epochs: 5

### 3.5 Training Details

**Optimizer**: Adam with cosine annealing learning rate scheduler
- Initial learning rate: 0.001
- Minimum learning rate: 1e-6
- Batch size: 32
- Total epochs: 30

**Data Split**:
- Training: 7,000 samples (70%)
- Validation: 1,500 samples (15%)
- Test: 1,500 samples (15%)

**Data Augmentation** (training only):
- Random Gaussian noise: σ ~ U(0, 0.01)
- Prevents overfitting to clean data

**Checkpoint Selection**:
- We save multiple checkpoints based on different metrics:
  - Best validation phase RMSE
  - Best validation Strehl ratio
  - Best validation main lobe energy
- For compensation-quality-oriented tasks, we use the best-Strehl checkpoint
- For phase-accuracy-oriented tasks, we use the best-RMSE checkpoint

**Hardware**: 
- Training: NVIDIA RTX 3060 GPU
- Inference: CPU (Intel i7) or GPU
- Training time: ~2 hours for 30 epochs

### 3.6 Evaluation Metrics

**Phase Accuracy Metrics**:

1. **Phase RMSE**: Circular root mean square error considering phase periodicity
```
error_k = atan2(sin(φ_pred,k - φ_true,k), cos(φ_pred,k - φ_true,k))
RMSE = sqrt(mean(error_k²))
```

2. **Phase MAE**: Mean absolute error in radians

**Compensation Quality Metrics**:

After applying predicted phase corrections φ_comp = -φ_pred, we evaluate:

1. **Strehl Ratio**: Normalized peak intensity relative to ideal coherent combining
```
Strehl = I_peak,compensated / I_peak,ideal
```

2. **Main Lobe Energy Ratio**: Energy fraction within central 3-pixel radius
```
η_main = E_mainlobe / E_total
```

3. **Synthesis Efficiency**: Square root of Strehl ratio
```
η_synthesis = sqrt(Strehl)
```

4. **Residual Phase RMSE**: RMSE of remaining phase errors after compensation

**Noise Robustness Metrics**:

We evaluate all above metrics under Gaussian noise: σ ∈ {0, 0.002, 0.005, 0.01, 0.02, 0.03}

### 3.7 Baseline Comparisons

We compare our dual-branch fusion model against:

1. **Simple CNN**: 3-layer convolutional network without physics constraints (Cycle 12)
2. **Physics-constrained CNN**: Simple CNN + far-field consistency loss (Cycle 13)
3. **Residual CNN + Physics**: Deep residual network with physics loss (Cycle 30)
4. **Simple Dual-Channel Stack**: Concatenating focal/defocused planes as input channels (Cycle 41)

All models are trained on the same 10k dataset with identical random seeds for fair comparison.

---

## Notes for Final Paper

- Add mathematical notation table
- Include system diagram (7-beam hexagonal array)
- Add network architecture diagram
- Reference related work (Hou 2019, Mills 2022, Xie 2024)
- Discuss differences: simulation vs experimental data
- Explain choice of defocus distance (-7 cm)
