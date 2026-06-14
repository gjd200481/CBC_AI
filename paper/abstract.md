# Abstract

Coherent beam combining (CBC) offers a promising solution for scaling laser power beyond single-aperture limits, but precise phase sensing remains a critical challenge for achieving high synthesis efficiency. Traditional iterative optimization methods suffer from slow convergence and limited noise robustness, hindering real-time control applications. We present a dual-branch fusion network that exploits multi-plane far-field intensity observations for single-shot phase retrieval in seven-beam hexagonal CBC systems. The proposed architecture employs separate encoders for focal and befocal plane images, followed by a gated fusion mechanism that adaptively combines plane-specific features. Trained with physics-guided far-field consistency loss and dynamic noise augmentation (σ ∈ [0, 0.005]), the model achieves a Strehl ratio of 0.683 and synthesis efficiency of 0.796 on clean data, representing 67% and 49% improvements over uncompensated baselines. Under moderate noise (σ=0.02), the noise-augmented variant (Cycle 44) maintains a Strehl ratio of 0.616, demonstrating 30% superior robustness compared to the baseline model while incurring only 5% clean-data performance penalty. Advanced attribution analysis via Integrated Gradients and Grad-CAM validates that the model learns physically-consistent representations aligned with the seven-beam geometry, with balanced energy distribution between focal (48.4%) and befocal (51.6%) planes. Ablation studies reveal that the dual-branch architecture (5.77M parameters) outperforms deeper monolithic alternatives (11.34M parameters), highlighting the value of physics-informed architectural design over brute-force scaling. With 15 ms inference time per sample, the proposed approach advances learning-based CBC phase sensing toward practical deployment in high-power laser systems, free-space optical communications, and directed-energy applications.

**Keywords**: Coherent beam combining, Phase retrieval, Multi-plane imaging, Deep learning, Noise robustness, Explainable AI

---

**Word count**: 250 words (within typical 150-250 range)

**Key components**:
- Problem: CBC phase sensing challenge
- Gap: Traditional methods slow and noise-sensitive
- Solution: Dual-branch fusion + multi-plane + noise augmentation
- Results: Strehl 0.683, +30% noise robustness at σ=0.02
- Validation: IG+Grad-CAM confirm physical learning
- Impact: Practical deployment for high-power laser applications
