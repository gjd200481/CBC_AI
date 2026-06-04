# Baseline 实验记录：双光束 CNN

## 实验目标

记录当前双光束相干合成相位估计模型的第一版 baseline 性能，作为后续复杂扰动、多路光束、物理约束损失和对比实验的参考基线。

## 数据集

- 图像文件：`dataset/two_beam/images_noise_0.05.npy`
- 标签文件：`dataset/two_beam/labels_noise_0.05.npy`
- 样本数：1000
- 图像尺寸：160 x 160
- 标签格式：`[sin(phi), cos(phi)]`
- 噪声类型：归一化远场光强图上叠加高斯噪声
- 噪声标准差：0.05

## 模型

- 模型文件：`models/two_beam_cnn__noise_0.05.pth`
- 模型类型：三层简单 CNN
- 输入：单通道远场光强图像
- 输出：`[sin(phi), cos(phi)]`
- 相位恢复：`phi = arctan2(sin(phi), cos(phi))`

## 评估命令

```bash
python model/demo_evaluate_two_beam_model.py --no-plot
```

## 评估结果

| 指标 | 数值 |
| --- | ---: |
| Samples | 1000 |
| RMSE(rad) | 0.02974343 |
| RMSE(deg) | 1.7041728 |
| Mean error(rad) | -0.0030478893 |
| Mean error(deg) | -0.17463118 |
| Device | cpu |

## 结论

当前简单 CNN 已经能够在双光束、固定噪声强度 `0.05` 的数据集上完成较稳定的相位估计，误差约为 `1.70 deg`。该结果可以作为后续 ResNet、动态卷积残差网络、物理约束损失和 SPGD 对比实验的初始 baseline。

## 后续动作

- 固定随机种子和训练/测试划分，提升 baseline 可复现性。
- 将评估结果自动写入 `result/metrics/`。
- 增加主瓣能量占比、Strehl 比和合成效率指标。

