# Cycle 11：7 光束基础仿真与 smoke 数据集

## 1. 本周期目标

本周期开始将项目主线从双光束验证正式推进到 7 光束多路相干合成。目标是建立 7 光束六边形阵列的基础仿真模块，并生成一个小规模 smoke 数据集，用于检查远场图像、相位标签和正余弦编码是否正确。

## 2. 相位定义

7 光束系统采用中心参考光束 + 外圈 6 路光束的六边形阵列：

```text
beam_0: center reference, phase = 0
beam_1 ... beam_6: outer ring, phase = phi_1 ... phi_6
label = [sin(phi_1), cos(phi_1), ..., sin(phi_6), cos(phi_6)]
```

这种定义可以消除全局相位不唯一问题，使网络只预测 6 路相对相位误差。

## 3. 新增文件

### `simulation/common/multi_beam_core.py`

该文件是 7 光束主线的公共物理仿真模块，主要包含：

- 生成中心 + 六边形外圈 7 光束坐标。
- 构造 7 光束近场复振幅。
- 使用 FFT 计算归一化远场光强。
- 将 6 路相对相位编码为 12 维 `sin/cos` 标签。
- 从 12 维标签反解相位，用于指标验证。
- 支持探测器高斯噪声、振幅失配和位置偏移接口。

### `simulation/static/generate_seven_beam_dataset.py`

该文件是 7 光束静态数据集生成脚本，输出远场图像、12 维标签、原始 6 路相位和 JSON 配置文件。

## 4. smoke 数据集命令

```powershell
python simulation\static\generate_seven_beam_dataset.py --num-samples 32 --noise-sigma 0 --num-points 256 --window-size 0.01 --waist 0.0005 --beam-distance 0.0015 --crop-size 160 --seed 20260611 --output-dir dataset\seven_beam\cycle11_smoke --prefix smoke_clean
```

## 5. 输出文件

```text
dataset/seven_beam/cycle11_smoke/images_smoke_clean.npy
dataset/seven_beam/cycle11_smoke/labels_smoke_clean.npy
dataset/seven_beam/cycle11_smoke/phases_smoke_clean.npy
dataset/seven_beam/cycle11_smoke/config_smoke_clean.json
```

数据集文件不提交到 Git，只保留代码、配置说明、实验日志和指标表。

## 6. 数值检查

| 指标 | 数值 |
| --- | --- |
| images shape | `(32, 160, 160)` |
| labels shape | `(32, 12)` |
| phases shape | `(32, 6)` |
| image min | `8.9119559033e-25` |
| image max | `1.0` |
| image mean | `0.0010455627` |
| phase min | `-3.1378977299` |
| phase max | `3.1042106152` |
| max label error | `0.0` |
| max phase decode error | `5.9604644775e-08` |
| centers shape | `(7, 2)` |

## 7. 结论

7 光束基础仿真链路已经跑通。当前模块能够生成六边形阵列近场、远场光强图像、6 路相对相位和 12 维正余弦标签。标签重构误差为 0，相位解码最大误差约为 `5.96e-08 rad`，满足后续训练数据生成要求。

下一周期建议进入 7 光束普通 CNN baseline：先生成较小规模主数据集，训练第一版 12 维输出 CNN，并输出整体相位 RMSE 与逐通道 RMSE。
