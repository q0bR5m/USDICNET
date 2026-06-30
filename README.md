# U-DICNet

Unsupervised DIC displacement measurement from a single speckle image pair.

Based on: ******

## Requirements

```bash
pip install -r requirements.txt
```

Tested on Python ≥ 3.7, PyTorch ≥ 1.7.

## Quick start

```bash
# default: U_DICNet (2-ch output) + patch_grad loss
python main.py --data-dir ./gauss_displacement
```

## Arguments

| argument | default | description |
|---|---|---|
| `--data-dir` | `./gauss_displacement` | directory with `re*.bmp` / `tar*.bmp` |
| `--arch` | `U_DICNet` | network: `U_DICNet` (2-ch) or `U_DICNet_shape2` (12-ch) |
| `--loss` | `patch_grad` | `patch_grad` = 2-ch + numerical gradients; `patch12` = 12-ch Taylor |
| `--pretrained` | `None` | path to checkpoint `.pth.tar` |
| `--solver` | `adam` | `adam` or `sgd` |
| `--lr` | `0.001` | initial learning rate |
| `--epochs` | `2500` | total epochs |
| `--radius` | `2` | subset radius (pixels) |
| `--order` | `2` | shape function order |
| `--norm-factor` | `10.0` | image normalisation = pixel / 255 × factor |
| `--save-dir` | same as `--data-dir` | output directory |
| `--seed` | `42` | random seed for reproducibility |
| `--auto-retry` | `False` | enable adaptive lr retry on stalled convergence |
## Convergence strategies
| argument | default | description |
| --- | --- | --- |
| `--warmup N` | `0` (off) | linear lr warmup: ramp from 1e-7 to `--lr` over N epochs |
| `--reverse-lr` | `False` | adaptive lr: raise (×2) when loss drops >10%, lower (÷2) when <0.5% per 50 epochs. Starts at 1e-6, ignores ReduceLROnPlateau |
| `--auto-retry` | `False` | auto-detect stalled/diverging runs: re-initialises model, lower lr on divergence, raise on slow convergence (max 5 retries) |
| `--early-stop` | `False` | stop training when loss is stuck on a high plateau (>0.05, improvement <0.2% over 50 epochs) |
## Output

- `dispx_*.csv` / `dispy_*.csv` — full displacement fields
- `checkpoint.pth.tar` — model checkpoint
- `result_figure.png` — visualisation (ROI colorbar excludes edge pixels)
- `train/` — TensorBoard logs

## Network variants

| Model | Output channels | Matched loss |
|---|---|---|
| `U_DICNet` | 2 (u, v) | `patch_grad` |
| `U_DICNet_shape2` | 12 (u,v + 1st/2nd derivatives) | `patch12` |




## Convergence

- The default scheduler is **ReduceLROnPlateau** . It monitors the loss and automatically halves lr when progress stalls (patience=20).
- Under normal training conditions, U‑DICNet typically converges to a loss on the order of 0.0001 or even lower, often within about 5 minutes for 2,500 epochs on an RTX 4060. However, due to random weight initialisation and variations in the input speckle image pairs, convergence can occasionally stall – and this usually becomes apparent early in training (within the first few dozen epochs), with the loss decreasing very slowly or remaining persistently high.

To address this, we recommend the following steps:

Fix a well‑tested random seed (e.g., --seed 42) to improve stability and reproducibility.
- **`--warmup 200`** — linear lr ramp from 1e-7 to 1e-4 over 200 epochs (verified to converge to ~0.006).
- **`--reverse-lr`** — adaptive lr starting at 1e-6 (verified to converge to ~0.006).
- **`--auto-retry`** — can handle both divergence and slow convergence (5 retries max)
- **`--early-stop`** — stops early when loss is stuck on a high plateau (>0.05), saving time on hopeless runs.
Load a pretrained model (--pretrained) to provide a better starting point.

Manually tune hyperparameters – for example, increase the initial learning rate (--lr 0.005), switch the solver (--solver sgd), These adjustments can often help the model escape local plateaus and resume effective convergence.

# U-DICNet

基于单张散斑图像对的无监督数字图像相关（DIC）位移测量

## 算法依据

论文：*********

## 环境依赖

### 安装依赖

```
pip install -r requirements.txt
```

### 适配版本

Python ≥ 3.7，PyTorch ≥ 1.7

## 快速启动

```
# 默认配置：U_DICNet（双通道输出）+ 分块梯度损失
python main.py --data-dir ./gauss_displacement
```

## 运行参数

表格

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| --data-dir | ./gauss\_displacement | 存放参考图 re\*.bmp、变形图 tar\*.bmp 的文件夹 |
| --arch | U\_DICNet | 网络模型：U\_DICNet（2 通道）/ U\_DICNet\_shape2（12 通道） |
| --loss | patch\_grad | 损失函数：patch\_grad = 双通道 + 数值梯度；patch12=12 通道泰勒损失 |
| --pretrained | None | 预训练权重文件.pth.tar 路径 |
| --solver | adam | 优化器：adam /sgd |
| --lr | 0.001 | 初始学习率 |
| --epochs | 2500 | 总训练轮次 |
| --radius | 2 | 匹配子集半径（像素） |
| --order | 2 | 泰勒展开阶数 |
| --norm-factor | 10.0 | 图像归一化系数：像素值 = 原图像素 / 255 × 系数 |
| --save-dir | 与 --data-dir 一致 | 结果输出目录 |
| --seed | 42 | 随机种子，保证实验可复现 |
| --auto-retry | False | 开启自适应学习率重试，解决收敛停滞 |

## 收敛优化参数

表格

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| --warmup N | 0（关闭） | 线性学习率预热：前 N 轮学习率从 1e-7 线性升至初始学习率 |
| --reverse-lr | False | 自适应学习率：每 50 轮损失下降超 10% 则 ×2；下降不足 0.5% 则 ÷2；起始 1e-6，不使用 ReduceLROnPlateau |
| --auto-retry | False | 自动检测发散 / 收敛缓慢：重新初始化模型；发散降学习率、收敛慢升学习率，最多重试 5 次 |
| --early-stop | False | 早停：损失持续高于 0.05、连续 50 轮提升不足 0.2% 时终止训练 |

## 输出文件

- dispx\_*.csv / dispy\_*.csv：完整 x、y 方向位移场
- checkpoint.pth.tar：模型断点权重
- result\_figure.png：可视化结果（感兴趣区域色标不含边缘像素）
- train/：TensorBoard 训练日志文件夹

## 网络模型版本

表格

| 模型 | 输出通道 | 配套损失函数 |
| --- | --- | --- |
| U\_DICNet | 2（u、v 位移） | patch\_grad |
| U\_DICNet\_shape2 | 12（u、v 位移 + 一、二阶位移导数） | patch12 |

## 收敛性与实验可复现说明

默认学习率调度器为 `ReduceLROnPlateau`，持续监控损失，训练停滞时自动减半学习率（等待轮数 patience=20）。

常规训练条件下，RTX 4060 显卡跑完 2500 轮仅需约 5 分钟，U-DICNet 损失通常可收敛至 0.0001 甚至更低。但受网络随机初始化、输入散斑图像对差异影响，训练偶尔会收敛停滞，该问题一般在训练前几十轮即可显现，表现为损失下降极慢或长期居高不下。

### 收敛停滞解决方案

1. 固定稳定随机种子（示例 `--seed 42`），提升训练稳定性与可复现性；
2. 开启预热：`--warmup 200`，前 200 轮学习率从 1e-7 线性提升至 1e-4，实测收敛损失约 0.006；
3. 开启自适应反向学习率：`--reverse-lr`，初始学习率 1e-6，实测收敛损失约 0.006；
4. 开启自动重试：`--auto-retry`，自动处理发散、收敛缓慢场景，最多重试 5 次；
5. 开启早停：`--early-stop`，提前终止无效训练节省算力；加载预训练权重 `--pretrained` 提供更优初始参数；
6. 手动调整超参：例如调大学习率 `--lr 0.005`、更换优化器 `--solver sgd`，帮助模型跳出局部损失平台，恢复有效收敛。
