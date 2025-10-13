# RKNN Model Convert

ONNX 模型转换为 RKNN 格式的工具，支持 Rockchip 各系列 NPU 平台。

## ✨ 功能特性

- 🚀 支持将 ONNX 模型转换为 RKNN 格式
- 🎯 支持多种 Rockchip NPU 平台（RK3562/RK3566/RK3568/RK3576/RK3588/RV1126/RV1109/RK1808）
- ⚡ 支持 INT8/UINT8 量化和 FP16 浮点模型
- 🎨 基于 loguru 的彩色日志输出
- 📁 智能输出路径处理（自动创建目录、自动推断文件名）
- ⚙️ 灵活的模型配置选项（归一化参数、量化数据集等）

## 📋 环境要求

- Python >= 3.12, < 3.13
- rknn-toolkit2 2.3.2
- loguru
- numpy
- opencv-python-headless

## 🔧 安装

### 使用 uv（推荐）

```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 同步依赖
uv sync
```

### 使用 pip

```bash
pip install -r requirements.txt
```

## 📖 使用方法

### 基本用法

```bash
python convert.py <model_path> <platform> [options]
```

### 查看帮助

```bash
python convert.py -h
```

## 📝 参数说明

### 必需参数

| 参数 | 说明 |
|------|------|
| `model` | 输入的 ONNX 模型文件路径 |
| `platform` | 目标 RKNN 平台，可选值：`rk3562`, `rk3566`, `rk3568`, `rk3576`, `rk3588`, `rv1126b`, `rv1109`, `rv1126`, `rk1808` |

### 可选参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--dtype` | 模型数据类型：`i8`（INT8）、`u8`（UINT8）、`fp`（浮点） | `i8` |
| `-o`, `--output` | 输出路径（文件或目录） | `yolo11.rknn` |
| `--dataset` | 量化数据集路径 | `datasets/COCO/coco_subset_20.txt` |
| `--no-quant` | 禁用量化 | - |
| `--mean-values` | RGB 均值（用于归一化） | `0 0 0` |
| `--std-values` | RGB 标准差（用于归一化） | `255 255 255` |
| `--verbose` | 启用详细输出 | - |
| `--custom-hybrid` | 自定义混合量化配置文件路径（YAML 格式） | - |
| `--auto-hybrid-quant` | 启用自动混合量化（适用于老平台） | - |

### 平台与数据类型兼容性

| 平台 | 支持的数据类型 |
|------|---------------|
| rk3562, rk3566, rk3568, rk3576, rk3588, rv1126b | `i8`, `fp` |
| rv1109, rv1126, rk1808 | `u8`, `fp` |

## 💡 使用示例

### 1. 基本转换（使用默认设置）

```bash
python convert.py models/yolo11n.onnx rk3588
```

输出：`yolo11.rknn`（INT8 量化）

### 2. 指定输出目录（自动创建并使用输入模型名）

```bash
python convert.py models/yolo11n.onnx rk3588 -o outputs
```

输出：`outputs/yolo11n.rknn`（目录不存在会自动创建）

### 3. 指定完整输出路径

```bash
python convert.py models/yolo11n.onnx rk3588 -o outputs/my_model.rknn
```

输出：`outputs/my_model.rknn`

### 4. 转换为浮点模型（不量化）

```bash
python convert.py model.onnx rk3588 --dtype fp --no-quant
```

### 5. 使用自定义量化数据集

```bash
python convert.py model.onnx rk3588 \
  --dataset datasets/imagenet/ILSVRC2012_img_val_samples/dataset_20.txt
```

### 6. 自定义归一化参数

```bash
python convert.py model.onnx rk3588 \
  --mean-values 127.5 127.5 127.5 \
  --std-values 127.5 127.5 127.5
```

### 7. 启用详细输出

```bash
python convert.py model.onnx rk3588 --verbose
```

### 8. 转换为 UINT8 格式（适用于 RV1109）

```bash
python convert.py model.onnx rv1109 --dtype u8
```

### 9. 使用自定义混合量化（适用于精度敏感模型）

```bash
# 创建或编辑 custom_hybrid.yaml 配置文件
python convert.py models/yolov8_pose.onnx rk3588 --custom-hybrid custom_hybrid_example.yaml
```

### 10. 使用自动混合量化（适用于老平台）

```bash
python convert.py model.onnx rv1109 --auto-hybrid-quant
```

## 🎨 混合量化（Hybrid Quantization）

混合量化是一种高级量化策略，可以让模型的某些关键层保持浮点精度，而其他层使用 INT8/UINT8 量化。这对于精度敏感的模型（如姿态估计、关键点检测等）特别有用。

### 什么时候使用混合量化？

- ✅ 模型在全量化后精度损失严重
- ✅ 检测/分类/姿态估计的关键输出层需要更高精度
- ✅ 某些层对量化误差特别敏感

### 自定义混合量化

#### 1. 创建配置文件

创建一个 YAML 文件（如 `custom_hybrid.yaml`）：

```yaml
custom_hybrid:
  # 定义需要保持浮点精度的层对
  # 格式：[[起始层, 结束层], [起始层, 结束层], ...]
  - ['/model.22/cv4.0/cv4.0.0/act/Mul_output_0', '/model.22/Concat_6_output_0']
  - ['/model.22/cv4.1/cv4.1.0/act/Mul_output_0', '/model.22/Concat_6_output_0']
  - ['/model.22/cv4.2/cv4.2.0/act/Mul_output_0', '/model.22/Concat_6_output_0']
```

#### 2. 查找层名称

使用 [Netron](https://netron.app/) 打开你的 ONNX 模型，找到需要保持浮点精度的层名称。

#### 3. 运行转换

```bash
python convert.py model.onnx rk3588 --custom-hybrid custom_hybrid.yaml -o outputs/
```

#### 4. 生成的中间文件

混合量化会生成以下中间文件（用于两步量化）：
- `model_name.model` - 第一步生成的模型文件
- `model_name.data` - 第一步生成的配置文件
- `model_name.quantization.cfg` - 第一步生成的量化配置文件

这些文件会自动用于第二步的量化过程。

### 自动混合量化

对于某些老平台（rv1109, rv1126, rk1808），可以使用自动混合量化：

```bash
python convert.py model.onnx rv1109 --auto-hybrid-quant
```

这会让 RKNN 工具链自动分析并选择合适的层进行混合量化。

## 📂 目录结构

```
rknn_model_convert/
├── convert.py              # 主转换脚本
├── pyproject.toml          # 项目配置文件
├── uv.lock                 # 依赖锁定文件
├── README.md               # 本文档
├── datasets/               # 量化数据集目录
│   ├── COCO/              # COCO 数据集
│   │   ├── coco_subset_20.txt
│   │   └── subset/
│   ├── imagenet/          # ImageNet 数据集
│   │   └── ILSVRC2012_img_val_samples/
│   ├── LPRNET/            # 车牌识别数据集
│   └── PPOCR/             # OCR 数据集
├── models/                 # 输入模型目录
└── outputs/                # 输出模型目录
```

## 🎯 输出路径规则

脚本会智能处理输出路径：

1. **没有后缀名** → 视为目录
   - 自动创建目录（如果不存在）
   - 使用输入模型的文件名，添加 `.rknn` 后缀
   ```bash
   -o outputs/rk3588  → outputs/rk3588/model_name.rknn
   ```

2. **有后缀名** → 视为文件路径
   - 自动创建父目录（如果不存在）
   - 使用指定的文件名
   ```bash
   -o outputs/my_model.rknn  → outputs/my_model.rknn
   ```

## 📊 日志输出

脚本使用 loguru 提供彩色日志输出，便于追踪转换过程：

```
2025-10-13 12:34:56 | INFO     | Configuring model
2025-10-13 12:34:56 | INFO     | Platform: rk3588
2025-10-13 12:34:56 | INFO     | Data type: i8
2025-10-13 12:34:56 | INFO     | Quantization: enabled
2025-10-13 12:34:57 | SUCCESS  | Model configuration completed
2025-10-13 12:34:57 | INFO     | Loading ONNX model: model.onnx
2025-10-13 12:35:00 | SUCCESS  | Model loaded successfully
2025-10-13 12:35:00 | INFO     | Building RKNN model
2025-10-13 12:35:30 | SUCCESS  | Model built successfully
2025-10-13 12:35:30 | INFO     | Exporting RKNN model to: output.rknn
2025-10-13 12:35:31 | SUCCESS  | Model exported successfully
2025-10-13 12:35:31 | SUCCESS  | Successfully converted model.onnx to output.rknn
```

## ⚠️ 注意事项

1. **量化数据集**：量化模型需要提供代表性数据集，数据集质量会影响模型精度
2. **归一化参数**：确保归一化参数与训练时保持一致
3. **平台选择**：选择正确的目标平台，不同平台的 NPU 架构可能不同
4. **内存占用**：转换大模型时可能需要较大内存
5. **混合量化**：
   - 使用混合量化时，需要先分析模型找到精度敏感层
   - 混合量化会生成中间文件（.model, .data, .quantization.cfg），这些文件可以删除
   - 过多的浮点层会降低 NPU 加速效果，建议只在必要时使用
6. **YAML 配置**：custom_hybrid 配置文件需要使用 UTF-8 编码

## 📄 许可证

本项目版本 8.0.5

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📮 联系方式

如有问题，请提交 Issue。

