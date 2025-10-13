# Changelog

## [最新版本] - 2025-10-13

### ✨ 新增功能

#### 1. 混合量化支持（Hybrid Quantization）

- **自定义混合量化** (`--custom-hybrid`)
  - 通过 YAML 配置文件指定需要保持浮点精度的层
  - 使用两步量化过程（`hybrid_quantization_step1` 和 `hybrid_quantization_step2`）
  - 适用于精度敏感的模型（如姿态估计、关键点检测等）
  - 示例：`python convert.py model.onnx rk3588 --custom-hybrid config.yaml`

- **自动混合量化** (`--auto-hybrid-quant`)
  - 让 RKNN 工具链自动分析并选择合适的层进行混合量化
  - 适用于老平台（rv1109, rv1126, rk1808）
  - 示例：`python convert.py model.onnx rv1109 --auto-hybrid-quant`

#### 2. 代码重构

- **使用 argparse 替代手动参数解析**
  - 自动生成帮助文档
  - 更好的参数验证和错误提示
  - 支持短参数和长参数（如 `-o` 和 `--output`）

- **使用 loguru 替代 print**
  - 彩色日志输出（INFO/SUCCESS/ERROR）
  - 带时间戳的日志记录
  - 统一的日志格式
  - 更专业的日志系统

#### 3. 智能输出路径处理

- **无后缀名自动识别为目录**
  - 自动创建不存在的目录（包括多级目录）
  - 从输入模型文件名自动生成输出文件名
  - 示例：`-o outputs/rk3588` → `outputs/rk3588/model.rknn`

- **有后缀名识别为文件路径**
  - 自动创建父目录
  - 使用指定的文件名
  - 示例：`-o outputs/my_model.rknn` → `outputs/my_model.rknn`

### 📄 新增文件

1. **custom_hybrid_example.yaml**
   - YOLOv8 Pose 模型的混合量化配置示例
   - 包含详细的注释说明
   - 展示了如何定义需要保持浮点精度的层对

2. **README.md**
   - 完整的项目文档
   - 详细的参数说明表格
   - 10+ 个使用示例
   - 混合量化专题说明
   - 注意事项和最佳实践

3. **CHANGELOG.md**
   - 版本变更记录
   - 新功能说明

### 🔧 改进

- 更详细的日志输出，显示配置信息
- 平台与数据类型的自动验证
- 更好的错误处理和提示
- YAML 配置文件格式验证
- 中间文件自动管理

### 📋 新增参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `--custom-hybrid` | 文件路径 | 自定义混合量化配置文件（YAML 格式） |
| `--auto-hybrid-quant` | 标志 | 启用自动混合量化 |

### 🎯 使用场景

#### 标准量化（默认）
```bash
python convert.py model.onnx rk3588
```

#### 自定义混合量化（推荐用于 YOLOv8 Pose）
```bash
python convert.py yolov8_pose.onnx rk3588 --custom-hybrid custom_hybrid_example.yaml
```

#### 自动混合量化（老平台）
```bash
python convert.py model.onnx rv1109 --auto-hybrid-quant
```

### ⚠️ 重要提示

1. **混合量化需要量化数据集**
   - 确保 `--dataset` 参数正确
   - 不能与 `--no-quant` 同时使用

2. **中间文件**
   - 混合量化会生成 `.model`, `.data`, `.quantization.cfg` 文件
   - 这些文件可以在转换完成后删除

3. **精度 vs 性能**
   - 过多的浮点层会降低 NPU 加速效果
   - 建议只在必要时使用混合量化

### 📚 相关文档

- [README.md](README.md) - 完整使用文档
- [custom_hybrid_example.yaml](custom_hybrid_example.yaml) - 配置示例
- [Netron](https://netron.app/) - 查看 ONNX 模型层名称的工具

---

## 技术细节

### 混合量化原理

混合量化允许模型的某些关键层保持浮点精度，而其他层使用 INT8/UINT8 量化。这通过以下方式实现：

1. **Step 1 - 分析阶段**
   ```python
   rknn.hybrid_quantization_step1(
       dataset=dataset_path,
       proposal=False,
       custom_hybrid=[[start_layer, end_layer], ...]
   )
   ```
   生成 `.model`, `.data`, `.quantization.cfg` 文件

2. **Step 2 - 构建阶段**
   ```python
   rknn.hybrid_quantization_step2(
       model_input="model.model",
       data_input="model.data",
       model_quantization_cfg="model.quantization.cfg"
   )
   ```
   根据配置文件构建最终模型

### YAML 配置格式

```yaml
custom_hybrid:
  - ['/layer/path/start', '/layer/path/end']
  - ['/layer/path/start', '/layer/path/end']
  ...
```

每一对表示从 start 到 end 的层将保持浮点精度。

---

## 待办事项

- [ ] 添加更多模型的混合量化配置示例
- [ ] 支持从模型自动推荐需要混合量化的层
- [ ] 添加精度评估功能
- [ ] 支持批量转换多个模型

