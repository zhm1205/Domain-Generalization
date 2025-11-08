# MIMIC-CXR 数据集生成工具使用文档

## 🎯 核心特性

### ✅ 完成的功能
1. **一致性保证**: 疾病分类和ReID数据集使用**完全相同**的数据，确保Unlearnable Example实验的一致性
2. **智能筛选**: ReID数据集只保留frontal views (PA+AP) + 每个患者≥2个研究
3. **灵活的Split策略**: 支持官方split或重新平衡split（更大的测试集）
4. **可配置的图像处理**: 支持不同图像大小、格式和处理参数
5. **快速测试工具**: 支持快速验证逻辑和小批量图像处理测试

## 📊 数据统计

### 原始数据
- **总图片**: 377,110张
- **总患者**: 65,379个
- **总研究**: 227,835个

### ReID筛选后
- **筛选后图片**: 214,106张 (减少45.2%)
- **筛选后患者**: 34,717个 (每个患者≥2个影像)
- **平均每患者图片数**: 6.65张

### Split策略对比

#### 官方Split (official)
- Train: 201,727张 (97.6%), 30,599个患者
- Validate: 1,670张 (0.8%), 235个患者  
- Test: 3,385张 (1.6%), 271个患者

#### 重新平衡Split (rebalanced)
- Train: 160,670张 (74.9%), 23,328个患者
- Validate: 21,504张 (10.3%), 3,110个患者
- Test: 30,653张 (14.8%), 4,667个患者

## 🔧 配置文件

`configs/scripts/mimic_cxr.yaml`:
```yaml
raw_data_folder: /home/dengzhipeng/data/chest/mimic_cxr_unzipped
output_folder: /home/dengzhipeng/data/project/reid_ue
resize: [224, 224]

# Split rebalancing configuration
split_ratios:
  train: 0.75    # 75% for training
  validate: 0.10 # 10% for validation  
  test: 0.15     # 15% for testing

# Image processing configuration
image_processing:
  format: "PNG"
  background_color: 0  # Black background for medical images
  compression_level: 3
```

## 🚀 使用示例

### 1. 仅生成CSV文件（推荐先运行）
```bash
# 使用官方split
python scripts/generate_mimic_cxr_dataset.py --no-resize --task both --split-strategy official

# 使用重新平衡的split（推荐）
python scripts/generate_mimic_cxr_dataset.py --no-resize --task both --split-strategy rebalanced
```

### 2. 生成完整数据集（包含图像处理）
```bash
# 生成完整数据集（需要20+分钟）
python scripts/generate_mimic_cxr_dataset.py --task both --split-strategy rebalanced

# 只生成ReID数据集
python scripts/generate_mimic_cxr_dataset.py --task reid --split-strategy rebalanced

# 只生成分类数据集（基于ReID筛选的数据）
python scripts/generate_mimic_cxr_dataset.py --task classification --split-strategy rebalanced
```

### 3. 快速测试工具

#### 验证数据逻辑（无图像处理）
```bash
python scripts/quick_test_mimic.py --save-csvs --subset-size 100
```

#### 测试图像处理功能
```bash
python scripts/test_image_processing.py \
    --csv /path/to/dataset.csv \
    --output /path/to/test_output \
    --samples 10
```

### 4. 训练集采样脚本
```bash
python scripts/mimic_cxr/sample_train_subset.py \
    --input /path/to/reid/reid_dataset.csv \
    --output /path/to/reid/reid_dataset_small.csv \
    --train-ratio 0.1
```
运行 `pytest scripts/mimic_cxr/test_sample_train_subset.py` 以验证采样逻辑。

## 📁 输出文件结构

```
{output_folder}/
├── disease_classification/
│   ├── disease_classification.csv    # 疾病分类数据集CSV
│   └── images/                       # 处理后的图片（如果启用resize）
│       └── *.png
├── reid/
│   ├── reid_dataset.csv             # ReID数据集CSV  
│   └── images/                       # 处理后的图片（如果启用resize）
│       └── *.png
└── quick_test/                       # 快速测试输出
    ├── classification_test.csv
    └── reid_test.csv
```

## 🎛️ 命令行参数

### 主生成脚本 (`generate_mimic_cxr_dataset.py`)
- `--config`: 配置文件路径 (默认: configs/scripts/mimic_cxr.yaml)
- `--no-resize`: 跳过图像处理，只生成CSV
- `--task`: 生成任务 [classification|reid|both] (默认: both)
- `--split-strategy`: Split策略 [official|rebalanced] (默认: official)

### 快速测试脚本 (`quick_test_mimic.py`)
- `--config`: 配置文件路径
- `--subset-size`: 测试子集大小 (默认: 100)
- `--save-csvs`: 保存测试CSV文件

### 图像处理测试 (`test_image_processing.py`)
- `--csv`: CSV文件路径
- `--output`: 输出文件夹
- `--samples`: 处理样本数 (默认: 10)
- `--config`: 配置文件路径

### 训练集采样脚本 (`sample_train_subset.py`)
- `--input`: 输入CSV文件路径
- `--output`: 输出CSV文件路径
- `--train-ratio`: 训练集保留比例 (0-1)
- `--seed`: 随机种子 (默认: 42)

## 💡 使用建议

### 开发流程
1. **先验证数据**: `python scripts/quick_test_mimic.py --save-csvs`
2. **仅生成CSV**: `python scripts/generate_mimic_cxr_dataset.py --no-resize --task both --split-strategy rebalanced`
3. **验证CSV正确性**: 检查生成的文件行数和内容
4. **测试图像处理**: `python scripts/test_image_processing.py --samples 10 ...`
5. **生成完整数据集**: `python scripts/generate_mimic_cxr_dataset.py --task both --split-strategy rebalanced`

### Unlearnable Example实验
- ✅ 两个数据集使用**完全相同**的数据，确保实验一致性
- ✅ 支持灵活的split策略，推荐使用`rebalanced`获得更大的测试集
- ✅ 所有患者都有≥2个研究，满足ReID任务需求

### 性能优化
- CSV生成: <2分钟
- 图像处理: ~20分钟（206K张图片）
- 推荐先生成CSV验证逻辑，再进行图像处理

## 🔍 数据验证

生成数据集后，可以验证：
```bash
# 检查文件行数是否一致
wc -l /path/to/disease_classification/disease_classification.csv
wc -l /path/to/reid/reid_dataset.csv

# 检查CSV内容
head -5 /path/to/reid/reid_dataset.csv

# 检查split分布
python -c "
import pandas as pd
df = pd.read_csv('/path/to/reid/reid_dataset.csv')
print('Split distribution:')
print(df['split'].value_counts())
print('\\nPatients per split:')
print(df.groupby('split')['subject_id'].nunique())
"
```

## 🚨 重要注意事项

1. **数据一致性**: 疾病分类和ReID数据集使用相同的数据，确保实验有效性
2. **Split完整性**: 所有split都保证患者级别的分离，无数据泄露
3. **图像格式**: 统一使用灰度PNG格式，适合医学图像
4. **ReID要求**: 只保留有≥2个研究的患者，满足ReID任务需求
5. **路径一致性**: 确保原始图像路径正确，避免处理错误

---

**准备就绪！🎉** 现在可以开始你的Unlearnable Example实验了！
