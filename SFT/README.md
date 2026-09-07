# SFT 微调框架

基于 LoRA 的大语言模型监督微调（Supervised Fine-Tuning）框架，支持 Qwen2 系列及兼容模型的参数高效微调。

## 项目结构

```
SFT/
├── train_sft.py              # 主训练脚本
├── evaluate_model.py         # 模型评估脚本
├── Config.yaml               # 训练/评估配置文件
├── accelerate_config.yaml    # 多卡加速配置（4卡 MUSA）
├── requirements.txt          # Python 依赖清单
├── tool_sft.py               # 工具函数库（数据加载、模型、回调、打印等）
├── example_data/             # 示例数据集
│   ├── data_train.jsonl      # 训练数据
│   └── data_test.jsonl       # 测试数据
└── output/                   # 输出目录
    ├── lora_adapter/         # LoRA 适配器权重（~几十 MB）
    ├── checkpoints/          # 训练检查点（含优化器、调度器状态）
    ├── logs/                 # 训练日志（JSONL 格式）
    ├── eval/                 # 评估报告（Markdown）
    ├── plots/                # 训练曲线图
    └── final_model/          # 历史输出目录（已弃用）
```

## 环境要求

| 组件 | 版本 |
|------|------|
| Python | 3.11+ |
| PyTorch | 2.7.1+cu118 |
| transformers | 5.10.2 |
| PEFT | 0.19.1 |
| accelerate | 1.13.0 |
| 显卡 | RTX 3060 12GB（或更高） |

### 安装依赖

```bash
conda activate cuda118env
pip install -r requirements.txt
```

> **注意**：当前为离线环境（无外网 pip 源），需提前准备好所有依赖包。

### ⚠️ Windows 已知问题

`transformers.trainer` 导入前必须先 `import pandas`，否则触发 `0xC0000005` 崩溃（numpy 2.4.6 + pandas 3.0.3 + pyarrow 加载顺序冲突）。所有使用 Trainer 的脚本已处理。

## 数据格式

数据集采用 JSONL 格式，每条样本包含 `messages` 字段（聊天格式）：

```json
{
  "messages": [
    {"role": "system", "content": "你是一个乐于助人的AI助手。"},
    {"role": "user", "content": "Python中如何列表推导式？"},
    {"role": "assistant", "content": "列表推导式是Python中创建列表的简洁方式..."}
  ]
}
```

## 快速开始

### 1. 配置修改

编辑 `Config.yaml`，主要修改：

- **`ModelConfig.model_name_or_path`**：基座模型路径（如 `"E:\\LLM\\models\\Qwen\\Qwen2-0___5B"`）
- **`DataConfig.train_data_path`** / `test_data_path`：训练/测试数据路径
- **`TrainConfig`**：训练超参数（学习率、batch size、epochs 等）
- **`LoRAConfig`**：LoRA 参数（秩、缩放系数、目标模块等）
- **`EvalConfig`**：评估开关与路径

### 2. 启动训练

**单卡训练**：
```bash
python train_sft.py --config Config.yaml
```

**多卡训练（4卡 MUSA）**：
```bash
accelerate launch --config_file accelerate_config.yaml train_sft.py --config Config.yaml
```

### 3. 模型评估

编辑 `Config.yaml` 中的 `EvalConfig` 部分，然后运行：
```bash
python evaluate_model.py
```

评估脚本会：
1. 可选：评估基座模型生成质量（`evaluate_base_model: true`）
2. 可选：加载 LoRA 适配器评估微调后模型（`evaluate_fine_tuned_model: true`）
3. 生成 Markdown 报告至 `output/eval/`

### 4. 训练可视化

```bash
python output/plot_sft.py
```

绘制 5 个子图：Loss、学习率、梯度范数、熵、平均 token 准确率。

## 输出说明

训练完成后，`output/` 目录包含：

| 目录/文件 | 内容 | 大小 |
|-----------|------|------|
| `lora_adapter/` | LoRA 适配器权重（`adapter_model.safetensors` + `adapter_config.json`） | ~几十 MB |
| `checkpoints/` | 训练检查点（含优化器、调度器、RNG 状态，用于断点续训） | ~几百 MB |
| `logs/train_log.jsonl` | 统一日志文件（每步 6 个指标） | ~几 KB/步 |
| `eval/*.md` | 评估报告（含生成样例，展示含角色标签的原始输入输出） | - |
| `plots/training_curves.png` | 训练曲线图 | - |

> **LoRA 适配器**：仅保存轻量级适配器权重，部署时与基座模型合并加载，无需保存完整模型权重。

## 配置文件详解

详见 `Config.yaml` 中的注释，核心配置段：

| 配置段 | 用途 |
|--------|------|
| `ModelConfig` | 基座模型路径、上下文长度、Flash Attention |
| `DataConfig` | 数据路径、使用率、验证集划分比例 |
| `LoRAConfig` | LoRA 秩、alpha、dropout、目标模块 |
| `TrainConfig` | 训练超参（学习率、batch size、调度器、精度等） |
| `EvalConfig` | 评估开关、模型路径、输出目录 |
| `OutputConfig` | 各类输出文件的存储位置 |
| `PlotConfig` | 训练曲线绘图参数 |

## 技术细节

- **自定义损失函数**：通过 `compute_loss_func` 注入 token 级指标（熵、token 计数、token 级准确率）
- **训练日志**：双文件机制（`train_log_YYYYMMDD_HHMMSS.jsonl` 独立文件 + `train_log.jsonl` 统一接口）
- **评估报告**：展示模型原始输入输出（含 `<|system|>`、`<|user|>`、`<|assistant|>` 角色标签）
