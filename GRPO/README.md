# GRPO 后训练管线（GRPO + LoRA）

## 概述

本仓库实现了一个基于 **GRPO（Group Relative Policy Optimization）** 的大语言模型后训练管线，使用 **LoRA（Low-Rank Adaptation）** 实现参数高效微调。管线覆盖完整生命周期：训练 → 评估 → 可视化。

- **基础框架**：Hugging Face Transformers 5.10.2 + TRL 1.5.1 + PEFT 0.19.1
- **训练引擎**：GRPOTrainer（TRL 提供）、Accelerate（多卡分布式）
- **支持硬件**：NVIDIA GPU（MULTI_GPU）/ 海光 DCU（MULTI_MUSA）
- **运行环境**：本地 Windows conda `cuda118env` / Linux 服务器多卡

## 环境依赖

| 包名 | 版本 | 说明 |
|------|------|------|
| torch | 2.7.1+cu118 | 主 ML 框架 |
| transformers | 5.10.2 | Hugging Face 核心库 |
| trl | 1.5.1 | GRPOTrainer / GRPOConfig |
| peft | 0.19.1 | LoRA 适配器 |
| accelerate | 1.13.0 | 多卡训练调度 |
| pandas | 3.0.3 | 数据处理（Windows 环境关键依赖） |
| numpy | — | 数值计算 |
| matplotlib | — | 训练曲线可视化 |
| pyyaml | — | 配置文件解析 |

> ⚠️ **Windows conda 环境导入顺序约束**：所有脚本必须在任何 `transformers`/`trl` 导入之前先 `import pandas`，否则触发 `0xC0000005` 访问违例崩溃。

## 目录结构

```
GRPO/
├── Config.yaml                      # 全部配置集中管理
├── accelerate_config.yaml           # Accelerate 多卡训练配置
├── reward_function.py               # 奖励函数模块
├── tool_grpo.py                     # 通用工具函数
├── train_grpo.py                    # 训练主脚本
├── evaluate_model.py                # 评估主脚本
├── plot_grpo.py                     # 训练曲线绘图脚本
├── example_data/
│   ├── data_train.jsonl             # 训练数据集示例（30 条）
│   └── data_test.jsonl              # 测试数据集示例（12 条）
├── output/                          # 训练/评估输出目录（自动创建）
│   ├── training/                    # 训练输出
│   │   ├── train_log.jsonl          # 统一日志接口
│   │   ├── train_log_YYYYMMDD_HHMMSS.jsonl  # 时间戳日志
│   │   ├── best_model/              # 最佳 LoRA adapter
│   │   ├── final_adapter/           # 最终 LoRA adapter
│   │   └── plots/                   # 训练曲线图
│   └── eval_results/                # 评估报告
│       ├── eval_result_YYYYMMDD_HHMMSS.json
│       └── eval_report_YYYYMMDD_HHMMSS.md
├── tmp/                             # 临时文件
└── example/                         # 参考代码
    ├── GRPO-Fine-tuning-1round/     # GRPO 参考训练脚本
    └── plot/                        # 参考绘图脚本
```

## 快速开始

### 1. 训练

**本地 Windows 单卡：**
```bash
cd GRPO
python train_grpo.py
```

**Linux 服务器多卡：**
```bash
accelerate launch --config_file accelerate_config.yaml train_grpo.py
```

### 2. 评估

**评估最佳模型：**
```bash
python evaluate_model.py --checkpoint output/training/best_model
```

**评估完整模型：**
```bash
python evaluate_model.py --model path/to/merged_model
```

**自定义参数：**
```bash
python evaluate_model.py \
    --checkpoint output/training/best_model \
    --num-generations 8 \
    --batch-size 4 \
    --device cuda:0
```

### 3. 绘图

**使用默认日志：**
```bash
python plot_grpo.py
```

**指定日志文件：**
```bash
python plot_grpo.py --log-file output/training/train_log_20240101_120000.jsonl
```

**自定义平滑步长和布局：**
```bash
python plot_grpo.py --stride 20 --ncols 4 --title "GRPO Training Curve"
```

## 配置文件详解

### Config.yaml

所有参数集中管理，无硬编码环境名称。

| 配置节 | 用途 |
|--------|------|
| `ModelConfig` | 基座模型路径、输入最大 token、设备 |
| `DataSetConfig` | 训练/测试数据集路径、采样率、验证集划分 |
| `TrainConfig` | GRPO 训练参数（学习率、batch size、KL 系数、生成参数等） |
| `LoraParamConfig` | LoRA 秩、缩放系数、dropout、目标模块 |
| `RewardFuncConfig` | 奖励函数权重、答案标签、违禁字符串 |
| `EvalConfig` | 评估参数（生成数、温度、批大小等） |
| `PlotConfig` | 绘图参数（列数、平滑步长、输出目录） |

### accelerate_config.yaml

多卡训练配置，默认适配海光 DCU（4 卡、bf16、`MULTI_MUSA`）。

## 模块说明

### reward_function.py — 奖励函数

```python
class GRPORewardFunction:
    """多组件加权奖励函数"""
```

- **长度奖励**：`1 - |gen_len - ref_len| / (gen_len + ref_len)`，归一化到 [0,1]
- **匹配奖励**：首字符匹配为 1.0，否则为 0.0
- **违禁词拦截**：命中违禁字符串则总奖励归零
- **批量处理**：支持批量输入，返回 `torch.Tensor`
- **内部缓存**：维护 `rewards` 字典供日志回调和评估报告提取

### tool_grpo.py — 通用工具

| 函数/类 | 说明 |
|---------|------|
| `ConfigLoader` | YAML 配置读取（按节/全量） |
| `load_dataset()` | JSONL 数据集加载 + 格式验证 |
| `validate_dataset_format()` | 数据集结构验证 |
| `setup_lora()` | 加载模型 + 配置 LoRA |
| `save_peft_model()` | 仅保存 LoRA adapter 权重 |
| `make_reward_function()` | 奖励函数实例化 |
| `load_hf_model()` / `save_hf_model()` | 完整 HuggingFace 模型加载/保存 |
| `merge_and_unload()` | 合并 LoRA 到基础模型 |
| `load_eval_config()` | 评估配置合并 |

### train_grpo.py — 训练管线

8 步流程：

1. 加载 `Config.yaml`
2. 加载数据集 + 验证
3. 应用 Chat Template 预处理
4. 加载模型 + LoRA 配置
5. 实例化奖励函数
6. 初始化 `GRPOTrainer`
7. 训练循环（含回调）
8. 保存最终 LoRA adapter

**回调机制**：

- `TrainingLoggerCallback`：双文件日志（时间戳文件 + 统一接口 `train_log.jsonl`），主进程写入
- `SaveBestModelCallback`：监控 `eval_reward`，自动保存最佳 LoRA adapter

### evaluate_model.py — 评估管线

支持三种模型模式：完整模型 / LoRA adapter / 基座模型。

**评估指标**：

| 指标 | 说明 |
|------|------|
| `avg_reward` | 平均奖励 |
| `reward_std` | 奖励标准差 |
| `Pass@k` | Pass@1 ~ Pass@k 通过率 |
| 奖励分布 | min/p25/median/p75/max |
| 分项奖励 | 长度奖励均值/标准差 |
| 生成速度 | tokens/s |
| 每个 Prompt 奖励标准差 | 多样性指标 |

**输出**：JSON 详细结果 + Markdown 可读报告

### plot_grpo.py — 训练曲线可视化

- 自动查找 `output/train_log.jsonl` 或最新时间戳日志文件
- 多子图布局（Train/Eval 同图对比）
- stride 平滑降采样（原始曲线 alpha=0.2 + 平滑线 alpha=0.8）
- `step_time` 特殊处理：累计成本曲线，红色标注终点
- 输出高分辨率 PNG（150 dpi）

## 数据格式

### JSONL 数据集格式

每条记录必须包含 `prompt`（问题）和 `answer`（参考答案/生成目标）字段：

```jsonl
{"prompt": "1 + 1 = ?", "answer": "2", "reference": "2"}
{"prompt": "2 + 3 = ?", "answer": "5", "reference": "5"}
```

可选 `messages` 格式（对话式）：

```jsonl
{"messages": [{"role": "user", "content": "1+1="}], "answer": "2", "reference": "2"}
```

可选 `reference` 字段（不提供则与 `answer` 相同）。

## 训练日志

采用双文件机制：

1. **`train_log_YYYYMMDD_HHMMSS.jsonl`** — 每次训练独立文件，追加写入
2. **`train_log.jsonl`** — 统一接口文件，追加写入所有步骤日志

日志每行是一个 JSON 对象，包含：`step`, `loss`, `learning_rate`, `reward`, `kl`, `global_step` 等。

## 注意事项

- ⚠️ **Windows 环境**：必须先 `import pandas` 再导入 transformers/trl
- ⚠️ **GPU 显存**：LoRA 训练可大幅降低显存需求，建议 per_device_train_batch_size 与 num_generations 的乘积不超过显存承载
- ⚠️ **Perplexity / 奖励值**：奖励值随权重配置浮动，关注相对趋势而非绝对大小
- ⚠️ **多 GPU**：非主进程跳过日志打印和模型保存（通过 `local_rank == 0` 判断）
