# Post-Training — 大模型后训练管线

基于 Hugging Face 生态的大语言模型后训练框架，使用 **LoRA** 实现参数高效微调，包含两条完整管线：

| 管线 | 方法 | 训练器 | 适用场景 |
|------|------|--------|----------|
| **GRPO** | Group Relative Policy Optimization | `GRPOTrainer`（TRL） | RLHF 风格强化学习后训练 |
| **SFT** | Supervised Fine-Tuning | `Trainer`（Transformers） | 标准监督微调 |

## 技术栈

| 组件 | 版本 |
|------|------|
| Python | 3.11+ |
| PyTorch | 2.7.1+cu118 |
| Transformers | 5.10.2 |
| TRL | 1.5.1（仅 GRPO） |
| PEFT | 0.19.1 |
| Accelerate | 1.13.0 |

## 项目结构

```
Post-Training/
├── README.md
├── GRPO/                              # GRPO 后训练管线
│   ├── Config.yaml                    # 训练及评估配置
│   ├── accelerate_config.yaml         # 多卡加速配置
│   ├── train_grpo.py                  # 训练主脚本
│   ├── reward_function.py             # 奖励函数模块
│   ├── tool_grpo.py                   # 通用工具函数
│   ├── evaluate_model.py              # 模型评估脚本
│   ├── plot_grpo.py                   # 训练曲线绘图
│   ├── example_data/                  # 示例数据集
│   ├── output/                        # 输出目录（训练产物）
│   ├── 工作文档.md                     # 工作文档
│   └── 验收报告.md                     # 验收报告
├── SFT/                               # SFT 微调管线
│   ├── Config.yaml                    # 训练及评估配置
│   ├── accelerate_config.yaml         # 多卡加速配置
│   ├── train_sft.py                   # 训练主脚本
│   ├── tool_sft.py                    # 通用工具函数
│   ├── evaluate_model.py              # 模型评估脚本
│   ├── plot_sft.py                    # 训练曲线绘图
│   ├── requirements.txt               # Python 依赖清单
│   ├── example_data/                  # 示例数据集
│   ├── output/                        # 输出目录（训练产物）
│   └── 工作文档.md                     # 工作文档
```

## 快速开始

### 环境准备

```bash
# 创建并激活 conda 环境
conda activate cuda118env

# 安装 SFT 依赖（GRPO 依赖见 GRPO/README.md）
pip install -r SFT/requirements.txt
```

### GRPO 训练

```bash
cd GRPO
# 本地单卡训练
python train_grpo.py
# 服务器多卡训练
accelerate launch --config_file accelerate_config.yaml train_grpo.py
```

详见 [GRPO/README.md](GRPO/README.md)。

### SFT 训练

```bash
cd SFT
# 本地单卡训练
python train_sft.py
# 服务器多卡训练
accelerate launch --config_file accelerate_config.yaml train_sft.py
```

详见 [SFT/README.md](SFT/README.md)。

## 硬件支持

| 环境 | GPU | 训练方式 |
|------|-----|----------|
| 本地 Windows | NVIDIA RTX 3060 12GB | 单卡 `python` |
| Linux 服务器 | 海光 DCU × 4 | 多卡 `accelerate launch` |

## 注意事项

- ⚠️ **Windows 环境**：所有脚本必须在 `transformers`/`trl` 导入前先 `import pandas`，否则触发 `0xC0000005` 崩溃
- ⚠️ **离线环境**：当前无外网 pip 源，需提前准备所有依赖包
- ⚠️ **LoRA 保存**：训练仅保存 LoRA 适配器（~几十 MB），不保存完整模型权重（~数 GB）