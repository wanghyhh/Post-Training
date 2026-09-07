"""
工具脚本：包含数据加载、模型加载、日志回调、打印工具等所有通用函数和类。

所有其他脚本（train_sft.py、evaluate_model.py）中的函数和类定义均放在此文件中。
"""

# NOTE: pandas 必须首行导入（无此依赖但为规避 Windows 下 numpy/pyarrow 加载顺序冲突，防止 Trainer 导入时 0xC0000005 崩溃）
import pandas  # noqa: F401  # 仅用于避免 Trainer 导入崩溃，未在其他逻辑中使用

import os
import json
import logging
import math
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import yaml
import numpy as np
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model


# =============================================================================
# 自定义 SFTConfig（替代已移除的 SFTConfig）
# =============================================================================

class SFTConfig(TrainingArguments):
    """
    自定义 SFTConfig 继承自 TrainingArguments。
    用于适配 transformers 5.x 中移除 SFTTrainer/SFTConfig 后的兼容性问题。

    使用方式:
        config = SFTConfig(**train_config_dict, ...)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


# =============================================================================
# 打印工具
# =============================================================================

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    创建一个统一格式的日志记录器。

    Args:
        name: 日志记录器名称，通常用 __name__
        level: 日志级别，默认 INFO

    Returns:
        配置好的 logging.Logger 实例
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def print_info(logger: logging.Logger, message: str) -> None:
    """
    打印一行信息到日志和标准输出。

    Args:
        logger: 日志记录器实例
        message: 要打印的消息
    """
    logger.info(message)


# =============================================================================
# 配置加载
# =============================================================================

def load_config(config_path: str = "Config.yaml") -> Dict[str, Any]:
    """
    从 YAML 文件加载配置，返回包含所有命名对象的字典。

    Args:
        config_path: 配置文件路径

    Returns:
        包含 ModelConfig / DataConfig / LoRAConfig / TrainConfig /
        EvalConfig / OutputConfig / PlotConfig 的字典
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


# =============================================================================
# 数据加载
# =============================================================================

def load_and_preprocess_data(config: Dict[str, Any]) -> Dict[str, Dataset]:
    """
    加载并预处理训练/测试数据集，按验证集比例划分。

    Args:
        config: 完整配置字典

    Returns:
        {"train": Dataset, "eval": Dataset, "test": Dataset}
    """
    logger = get_logger("tool_data")
    data_config = config["DataConfig"]

    print_info(logger, f"加载训练数据: {data_config['train_data_path']}")
    train_dataset = load_dataset(
        "json",
        data_files=data_config["train_data_path"],
        split=f"train[:{int(data_config['train_data_usage'] * 100)}%]"
        if data_config["train_data_usage"] < 1.0
        else "train",
    )

    print_info(logger, f"加载测试数据: {data_config['test_data_path']}")
    test_dataset = load_dataset(
        "json",
        data_files=data_config["test_data_path"],
        split=f"train[:{int(data_config['test_data_usage'] * 100)}%]"
        if data_config["test_data_usage"] < 1.0
        else "train",
    )

    # 从训练数据中划分验证集
    val_ratio = data_config.get("val_ratio", 0.1)
    if val_ratio > 0:
        train_val_split = train_dataset.train_test_split(test_size=val_ratio, seed=42)
        train_ds = train_val_split["train"]
        eval_ds = train_val_split["test"]
        print_info(
            logger,
            f"训练集: {len(train_ds)} 样本, 验证集: {len(eval_ds)} 样本",
        )
    else:
        train_ds = train_dataset
        eval_ds = train_dataset
        print_info(logger, "未划分验证集，使用全部训练数据")

    return {"train": train_ds, "eval": eval_ds, "test": test_dataset}


def format_messages(example: Dict, tokenizer=None) -> Dict[str, str]:
    """
    将多轮对话样本渲染为带 chat 模板的文本。

    Args:
        example: 包含 messages 字段的样本字典
        tokenizer: 用于 apply_chat_template 的 tokenizer（由 main() 传入）

    Returns:
        渲染后的对话文本字符串，键为 "text"
    """
    text = tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False
    )
    return {"text": text}


def tokenize_dataset(example: Dict, tokenizer=None, max_length: int = 512) -> Dict:
    """
    将格式化后的文本 tokenize 为模型输入格式。

    Args:
        example: 包含 text 字段的样本字典
        tokenizer: 用于 tokenization 的 tokenizer
        max_length: 最大序列长度，超长时截断

    Returns:
        包含 input_ids, attention_mask, labels 的字典
    """
    tokenized = tokenizer(
        example["text"],
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
    )
    # labels 与 input_ids 相同（监督学习：模型需要预测完整序列）
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


# =============================================================================
# 模型加载
# =============================================================================

def load_model_and_tokenizer(
    config: Dict[str, Any],
) -> tuple:
    """
    加载基座模型的 tokenizer 和模型。

    Args:
        config: 完整配置字典

    Returns:
        (tokenizer, model) 元组
    """
    logger = get_logger("tool_model")
    model_config = config["ModelConfig"]

    print_info(logger, f"加载模型: {model_config['model_name_or_path']}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_config["model_name_or_path"],
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_config["model_name_or_path"],
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    print_info(logger, f"模型加载完成, VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    return tokenizer, model


# =============================================================================
# LoRA 配置
# =============================================================================

def create_lora_config(config: Dict[str, Any]) -> LoraConfig:
    """
    根据配置创建 LoRA 配置对象。

    Args:
        config: 完整配置字典

    Returns:
        peft.LoraConfig 实例
    """
    lora_cfg = config["LoRAConfig"]
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    return lora_config


def apply_lora(model, lora_config: LoraConfig):
    """
    将 LoRA 适配器应用到模型上。

    Args:
        model: 待微调的基座模型
        lora_config: LoRA 配置对象

    Returns:
        应用 LoRA 后的模型
    """
    logger = get_logger("tool_lora")
    print_info(logger, "应用 LoRA 适配器...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# =============================================================================
# Token 级指标累积器与自定义损失函数
# =============================================================================

class TokenMetricsAccumulator:
    """
    按 model.training 状态分写的两个累积器，记录当前 logging step 的窗口指标。

    全局 token 计数共享、窗口均值记录后归零。
    在 compute_loss_func 中累加，在 on_log 中读取并发布到 logs。
    """

    def __init__(self):
        self.train_tokens = 0
        self.train_loss_sum = 0.0
        self.train_correct = 0
        self.eval_tokens = 0
        self.eval_loss_sum = 0.0
        self.eval_correct = 0
        self._pending = None  # 当前 step 累积结果

    def update(self, is_train: bool, loss: float, num_tokens: int, correct_tokens: int = 0):
        """
        在 compute_loss_func 中调用，累积当前 step 的指标。

        Args:
            is_train: 是否处于训练模式
            loss: 当前 batch 的损失值（标量）
            num_tokens: 当前 batch 的有效 token 数
            correct_tokens: 当前 batch 预测正确的 token 数（用于 mean_token_accuracy）
        """
        if is_train:
            self.train_tokens += num_tokens
            self.train_loss_sum += float(loss) * num_tokens
            self.train_correct += correct_tokens
        else:
            self.eval_tokens += num_tokens
            self.eval_loss_sum += float(loss) * num_tokens
            self.eval_correct += correct_tokens

    def flush(self) -> Dict[str, float]:
        """
        刷新当前累积值，返回窗口指标字典，然后归零。

        Returns:
            {
                'entropy': float,           # 训练窗口熵
                'num_tokens': float,        # 训练窗口 token 数
                'mean_token_accuracy': float,  # 训练窗口 token 准确率
                'eval_entropy': float,      # 评估窗口熵
                'eval_num_tokens': float,   # 评估窗口 token 数
                'eval_mean_token_accuracy': float,  # 评估窗口 token 准确率
            }
        """
        result = {}

        # 训练窗口指标
        if self.train_tokens > 0:
            avg_loss = self.train_loss_sum / self.train_tokens
            result['entropy'] = avg_loss
            result['num_tokens'] = float(self.train_tokens)
            if self.train_tokens > 0:
                result['mean_token_accuracy'] = self.train_correct / self.train_tokens
            # 归零
            self.train_tokens = 0
            self.train_loss_sum = 0.0
            self.train_correct = 0

        # 评估窗口指标（总是写入，即使 eval_tokens == 0 以避免 null 值）
        avg_eval_loss = 0.0
        if self.eval_tokens > 0:
            avg_eval_loss = self.eval_loss_sum / self.eval_tokens
            result['eval_entropy'] = avg_eval_loss
            result['eval_num_tokens'] = float(self.eval_tokens)
            result['eval_mean_token_accuracy'] = self.eval_correct / self.eval_tokens
            # 归零
            self.eval_tokens = 0
            self.eval_loss_sum = 0.0
            self.eval_correct = 0
        else:
            # eval_tokens == 0 时写入默认值（避免日志中为 null）
            result['eval_entropy'] = 0.0
            result['eval_num_tokens'] = 0.0
            result['eval_mean_token_accuracy'] = 0.0

        return result


class TokenMetricsLossFunc:
    """
    自定义损失函数类，用于 Trainer 的 compute_loss_func 参数。

    transformers 5.x 的 Trainer.compute_loss 流程：
    1. 弹出 labels：`labels = inputs.pop("labels")`（若无 labels 则为 None）
    2. 调用 model(**inputs) 得到 outputs
    3. 调用 compute_loss_func(outputs, labels, num_items_in_batch=...)

    口径与内置交叉熵完全一致（位移 logits[:,:-1] vs labels[:,1:]，ignore_index=-100），
    同时累积 token 级指标（entropy、num_tokens、mean_token_accuracy）。
    """

    def __init__(self, accumulator: TokenMetricsAccumulator):
        self.accumulator = accumulator

    def __call__(self, outputs, labels, return_outputs=False, num_items_in_batch=None, model_training_state=None):
        """
        自定义损失函数（transformers 5.x 签名）。

        注意：第二个参数名为 labels（非 inputs），因为 Trainer.compute_loss
        已先弹出 labels：labels = inputs.pop("labels")

        Args:
            outputs: 模型输出（CausalLMOutputWithPast 实例）
            labels: 训练时为 Tensor（input_ids 副本），评估时可能仍有值
            return_outputs: 是否返回模型输出
            num_items_in_batch: 有效 token 数（可选）
            model_training_state: 模型训练状态（可选，True=训练，False=评估）

        Returns:
            loss 或 (loss, outputs)
        """
        logits = outputs.logits

        # 使用 model.training_state 判断训练/评估（而非 labels）
        # 因为评估时 labels 也可能存在（评估数据集包含 labels）
        is_train = model_training_state if model_training_state is not None else True

        if not is_train:
            # 评估阶段：labels 存在，可以计算交叉熵作为 eval_entropy
            if labels is not None:
                # 使用 labels 计算交叉熵（与训练阶段相同逻辑）
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                valid_mask = shift_labels != -100
                num_tokens = int(valid_mask.sum().item())
                correct_tokens = int(((shift_logits.argmax(dim=-1) == shift_labels) & valid_mask).sum().item())
                # 评估时使用 loss.mean() 作为 entropy
                loss = loss.mean()
            else:
                # 兜底：labels 为 None 时无法计算
                loss = torch.tensor(0.0, device=logits.device)
                num_tokens = 0
                correct_tokens = 0
        else:
            # 训练阶段：使用 labels 计算交叉熵 + token 级指标
            # 位移：logits[:,:-1] vs labels[:,1:]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous() if labels is not None else None

            if shift_labels is None:
                # 兜底：labels 为 None 时无法计算
                num_tokens = 0
                correct_tokens = 0
                loss = torch.tensor(0.0, device=logits.device)
            else:
                # 展平
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

                # 计算有效 token 和正确 token
                valid_mask = shift_labels != -100
                num_tokens = int(valid_mask.sum().item())
                correct_tokens = int(((shift_logits.argmax(dim=-1) == shift_labels) & valid_mask).sum().item())

                # 如果有 num_items_in_batch，按该值归一化（口径与内置损失一致）
                if num_items_in_batch is not None and num_items_in_batch > 0:
                    loss = loss.sum() / num_items_in_batch
                else:
                    loss = loss.mean()

        # 累积指标（训练/评估分写，窗口均值记录后归零）
        self.accumulator.update(
            is_train=is_train,
            loss=loss.item(),
            num_tokens=num_tokens,
            correct_tokens=correct_tokens,
        )

        if return_outputs:
            return loss, outputs
        return loss


def create_compute_loss_func(accumulator: TokenMetricsAccumulator) -> TokenMetricsLossFunc:
    """
    创建自定义损失函数实例。

    Args:
        accumulator: TokenMetricsAccumulator 实例

    Returns:
        TokenMetricsLossFunc 实例
    """
    return TokenMetricsLossFunc(accumulator)


# =============================================================================
# 自定义 Trainer（修复 PEFT 分支绕过 compute_loss_func 的问题）
# =============================================================================


class CustomTrainer(Trainer):
    """
    自定义 Trainer：重写 compute_loss，确保 PEFT 模型下也调用 compute_loss_func。

    问题：transformers 内置 Trainer.compute_loss 对 PEFT 模型有特殊分支，
    可能绕过 compute_loss_func，导致 token 级指标（entropy/num_tokens/mean_token_accuracy）
    在训练阶段无法累积，日志中显示为 null。

    解决：重写 compute_loss，始终调用 compute_loss_func（如果已配置）。
    """

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        """
        重写 compute_loss，确保 compute_loss_func 始终被调用。

        Args:
            model: 模型
            inputs: 输入字典
            return_outputs: 是否返回模型输出
            num_items_in_batch: 有效 token 数

        Returns:
            loss 或 (loss, outputs)
        """
        # 弹出 labels（与内置逻辑一致）
        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        # 处理 model_accepts_loss_kwargs
        if self.model_accepts_loss_kwargs:
            kwargs = {}
            if num_items_in_batch is not None:
                kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **kwargs}

        # 前向传播
        outputs = model(**inputs)

        # 始终调用 compute_loss_func（如果已配置）
        if self.compute_loss_func is not None:
            loss = self.compute_loss_func(
                outputs,
                labels,
                num_items_in_batch=num_items_in_batch,
                model_training_state=self.model.training,  # 传入模型训练状态
            )
        # 无自定义函数时使用默认逻辑
        elif labels is not None:
            loss = self.label_smoother(outputs, labels, shift_labels=True)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss


# =============================================================================
# 训练日志回调
# =============================================================================

class TrainingLoggerCallback(TrainerCallback):
    """
    自定义训练日志回调类：
    - 每步训练/评估日志直接透传 Trainer 原始 logs 字典为 JSONL 一行
    - 仅主进程写入，避免多卡重复
    - 维护独立时间戳日志文件 + 统一接口文件 train_log.jsonl
    - 注入 token 级指标（entropy、num_tokens、mean_token_accuracy）

    日志文件：
      - train_log_YYYYMMDD_HHMMSS.jsonl：每次训练的独立日志，追加写入
      - train_log.jsonl：当前最新训练的完整日志，每一步同步（清空后重写）
    """

    def __init__(self, log_file_path: str, unified_log_file_path: str, token_accumulator: TokenMetricsAccumulator):
        """
        初始化训练日志回调。

        Args:
            log_file_path: 时间戳日志文件完整路径（如 ./logs/train_log_20260902_115858.jsonl）
            unified_log_file_path: 统一接口日志文件完整路径（如 ./logs/train_log.jsonl）
            token_accumulator: TokenMetricsAccumulator 实例，用于读取 token 级指标
        """
        super().__init__()
        self.log_file_path = log_file_path
        self.unified_log_file_path = unified_log_file_path
        self.token_accumulator = token_accumulator
        log_dir = os.path.dirname(self.log_file_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        # 内存中累积所有日志行，用于同步 unified 文件
        self._all_lines: List[str] = []

    def _sync_unified_file(self):
        """将所有累积的日志行写入统一接口文件（清空后重写）。"""
        try:
            with open(self.unified_log_file_path, 'w', encoding='utf-8') as f:
                for line in self._all_lines:
                    f.write(line + '\n')
        except Exception as e:
            print(f"[警告] 同步统一接口文件失败: {e}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        每次训练日志记录时触发。

        直接透传 Trainer logs 字典（模仿 example/sft），不经过任何过滤或包装。
        先追加写入时间戳日志文件，再同步统一接口文件。
        按训练/评估状态分别注入对应版本的 token 指标。
        """
        is_main = getattr(args, 'is_main_process', getattr(state, 'local_rank', 0) == 0)
        if not is_main:
            return

        if logs is None:
            return
        # 多卡环境下只处理 rank 0 的日志
        if getattr(state, 'local_rank', 0) != 0 and torch.distributed.is_initialized():
            return

        try:
            # 刷新 token 累积器，获取窗口指标
            token_metrics = self.token_accumulator.flush()
            if token_metrics:
                # 按训练/评估状态区分注入
                if "train_runtime" in logs:
                    # 训练结束摘要，不注入 token 指标
                    pass
                elif "eval_loss" in logs:
                    # 评估日志：只注入 eval_ 前缀的 token 指标
                    logs["eval_entropy"] = token_metrics.get("eval_entropy")
                    logs["eval_num_tokens"] = token_metrics.get("eval_num_tokens")
                    logs["eval_mean_token_accuracy"] = token_metrics.get("eval_mean_token_accuracy")
                else:
                    # 训练日志：只注入非 eval 前缀的 token 指标
                    logs["entropy"] = token_metrics.get("entropy")
                    logs["num_tokens"] = token_metrics.get("num_tokens")
                    logs["mean_token_accuracy"] = token_metrics.get("mean_token_accuracy")

            log_line_str = json.dumps({k: v for k, v in logs.items()}, ensure_ascii=False)
            # 1. 追加写入时间戳日志文件
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                f.write(log_line_str + '\n')
            # 2. 累积并同步统一接口文件
            self._all_lines.append(log_line_str)
            self._sync_unified_file()
        except Exception as e:
            if is_main:
                print(f"[警告] 日志写入失败: {e}")

    def on_train_end(self, args, state, control, **kwargs):
        """训练结束时打印日志路径。"""
        is_main = getattr(args, 'is_main_process', getattr(state, 'local_rank', 0) == 0)
        if is_main:
            print_info(
                get_logger("tool_callback"),
                f"[日志] 训练日志已保存到: {self.log_file_path}",
            )
            print_info(
                get_logger("tool_callback"),
                f"[日志] 统一接口日志: {self.unified_log_file_path}",
            )


# =============================================================================
# 检查点管理
# =============================================================================

def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    获取最新检查点目录。

    NOTE: transformers 5.x 中 state._saved_checkpoint_dir 不再设置，
    检查点目录按 output_dir/checkpoint-{step} 约定构造。

    Args:
        checkpoint_dir: 检查点根目录

    Returns:
        最新检查点路径，不存在则返回 None
    """
    if not os.path.exists(checkpoint_dir):
        return None
    checkpoints = [
        d for d in os.listdir(checkpoint_dir)
        if d.startswith("checkpoint-")
    ]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    return os.path.join(checkpoint_dir, checkpoints[-1])


def save_lora_adapter(model, adapter_dir: str) -> None:
    """
    保存 LoRA 适配器权重。

    Args:
        model: 带 LoRA 的模型
        adapter_dir: 适配器保存路径
    """
    os.makedirs(adapter_dir, exist_ok=True)
    model.save_pretrained(adapter_dir)
    print_info(
        get_logger("tool_checkpoint"),
        f"LoRA 适配器已保存到: {adapter_dir}",
    )


# =============================================================================
# 评估工具
# =============================================================================

def compute_metrics(eval_preds) -> Dict[str, float]:
    """
    计算评估指标：loss 和 PPL。

    NOTE: token 准确率统计必须 preds[:, :-1] 对比 labels[:, 1:]（因果 LM 位置 i 预测 i+1）

    Args:
        eval_preds: Trainer 的 EvalPrediction 对象 (predictions, label_ids)

    Returns:
        包含指标名称和值的字典
    """
    logger = get_logger("tool_eval")
    preds, labels = eval_preds

    # 处理 logits
    if isinstance(preds, tuple):
        preds = preds[0]

    preds = preds.argmax(axis=-1)

    # 替换 -100 为 tokenizer 的忽略索引
    labels = np.where(labels != -100, labels, preds)

    # 计算 perplexity - 注意因果 LM 位置 i 预测 i+1
    shift_predictions = preds[:, :-1]
    shift_labels = labels[:, 1:]

    min_length = min(shift_predictions.shape[1], shift_labels.shape[1])
    shift_predictions = shift_predictions[:, :min_length]
    shift_labels = shift_labels[:, :min_length]

    total_tokens = 0
    total_loss = 0.0

    for pred, label in zip(shift_predictions, shift_labels):
        valid_mask = label != -100
        if valid_mask.sum() > 0:
            total_tokens += valid_mask.sum()
            loss = np.mean(
                np.take_along_axis(
                    np.log(np.maximum(np.exp(pred) / np.sum(np.exp(pred), axis=-1, keepdims=True), 1e-10), 1)
                    , np.expand_dims(valid_mask.astype(int), axis=-1), axis=-1
                )
            )
            total_loss += loss * valid_mask.sum()

    ppl = math.exp(total_loss / max(total_tokens, 1)) if total_tokens > 0 else 0.0

    return {"ppl": round(ppl, 4)}


# =============================================================================
# 输出目录管理
# =============================================================================

def prepare_output_dirs(config: Dict[str, Any]) -> Dict[str, str]:
    """
    根据配置创建输出目录。

    Args:
        config: 完整配置字典

    Returns:
        包含各输出路径的字典
    """
    output_config = config["OutputConfig"]

    # 根输出目录
    base_dir = output_config["base_dir"]
    os.makedirs(base_dir, exist_ok=True)

    # 各子目录（仅保留 LoRA 适配器和日志/检查点所需目录）
    dirs = {
        "lora": os.path.join(base_dir, "lora_adapter"),
        "log": os.path.join(base_dir, "logs"),
        "checkpoint": os.path.join(base_dir, "checkpoints"),
    }

    for name, path in dirs.items():
        os.makedirs(path, exist_ok=True)

    return dirs


# =============================================================================
# 训练器创建
# =============================================================================

def create_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    config: Dict[str, Any],
    output_dirs: Dict[str, str],
) -> "Trainer":
    """
    创建并配置 Trainer。

    NOTE: transformers 5.x 中 Trainer(tokenizer=...) 已改为 Trainer(processing_class=...)
    TrainingArguments 不再接受 save_safetensors 参数
    warmup_ratio 已弃用，改用 warmup_steps
    """
    model_config = config["ModelConfig"]
    train_config = config["TrainConfig"]

    # 计算 warmup_steps（根据训练数据量和 batch 大小估算）
    total_steps = len(train_dataset) // (
        train_config["per_device_train_batch_size"] * train_config.get("gradient_accumulation_steps", 1)
    ) * train_config["num_train_epochs"]
    warmup_steps = int(
        train_config.get("warmup_ratio", 0.05) * total_steps
        if "warmup_ratio" in train_config
        else train_config.get("warmup_steps", 0)
    )

    # 训练参数
    training_args = SFTConfig(
        output_dir=output_dirs["checkpoint"],
        num_train_epochs=train_config["num_train_epochs"],
        per_device_train_batch_size=train_config["per_device_train_batch_size"],
        gradient_accumulation_steps=train_config.get("gradient_accumulation_steps", 1),
        per_device_eval_batch_size=train_config.get("per_device_eval_batch_size", 2),
        learning_rate=train_config["learning_rate"],
        lr_scheduler_type=train_config.get("lr_scheduler_type", "cosine"),
        weight_decay=train_config.get("weight_decay", 0.01),
        warmup_steps=warmup_steps,
        logging_steps=train_config.get("logging_steps", 10),
        save_strategy=train_config.get("save_strategy", "epoch"),
        save_steps=train_config.get("save_steps"),
        save_total_limit=train_config.get("save_total_limit", 3),
        eval_strategy=train_config.get("evaluation_strategy", "epoch"),
        eval_steps=train_config.get("eval_steps"),
        load_best_model_at_end=train_config.get("load_best_model_at_end", True),
        metric_for_best_model=train_config.get("metric_for_best_model", "eval_loss"),
        eval_on_start=train_config.get("eval_on_start", True),
        fp16=train_config.get("fp16", False),
        bf16=train_config.get("bf16", True),
        report_to="none",
        remove_unused_columns=train_config.get("remove_unused_columns", False),
        dataloader_num_workers=train_config.get("dataloader_num_workers", 0),
        seed=train_config.get("seed", 42),
        max_grad_norm=train_config.get("max_grad_norm", 1.0),
        # NOTE: transformers 5.x 中 save_safetensors 参数已移除
        # NOTE: warmup_ratio 已弃用，改用 warmup_steps
        # NOTE: evaluation_strategy 已改为 eval_strategy
        # NOTE: packing 不再是 TrainingArguments 的参数
    )

    # 创建 DataCollator（用于 batch 内填充不等长序列）
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        max_length=model_config.get("max_seq_length", 512),
    )

    # 创建 token 级指标累积器
    token_accumulator = TokenMetricsAccumulator()

    # 创建自定义损失函数（计算 entropy、num_tokens、mean_token_accuracy）
    compute_loss_func = create_compute_loss_func(token_accumulator)

    # 创建训练日志回调（双文件：时间戳日志 + 统一接口）
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(output_dirs["log"], f"train_log_{now}.jsonl")
    unified_log_file_path = os.path.join(output_dirs["log"], "train_log.jsonl")
    # 训练前创建/清空统一接口文件
    os.makedirs(os.path.dirname(unified_log_file_path), exist_ok=True)
    with open(unified_log_file_path, 'w', encoding='utf-8') as f:
        pass  # 空文件，后续由回调每步同步重写
    logger_callback = TrainingLoggerCallback(
        log_file_path=log_file_path,
        unified_log_file_path=unified_log_file_path,
        token_accumulator=token_accumulator,
    )
    print_info(get_logger("tool_callback"), f"[日志] 训练日志将保存到: {log_file_path}")
    print_info(get_logger("tool_callback"), f"[日志] 统一接口日志: {unified_log_file_path}")

    # 创建 CustomTrainer（重写 compute_loss，确保 PEFT 下也调用 compute_loss_func）
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[logger_callback],
        processing_class=tokenizer,  # 替代旧的 tokenizer= 参数
        compute_loss_func=compute_loss_func,  # 自定义损失函数（记录 token 级指标）
    )

    return trainer
