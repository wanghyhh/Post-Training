"""
DPO 训练工具脚本

提供 DPO 训练、评估、绘图所需的全部工具函数和类。
"""

import json
import os
import sys
import time
import logging
import copy
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import yaml
from datasets import load_dataset, Dataset
from peft import LoraConfig as PeftLoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from transformers.trainer_callback import TrainerCallback

logger = logging.getLogger(__name__)


# ============================================================
# 训练回调类
# ============================================================

class DPOTrainerCallback(TrainerCallback):
    """
    自定义 DPO 训练日志回调。

    在每一次 on_log 事件中记录训练指标，计算 step_time，
    同时写入时间戳日志文件和统一的 train_log.jsonl。

    Attributes:
        output_dir: 日志输出目录
        _start_time: 训练开始时间（记录于 on_train_begin）
        _last_log_time: 上一次 on_log 的时间，用于计算 step_time
        _log_file_path: 当前时间戳日志文件路径
        _log_file_handle: 当前时间戳日志文件句柄
        _step_counter: 累计步数
    """

    def __init__(self, output_dir: str, logs_subdir: str = "logs"):
        """
        初始化回调。

        Args:
            output_dir: 输出根目录
            logs_subdir: 日志子目录名
        """
        self.output_dir = output_dir
        self.logs_dir = os.path.join(output_dir, logs_subdir)
        os.makedirs(self.logs_dir, exist_ok=True)

        # 时间戳命名日志文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_file_path = os.path.join(self.logs_dir, f"train_log_{timestamp}.jsonl")
        self._unified_log_path = os.path.join(self.logs_dir, "train_log.jsonl")

        self._log_file_handle = None
        self._start_time = None
        self._last_log_time = None
        self._step_counter = 0

    def on_init_end(self, args, state, control, **kwargs):
        """训练器初始化完成后打开日志文件句柄（仅 rank 0）。"""
        if state.is_world_process_zero:
            self._log_file_handle = open(self._log_file_path, "w", encoding="utf-8")

    def on_train_begin(self, args, state, control, **kwargs):
        """训练开始，记录起始时间（仅 rank 0 打印配置概要）。"""
        self._start_time = time.time()
        self._last_log_time = self._start_time
        if state.is_world_process_zero:
            logger.info("=" * 60)
            logger.info("开始 DPO 训练")
            logger.info(f"  输出目录: {self.output_dir}")
            logger.info(f"  日志文件: {self._log_file_path}")
            logger.info("=" * 60)

    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        每一步训练或评估后的日志记录。

        计算 step_time 并写入日志文件。先写入时间戳 jsonl，再同步到 train_log.jsonl。
        """
        if logs is None:
            return
        if state.is_world_process_zero and self._log_file_handle is not None:
            now = time.time()
            # 计算 step_time（秒/步）
            if self._last_log_time is not None:
                step_time = now - self._last_log_time
            else:
                step_time = 0.0
            self._last_log_time = now

            # 构造日志行
            log_entry = {
                "step": state.global_step,
                "epoch": state.epoch if state.epoch is not None else 0.0,
                "timestamp": datetime.now().isoformat(),
                "step_time": round(step_time, 4),
            }
            # 合并训练器 logs（含 train 或 eval 指标）
            log_entry.update(logs)

            # 写入时间戳日志
            line = json.dumps(log_entry, ensure_ascii=False)
            self._log_file_handle.write(line + "\n")
            self._log_file_handle.flush()

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """
        每次评估完成时打印评估指标。
        """
        if metrics is None:
            return
        if state.is_world_process_zero:
            logger.info(f"[eval][step={state.global_step}] 评估结果:")
            for key, val in sorted(metrics.items()):
                if isinstance(val, float):
                    logger.info(f"  {key} = {val:.6f}")
                else:
                    logger.info(f"  {key} = {val}")

    def on_train_end(self, args, state, control, **kwargs):
        """
        训练结束。关闭日志文件，同步统一日志，打印总时长。
        """
        if state.is_world_process_zero:
            elapsed = time.time() - self._start_time if self._start_time else 0
            total_steps = state.global_step
            minutes = elapsed / 60.0
            logger.info("=" * 60)
            logger.info(f"训练结束 | 总步数: {total_steps} | 总耗时: {minutes:.2f} 分钟 ({elapsed:.1f} 秒)")
            logger.info("=" * 60)

            # 关闭时间戳日志文件句柄
            if self._log_file_handle is not None:
                self._log_file_handle.close()
                self._log_file_handle = None

            # 同步 train_log.jsonl = 时间戳日志的完全复制
            if os.path.exists(self._log_file_path):
                import shutil
                shutil.copy2(self._log_file_path, self._unified_log_path)
                logger.info(f"统一日志已同步: {self._unified_log_path}")


# ============================================================
# 配置加载
# ============================================================

def load_config(config_path: str) -> dict:
    """
    从 YAML 文件加载配置。

    Args:
        config_path: YAML 配置文件路径

    Returns:
        解析后的配置字典
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


# ============================================================
# 日志设置
# ============================================================

def setup_logging(output_dir: str, logs_subdir: str = "logs"):
    """
    配置日志输出（同时写入文件和控制台）。

    Args:
        output_dir: 输出根目录
        logs_subdir: 日志子目录名
    """
    log_dir = os.path.join(output_dir, logs_subdir)
    os.makedirs(log_dir, exist_ok=True)

    # 根 logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # 清除已有 handler 避免重复
    root_logger.handlers.clear()

    # 格式
    formatter = logging.Formatter(
        "[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S"
    )

    # 文件 handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f"run_{timestamp}.log"), encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # 控制台 handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    return root_logger


# ============================================================
# 数据集加载与预处理
# ============================================================

def load_preference_dataset(data_path: str, data_usage: float = 1.0) -> Dataset:
    """
    从 jsonl 加载偏好对数据集。

    数据格式（每行一个 JSON）：
        {"prompt": [{"role": "user", "content": "..."}],
         "chosen": [{"role": "assistant", "content": "..."}],
         "rejected": [{"role": "assistant", "content": "..."}]}

    Args:
        data_path: jsonl 文件路径
        data_usage: 数据使用率 (0~1)

    Returns:
        加载后的 Dataset 对象（内容为 Python dict，prompt/chosen/rejected 均为消息列表）
    """
    dataset = load_dataset("json", data_files=data_path, split="train")
    if data_usage < 1.0:
        num_samples = int(len(dataset) * data_usage)
        dataset = dataset.select(range(num_samples))
        logger.info(f"使用率 {data_usage}: 从 {len(dataset)} 条中选用前 {num_samples} 条")
    logger.info(f"加载数据集: {data_path} | 样本数: {len(dataset)}")
    return dataset


def split_train_val(dataset: Dataset, val_ratio: float, seed: int = 42) -> tuple[Dataset, Dataset]:
    """
    按比例划分训练集和验证集。

    使用 shuffle + select 方式确保随机划分。

    Args:
        dataset: 原始数据集
        val_ratio: 验证集占比 (0~1)
        seed: 随机种子

    Returns:
        (train_dataset, eval_dataset) 元组
    """
    if val_ratio <= 0:
        return dataset, None
    dataset = dataset.shuffle(seed=seed)
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_dataset = dataset.select(range(train_size))
    eval_dataset = dataset.select(range(train_size, len(dataset)))
    logger.info(f"数据集划分: 训练 {len(train_dataset)} 条 | 验证 {len(eval_dataset)} 条")
    return train_dataset, eval_dataset


def truncate_dataset_text(dataset: Dataset, tokenizer, max_prompt_length: int = None,
                          max_completion_length: int = None) -> Dataset:
    """
    对数据集中的 prompt/chosen/rejected 文本做 token 级截断。

    由于 TRL 1.5.1 的 DPOConfig 仅支持 max_length（总长）截断，
    本函数在数据加载阶段对 prompt 和 completion 分别做文本级截断，
    避免 prompt 过长导致回答被截掉（truncation_mode=keep_end）或反之。

    Args:
        dataset: 偏好对数据集
        tokenizer: 用于 tokenize 的 tokenizer
        max_prompt_length: prompt 最大 token 数（None=不限制）
        max_completion_length: chosen/rejected 最大 token 数（None=不限制）

    Returns:
        截断后的数据集（文本被截断后 decode 回原始格式，仍保持 messages 列表结构）
    """

    def _truncate_messages_text(messages, tokenizer, max_tokens):
        """将 messages 中的文本按 token 数截断。"""
        if max_tokens is None or max_tokens <= 0:
            return messages
        truncated = []
        remaining = max_tokens
        for msg in messages:
            tokens = tokenizer.encode(msg["content"], add_special_tokens=False)
            if len(tokens) > remaining:
                # 截断该消息内容
                kept_tokens = tokens[:remaining]
                truncated_text = tokenizer.decode(kept_tokens, skip_special_tokens=True)
                truncated.append({"role": msg["role"], "content": truncated_text})
                remaining = 0
            else:
                truncated.append(msg)
                remaining -= len(tokens)
        return truncated

    def _truncate_fn(example):
        result = {}
        result["prompt"] = _truncate_messages_text(example["prompt"], tokenizer, max_prompt_length)
        result["chosen"] = _truncate_messages_text(example["chosen"], tokenizer, max_completion_length)
        result["rejected"] = _truncate_messages_text(example["rejected"], tokenizer, max_completion_length)
        return result

    if max_prompt_length is not None or max_completion_length is not None:
        logger.info(f"执行文本级截断: max_prompt_length={max_prompt_length}, max_completion_length={max_completion_length}")
        dataset = dataset.map(_truncate_fn)
    return dataset


# ============================================================
# LoRA 配置构建
# ============================================================

def build_peft_config(lora_config: dict) -> PeftLoraConfig:
    """
    从配置字典构建 PEFT LoRA 配置。

    Args:
        lora_config: 包含 LoRA 参数的字典（r, lora_alpha, lora_dropout,
                     target_modules, bias, task_type 等）

    Returns:
        PEFT LoraConfig 对象
    """
    config = PeftLoraConfig(
        r=lora_config.get("r", 8),
        lora_alpha=lora_config.get("lora_alpha", 16),
        lora_dropout=lora_config.get("lora_dropout", 0.05),
        target_modules=lora_config.get("target_modules", ["q_proj", "v_proj"]),
        bias=lora_config.get("bias", "none"),
        task_type=lora_config.get("task_type", "CAUSAL_LM"),
    )
    return config


# ============================================================
# 模型与 Tokenizer 加载
# ============================================================

def load_model_and_tokenizer(model_path: str, torch_dtype: str = "bfloat16",
                             trust_remote_code: bool = False):
    """
    加载预训练模型和 tokenizer。

    Args:
        model_path: 模型路径或 HuggingFace 模型名
        torch_dtype: 模型加载精度
        trust_remote_code: 是否信任远程代码

    Returns:
        (model, tokenizer) 元组
    """
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "auto": "auto",
    }
    dtype = dtype_map.get(torch_dtype, torch.bfloat16)

    logger.info(f"加载模型: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    logger.info(f"模型参数量: {model.num_parameters() / 1e6:.2f}M")

    logger.info(f"加载 tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        padding_side="right",
    )
    # 设置 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# ============================================================
# 训练元信息持久化
# ============================================================

def save_train_metadata(output_dir: str, logs_subdir: str, config: dict,
                        start_time: datetime, end_time: datetime,
                        total_steps: int):
    """
    保存训练元信息文件。

    Args:
        output_dir: 输出根目录
        logs_subdir: 日志子目录
        config: 完整配置字典
        start_time: 训练开始时间
        end_time: 训练结束时间
        total_steps: 总训练步数
    """
    meta = {
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "total_steps": total_steps,
        "model_path": config.get("ModelConfig", {}).get("model_name_or_path", ""),
        "beta": config.get("TrainConfig", {}).get("beta", 0.1),
        "loss_type": config.get("TrainConfig", {}).get("loss_type", ["sigmoid"]),
        "learning_rate": config.get("TrainConfig", {}).get("learning_rate", 5e-6),
        "num_train_epochs": config.get("TrainConfig", {}).get("num_train_epochs", 3.0),
        "batch_size": config.get("TrainConfig", {}).get("per_device_train_batch_size", 1),
        "gradient_accumulation_steps": config.get("TrainConfig", {}).get("gradient_accumulation_steps", 1),
        "lora_r": config.get("LoraConfig", {}).get("r", 8),
        "lora_alpha": config.get("LoraConfig", {}).get("lora_alpha", 16),
        "max_length": config.get("TrainConfig", {}).get("max_length", 512),
        "truncation_mode": config.get("TrainConfig", {}).get("truncation_mode", "keep_end"),
    }
    log_dir = os.path.join(output_dir, logs_subdir)
    os.makedirs(log_dir, exist_ok=True)
    meta_path = os.path.join(log_dir, "train_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    logger.info(f"训练元信息已保存: {meta_path}")


# ============================================================
# 输出目录路径解析
# ============================================================

def get_output_paths(config: dict) -> dict:
    """
    从配置中计算所有输出路径。

    Args:
        config: 完整配置字典

    Returns:
        包含各输出路径的字典
    """
    output_cfg = config.get("OutputConfig", {})
    output_dir = output_cfg.get("output_dir", "./output")
    return {
        "output_dir": output_dir,
        "logs_dir": os.path.join(output_dir, output_cfg.get("logs_subdir", "logs")),
        "checkpoints_dir": os.path.join(output_dir, output_cfg.get("checkpoints_subdir", "checkpoints")),
        "final_model_dir": os.path.join(output_dir, output_cfg.get("final_model_subdir", "final_model")),
        "eval_dir": os.path.join(output_dir, output_cfg.get("eval_subdir", "eval")),
        "plots_dir": os.path.join(output_dir, output_cfg.get("plots_subdir", "plots")),
    }