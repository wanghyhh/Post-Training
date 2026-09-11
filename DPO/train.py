"""
DPO 训练脚本

完全遵循 DPO/工作文档.md 的各项要求：
  - 使用 TRL DPOTrainer + LoRA 微调
  - 自定义日志回调，记录 step_time 并写入 jsonl
  - 支持 eval_on_start + 每 epoch 评估
  - 仅保存 LoRA 适配器
"""

import os
import sys
import copy
import json
from datetime import datetime

import yaml
import torch
from datasets import Dataset
from peft import get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from trl import DPOConfig, DPOTrainer

# 导入工具函数
from tool_dpo import (
    load_config,
    setup_logging,
    load_preference_dataset,
    split_train_val,
    truncate_dataset_text,
    build_peft_config,
    load_model_and_tokenizer,
    DPOTrainerCallback,
    save_train_metadata,
    get_output_paths,
    logger,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    """
    DPO 训练主函数。

    步骤组织：
      1. 加载配置
      2. 创建输出目录 & 日志系统
      3. 加载模型与 tokenizer
      4. 加载与预处理数据集
      5. 读取并分离 DPOConfig 所需参数
      6. 构建 DPOConfig
      7. 构建 LoRA 配置
      8. 初始化训练器
      9. 训练
      10. 保存模型与元信息
    """
    # ============ 1. 加载配置 ============
    config_path = "Config.yaml"
    if not os.path.exists(config_path):
        # 支持从上级目录或 DPO 目录启动
        for candidate in ["./DPO/Config.yaml", "../Config.yaml"]:
            if os.path.exists(candidate):
                config_path = candidate
                break
    if not os.path.exists(config_path):
        print(f"错误: 找不到配置文件 {config_path}")
        sys.exit(1)

    config = load_config(config_path)
    model_cfg = config.get("ModelConfig", {})
    train_cfg_raw = config.get("TrainConfig", {})
    dataset_cfg = config.get("DatasetConfig", {})
    lora_cfg = config.get("LoraConfig", {})
    output_cfg = config.get("OutputConfig", {})

    # ============ 2. 创建输出目录 & 日志系统 ============
    paths = get_output_paths(config)
    for dir_key in ["output_dir", "logs_dir", "checkpoints_dir", "final_model_dir",
                     "eval_dir", "plots_dir"]:
        os.makedirs(paths[dir_key], exist_ok=True)

    log = setup_logging(paths["output_dir"], output_cfg.get("logs_subdir", "logs"))
    logger.info("配置加载完成")

    # ============ 3. 加载模型与 tokenizer ============
    model_path = model_cfg["model_name_or_path"]
    logger.info(f"模型路径: {model_path}")
    model, tokenizer = load_model_and_tokenizer(
        model_path=model_path,
        torch_dtype=model_cfg.get("torch_dtype", "bfloat16"),
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )

    # ============ 4. 加载与预处理数据集 ============
    logger.info("加载训练集...")
    train_dataset_raw = load_preference_dataset(
        dataset_cfg["train_data_path"],
        dataset_cfg.get("train_data_usage", 1.0),
    )
    logger.info("加载测试集...")
    test_dataset_raw = load_preference_dataset(
        dataset_cfg["test_data_path"],
        dataset_cfg.get("test_data_usage", 1.0),
    )

    # 对数据集做文本级 token 截断
    train_dataset_raw = truncate_dataset_text(
        train_dataset_raw, tokenizer,
        max_prompt_length=dataset_cfg.get("max_prompt_length", None),
        max_completion_length=dataset_cfg.get("max_completion_length", None),
    )
    test_dataset_raw = truncate_dataset_text(
        test_dataset_raw, tokenizer,
        max_prompt_length=dataset_cfg.get("max_prompt_length", None),
        max_completion_length=dataset_cfg.get("max_completion_length", None),
    )

    # 划分验证集
    val_ratio = dataset_cfg.get("val_ratio", 0.0)
    train_dataset, eval_dataset = split_train_val(train_dataset_raw, val_ratio)

    logger.info(f"训练集: {len(train_dataset)} 条")
    logger.info(f"验证集: {len(eval_dataset)} 条" if eval_dataset is not None else "验证集: 无")
    logger.info(f"测试集: {len(test_dataset_raw)} 条")

    # ============ 5. 分离 DPOConfig 所需参数 ============
    # 复制一份，pop 掉 DPOConfig 不接受的键
    train_cfg = copy.deepcopy(train_cfg_raw)
    # max_prompt_length / max_completion_length 已在数据集预处理中使用，不传给 DPOConfig
    for key in ["max_prompt_length", "max_completion_length"]:
        train_cfg.pop(key, None)

    # ============ 6. 构建 DPOConfig ============
    train_args = DPOConfig(
        **train_cfg,
        output_dir=paths["checkpoints_dir"],
    )

    # 记录启动配置
    logger.info("=" * 60)
    logger.info(f"训练配置 | beta={train_args.beta} | loss_type={train_args.loss_type} | "
                f"lr={train_args.learning_rate} | batch={train_args.per_device_train_batch_size} | "
                f"grad_acc={train_args.gradient_accumulation_steps} | "
                f"epochs={train_args.num_train_epochs} | "
                f"max_length={train_args.max_length} | truncation={train_args.truncation_mode} | "
                f"precompute_ref={train_args.precompute_ref_log_probs}")
    logger.info(f"LoRA 配置 | r={lora_cfg.get('r')} | alpha={lora_cfg.get('lora_alpha')} | "
                f"dropout={lora_cfg.get('lora_dropout')} | "
                f"targets={lora_cfg.get('target_modules')}")
    logger.info(f"数据集 | 训练={len(train_dataset)} | 验证={len(eval_dataset) if eval_dataset else 0} | "
                f"测试={len(test_dataset_raw)}")
    logger.info("=" * 60)

    # ============ 7. 构建 LoRA 配置 ============
    peft_config = build_peft_config(lora_cfg)

    # ============ 8. 初始化训练器 ============
    callbacks = []
    dpo_callback = DPOTrainerCallback(
        output_dir=paths["output_dir"],
        logs_subdir=output_cfg.get("logs_subdir", "logs"),
    )
    callbacks.append(dpo_callback)

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # LoRA 模式下训练器自动复用基座权重（禁用适配器）
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=callbacks,
    )

    # ============ 9. 训练 ============
    train_start_time = datetime.now()
    logger.info(f"训练开始 | {train_start_time.isoformat()}")
    trainer.train()
    train_end_time = datetime.now()
    logger.info(f"训练结束 | {train_end_time.isoformat()}")

    # ============ 10. 保存模型与元信息 ============
    # 保存 LoRA 适配器
    final_model_path = paths["final_model_dir"]
    logger.info(f"保存 LoRA 适配器 -> {final_model_path}")
    trainer.save_model(final_model_path)
    logger.info("模型保存完成")

    # 保存训练元信息
    save_train_metadata(
        output_dir=paths["output_dir"],
        logs_subdir=output_cfg.get("logs_subdir", "logs"),
        config=config,
        start_time=train_start_time,
        end_time=train_end_time,
        total_steps=trainer.state.global_step,
    )

    logger.info("训练流程全部完成")


if __name__ == "__main__":
    main()