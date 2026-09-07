"""
主训练脚本：加载配置、准备目录、加载数据、加载模型、应用 LoRA、
格式化数据、创建训练器、训练、保存适配器。
仅包含 main() 函数，其余全部定义在 tool_sft.py 中。
"""

# NOTE: pandas 必须首行导入（Windows Trainer 崩溃修复）
import pandas  # noqa: F401

import argparse
from tool_sft import (
    load_config,
    get_logger,
    print_info,
    load_and_preprocess_data,
    load_model_and_tokenizer,
    create_lora_config,
    apply_lora,
    format_messages,
    tokenize_dataset,
    prepare_output_dirs,
    create_trainer,
    save_lora_adapter,
)


# =============================================================================
# 主函数
# =============================================================================

def main():
    """
    主训练流程：
    1. 加载配置
    2. 准备目录
    3. 加载数据
    4. 加载模型
    5. 应用 LoRA
    6. 格式化数据
    7. 创建训练器
    8. 训练
    9. 保存适配器
    """
    parser = argparse.ArgumentParser(description="SFT 训练脚本")
    parser.add_argument("--config", type=str, default="Config.yaml", help="配置文件路径")
    args = parser.parse_args()

    print_info(get_logger("train_sft"), "=" * 60)
    print_info(get_logger("train_sft"), "开始 SFT 训练")
    print_info(get_logger("train_sft"), "=" * 60)

    # 1. 加载配置
    print_info(get_logger("train_sft"), "1/9 加载配置...")
    config = load_config(args.config)

    # 2. 准备目录
    print_info(get_logger("train_sft"), "2/9 准备输出目录...")
    output_dirs = prepare_output_dirs(config)

    # 3. 加载数据
    print_info(get_logger("train_sft"), "3/9 加载数据...")
    datasets = load_and_preprocess_data(config)

    # 4. 加载模型
    print_info(get_logger("train_sft"), "4/9 加载模型...")
    tokenizer, model = load_model_and_tokenizer(config)

    # 5. 应用 LoRA
    print_info(get_logger("train_sft"), "5/9 应用 LoRA...")
    lora_config = create_lora_config(config)
    model = apply_lora(model, lora_config)

    # 6. 格式化并 tokenize 数据
    print_info(get_logger("train_sft"), "6/9 格式化并 tokenize 数据...")
    model_config = config["ModelConfig"]
    for split in ["train", "eval"]:
        # 首先格式化文本
        datasets[split] = datasets[split].map(
            format_messages,
            fn_kwargs={"tokenizer": tokenizer},
            remove_columns=datasets[split].column_names,
        )
        # 然后 tokenize
        datasets[split] = datasets[split].map(
            tokenize_dataset,
            fn_kwargs={"tokenizer": tokenizer, "max_length": model_config.get("max_seq_length", 512)},
            remove_columns=datasets[split].column_names,
        )

    # 7. 创建训练器
    print_info(get_logger("train_sft"), "7/9 创建训练器...")
    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=datasets["train"],
        eval_dataset=datasets["eval"],
        config=config,
        output_dirs=output_dirs,
    )

    # 8. 训练
    print_info(get_logger("train_sft"), "8/9 开始训练...")
    trainer.train()

    # 9. 保存 LoRA 适配器（仅保存轻量级适配器，不保存完整权重）
    print_info(get_logger("train_sft"), "9/9 保存 LoRA 适配器...")
    lora_dir = output_dirs["lora"]
    save_lora_adapter(model, lora_dir)
    print_info(
        get_logger("train_sft"),
        f"LoRA 适配器已保存到: {lora_dir}",
    )
    print_info(
        get_logger("train_sft"),
        "注意：仅保存 LoRA 适配器（约几十 MB），未保存完整模型权重（约数 GB）",
    )

    print_info(get_logger("train_sft"), "=" * 60)
    print_info(get_logger("train_sft"), "训练完成!")
    print_info(get_logger("train_sft"), "=" * 60)


if __name__ == "__main__":
    main()
