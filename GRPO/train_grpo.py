# ====================================================
# GRPO 训练主脚本
# 用途：读取 Config.yaml，完成数据集加载、LoRA 配置、
#       训练器初始化与训练、最终模型保存。
# 用法：
#   单进程直接运行:
#       python train_grpo.py
#   多进程（按 accelerate_config.yaml 启动，自动数据分片）:
#       accelerate launch --config_file accelerate_config.yaml train_grpo.py
# ====================================================

import os
import sys
import time
import random
import traceback

# Windows 控制台默认 GBK 编码，改为 UTF-8 以正常输出中文。
# ★ 必须同时开启 line_buffering：解释器原本对 tty 是行缓冲的，而手工用
#   io.TextIOWrapper 重建 stdout 会丢掉该属性、退化为块缓冲（攒满约 8KB 才刷盘），
#   表现为「脚本明明在跑但屏幕长时间无任何输出，退出或 Ctrl+C 时才一次性涌出」。
#   这里用 reconfigure 原地改配置，既保留原 stdout 对象，又显式恢复行缓冲。
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

# 将本脚本所在目录加入 sys.path，确保导入的是本地模块而非第三方同名库
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# ============================================================
# 依赖导入区
# ★ Windows 平台 0xC0000005 crash 防护：
#   必须先导入 pandas，再导入 transformers / trl / datasets
# ============================================================
import pandas  # noqa: F401  仅为导入顺序防护，本文件不直接使用

import datasets as hf_datasets
from datasets import Dataset
from transformers import AutoTokenizer, PrinterCallback, ProgressCallback
from transformers.utils import logging as hf_logging
from trl import GRPOConfig, GRPOTrainer

# 本地模块导入（所有通用函数与类均定义在 tool_grpo.py 中）
from tool_grpo import (
    ConfigLoader,
    TrainingLoggerCallback,
    load_dataset,
    make_reward_function,
    resolve_torch_dtype,
    save_peft_model,
    setup_lora,
)

# 抑制第三方库进度条：进度条走 stderr，会用 \r 覆盖 stdout 日志造成输出损坏
hf_datasets.disable_progress_bars()   # 抑制 datasets 的 "Map: 100%|" 等
hf_logging.disable_progress_bar()     # 抑制 transformers 的 "Loading weights: 100%|" 等


# ============================================================
# 辅助函数
# ============================================================


def subset_by_ratio(dataset: Dataset, ratio: float, label: str, seed: int = 42) -> Dataset:
    """
    按使用率对数据集做随机降采样。

    Args:
        dataset: 原始数据集对象。
        ratio: 使用率，取值 (0, 1]；>=1 时原样返回。
        label: 数据集名称，用于日志打印。
        seed: 随机种子，保证多次运行采样结果一致。

    Returns:
        降采样后的数据集。

    Raises:
        ValueError: ratio <= 0。
    """
    if ratio is None or ratio >= 1.0:
        return dataset
    if ratio <= 0:
        raise ValueError(f"{label}使用率必须大于 0，当前为 {ratio}")

    keep = max(1, int(round(len(dataset) * ratio)))
    if keep >= len(dataset):
        return dataset

    # 随机采样而非截断前 N 条，避免引入顺序偏差；固定种子保证可复现
    indices = sorted(random.Random(seed).sample(range(len(dataset)), keep))
    print(f"  {label}使用率 {ratio} → 随机保留 {keep}/{len(dataset)} 条样本")
    return dataset.select(indices)


# ============================================================
# 主流程
# ============================================================


def main():
    """
    GRPO 训练主函数。

    流程：
        1. 加载 Config.yaml 配置
        2. 加载数据集（应用使用率）并划分验证集
        3. 预处理数据集（应用 chat template）
        4. 加载基座模型并配置 LoRA
        5. 实例化奖励函数
        6. 初始化 GRPOTrainer
        7. 训练模型
        8. 保存最终模型
    """
    t_start = time.time()
    print("=" * 70)
    print("  GRPO 训练管线")
    print("=" * 70)

    # -------------------------------------------
    # Step 1: 加载配置
    # -------------------------------------------
    config_path = os.path.join(_SCRIPT_DIR, "Config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    print(f"\n[1/8] 加载配置: {config_path}")
    all_config = ConfigLoader(config_path).load_all()

    model_cfg = all_config.get("ModelConfig", {})
    data_cfg = all_config.get("DataSetConfig", {})
    train_cfg = dict(all_config.get("TrainConfig", {}))
    lora_cfg = all_config.get("LoraParamConfig", {})
    reward_cfg = all_config.get("RewardFuncConfig", {})
    output_cfg = all_config.get("OutputConfig", {})

    if "model_name_or_path" not in model_cfg:
        raise KeyError("Config.yaml 的 ModelConfig 缺少必需键 'model_name_or_path'")

    # 输出路径统一以 OutputConfig 为准，保证打印的路径与实际落盘位置一致
    log_dir = output_cfg.get("train_log_dir", "output/logs")
    checkpoint_dir = output_cfg.get("checkpoint_dir", "output/checkpoints")
    final_model_dir = output_cfg.get("final_model_dir", "output/final_model")
    # Trainer 的检查点目录由 TrainConfig.output_dir 决定，此处强制对齐 OutputConfig
    train_cfg["output_dir"] = checkpoint_dir

    # 数据集配置：兼容「平铺键名」与「嵌套字典」两种写法
    def _ds_path(flat_key: str, nest_key: str) -> str:
        return data_cfg.get(flat_key) or data_cfg.get(nest_key, {}).get("path", "")

    train_data_path = _ds_path("train_data_path", "train")
    valid_data_path = _ds_path("valid_data_path", "valid")
    train_data_ratio = data_cfg.get("train_data_ratio", 1.0)
    validation_split_ratio = data_cfg.get("validation_split_ratio", 0.0)

    print("  ✓ 配置加载完成")
    print(f"    基座模型  : {model_cfg['model_name_or_path']}")
    print(f"    训练集    : {train_data_path or '未配置'}（使用率 {train_data_ratio}）")
    if valid_data_path:
        print(f"    验证集    : {valid_data_path}（独立文件）")
    elif 0 < validation_split_ratio < 1.0:
        print(f"    验证集    : 将从训练集按 {validation_split_ratio} 自动划分")
    else:
        print("    验证集    : 无")
    print(f"    检查点目录: {checkpoint_dir}")
    print(f"    最终模型  : {final_model_dir}")
    print(f"    训练日志  : {log_dir}")
    print(f"    LoRA      : r={lora_cfg.get('r', 'N/A')}, alpha={lora_cfg.get('lora_alpha', 'N/A')}")
    print(f"    训练轮数  : {train_cfg.get('num_train_epochs', 'N/A')}")

    # -------------------------------------------
    # Step 2: 加载数据集并划分验证集
    # -------------------------------------------
    print("\n[2/8] 加载数据集")
    if not train_data_path or not os.path.exists(train_data_path):
        raise FileNotFoundError(f"训练数据集不存在: {train_data_path}")

    prompts, answers, references = load_dataset(train_data_path)
    # 参考答案列名使用复数 references，与奖励函数形参名一致：
    # GRPOTrainer 按「列名」把额外字段透传给奖励函数，列名不匹配会导致
    # 奖励函数收不到参考答案（references 恒为 None）
    full_dataset = Dataset.from_dict(
        {"prompt": prompts, "answer": answers, "references": references}
    )
    print(f"  原始数据集: {len(full_dataset)} 条样本")

    # 应用训练数据使用率（工作文档 3.1.6）
    full_dataset = subset_by_ratio(full_dataset, train_data_ratio, "训练集")

    if valid_data_path and os.path.exists(valid_data_path):
        # 独立验证集文件：与训练集互不相干，训练集保持完整
        eval_prompts, eval_answers, eval_refs = load_dataset(valid_data_path)
        eval_dataset = Dataset.from_dict(
            {"prompt": eval_prompts, "answer": eval_answers, "references": eval_refs}
        )
        train_dataset = full_dataset
        print(f"  训练集: {len(train_dataset)} 条样本")
        print(f"  验证集: {len(eval_dataset)} 条样本（独立文件）")
    elif 0 < validation_split_ratio < 1.0:
        # 从训练集中按比例随机划分验证集
        splits = full_dataset.train_test_split(
            test_size=validation_split_ratio, seed=42
        )
        train_dataset = splits["train"]
        eval_dataset = splits["test"]
        print(f"  按 validation_split_ratio={validation_split_ratio} 自动划分验证集")
        print(f"  训练集: {len(train_dataset)} 条样本")
        print(f"  验证集: {len(eval_dataset)} 条样本")
    else:
        train_dataset = full_dataset
        eval_dataset = None
        print(f"  训练集: {len(train_dataset)} 条样本（未划分验证集）")

    if len(train_dataset) == 0:
        raise ValueError("训练集为空，请检查数据文件与 train_data_ratio 配置")

    # -------------------------------------------
    # Step 3: 预处理数据集（应用 chat template）
    # -------------------------------------------
    print("\n[3/8] 预处理数据集（应用 chat template）")
    system_prompt = model_cfg.get("system_prompt", "你是一个有用的 AI 助手。")

    # GRPOTrainer 在训练前向传播时要求右填充
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["model_name_or_path"],
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def to_chat_prompt(example: dict) -> dict:
        """
        把原始 prompt 包装为 ChatML 格式，并保留生成提示符。

        Args:
            example: 单条样本字典，需含 "prompt" 键。

        Returns:
            仅含 "prompt" 键的字典，值为应用 chat template 后的完整提示串。
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["prompt"]},
        ]
        return {
            "prompt": tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        }

    # 预处理后数据集仅保留两列：prompt（ChatML 提示串）+ references（参考答案）
    drop_cols = [c for c in ("answer", "messages") if c in train_dataset.column_names]
    train_dataset = train_dataset.map(to_chat_prompt, remove_columns=drop_cols)
    if eval_dataset is not None:
        drop_cols_eval = [c for c in ("answer", "messages") if c in eval_dataset.column_names]
        eval_dataset = eval_dataset.map(to_chat_prompt, remove_columns=drop_cols_eval)

    print(f"  ✓ 预处理完成，数据集列: {train_dataset.column_names}")

    # -------------------------------------------
    # Step 4: 加载模型并配置 LoRA
    # -------------------------------------------
    print("\n[4/8] 加载模型并配置 LoRA")
    # 精度与量化开关从 ModelConfig 读取（transformers 5.x 使用 dtype 而非 torch_dtype）
    model_kwargs = {
        "trust_remote_code": True,
        "dtype": resolve_torch_dtype(model_cfg.get("torch_dtype", "bfloat16")),
    }
    if model_cfg.get("load_in_8bit", False):
        model_kwargs["load_in_8bit"] = True
    if model_cfg.get("load_in_4bit", False):
        model_kwargs["load_in_4bit"] = True

    # 复用 Step 3 已加载的 tokenizer，避免重复加载
    _, model = setup_lora(
        model_name_or_path=model_cfg["model_name_or_path"],
        lora_config=lora_cfg,
        tokenizer=tokenizer,
        **model_kwargs,
    )

    # PEFT + 梯度检查点兼容：确保输入嵌入可回传梯度
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    print("  ✓ LoRA 配置完成")

    # -------------------------------------------
    # Step 5: 实例化奖励函数
    # -------------------------------------------
    print("\n[5/8] 实例化奖励函数")
    reward_fn = make_reward_function(reward_cfg)
    print(f"  ✓ 权重配置: {reward_fn.weights}")

    # 冒烟测试：确认奖励函数可正常返回，并打印各分量便于排查接线问题
    if len(prompts) > 0 and len(references) > 0:
        test_reward = reward_fn(
            prompts=[prompts[0]],
            completions=[f"答案是：{references[0]}"],
            references=[references[0]],
        )
        _cache = reward_fn.rewards
        print(
            f"  ✓ 冒烟测试: total={float(test_reward[0]):.4f} | "
            f"length={_cache['length_rewards'][0]:.4f} | "
            f"match={_cache['match_rewards'][0]:.4f}"
        )

    # -------------------------------------------
    # Step 6: 初始化 GRPOTrainer
    # -------------------------------------------
    print("\n[6/8] 初始化 GRPOTrainer")
    train_args = GRPOConfig(**train_cfg)

    # 传入 reward_fn，使回调能在 on_log 时提取奖励分量并注入日志（工作文档 3.5.8）
    log_callback = TrainingLoggerCallback(log_dir, reward_fn=reward_fn)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[log_callback],
    )

    # 移除默认打印回调：它们会把原始指标字典整行 print 出来，
    # 与自定义的单行摘要风格冲突且极其冗长。关键指标由 log_callback 统一输出。
    trainer.callback_handler.callbacks = [
        cb
        for cb in trainer.callback_handler.callbacks
        if not isinstance(cb, (PrinterCallback, ProgressCallback))
    ]
    print("  ✓ Trainer 初始化完成")

    # -------------------------------------------
    # Step 7: 训练模型
    # -------------------------------------------
    print("\n[7/8] 开始训练")
    print("=" * 70)
    try:
        trainer.train()
    except Exception as e:
        traceback.print_exc()
        print(f"\n✗ 训练异常: {e}")
        raise
    print("=" * 70)
    t_train = time.time() - t_start
    print(f"✓ 训练结束，训练耗时: {t_train:.1f} 秒")

    # -------------------------------------------
    # Step 8: 保存最终模型（仅主进程，避免多卡竞态写盘）
    # -------------------------------------------
    print("\n[8/8] 保存最终模型")
    if trainer.is_world_process_zero():
        save_peft_model(model, tokenizer, final_model_dir)
        print(f"  ✓ LoRA adapter 已保存至: {final_model_dir}")
    else:
        print("  （非主进程，跳过保存）")

    print("\n" + "=" * 70)
    print(f"训练管线全部完成！总耗时: {time.time() - t_start:.1f} 秒")
    print("=" * 70)


if __name__ == "__main__":
    main()
