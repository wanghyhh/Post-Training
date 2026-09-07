# =============================================================================
# 模型评估脚本：评估基座模型和微调后模型的生成质量。
# =============================================================================
#
# 用法:
#     python evaluate_model.py
#

# NOTE: pandas 必须首行导入（Windows Trainer 崩溃修复）
import pandas  # noqa: F401

import os
import math
import time
from typing import Any, Dict

import torch
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 从工具脚本导入
from tool_sft import (
    load_config,
    format_messages,
    print_info,
    get_logger,
    get_latest_checkpoint,
)


# =============================================================================
# 评估函数
# =============================================================================

def evaluate_model(
    model,
    tokenizer,
    test_dataset,
    model_name: str = "model",
    batch_size: int = 2,
    max_length: int = 512,
) -> Dict[str, Any]:
    """
    评估模型在测试集上的性能。

    Args:
        model: 待评估的模型
        tokenizer: 对应的 tokenizer
        test_dataset: 测试数据集（Dataset 对象）
        model_name: 模型名称（用于报告）
        batch_size: 评估 batch size
        max_length: 最大序列长度

    Returns:
        包含评估指标和结果的字典:
        {
            "model_name": str,
            "num_samples": int,
            "metrics": Dict[str, float],
            "samples": List[Dict]  # 部分生成样例
        }
    """
    logger = get_logger("evaluate")
    metrics = {"loss": 0.0, "ppl": 0.0}
    samples = []

    model.eval()

    # 逐样本评估
    for i, example in enumerate(test_dataset):
        messages = example["messages"]

        # 获取用户问题（最后一条用户消息）
        user_turn = None
        for msg in reversed(messages):
            if msg["role"] == "user":
                user_turn = msg["content"]
                break

        if user_turn is None:
            continue

        # 渲染完整对话文本用于计算 loss
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        inputs = tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # 计算 loss
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
            metrics["loss"] += loss

        # 生成响应
        prompt_text = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )
        prompt_inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        prompt_inputs = {k: v.to(model.device) for k, v in prompt_inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(
                **prompt_inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_text = tokenizer.decode(
            generated_ids[0][prompt_inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        # 后处理：移除尾部重复的角色标签（模型训练数据含角色标记时，生成会追加 "assistant" 作为下一轮前缀）
        for prefix in ("assistant", "Assistant", "user", "User"):
            while generated_text.lower().endswith(prefix.lower()):
                generated_text = generated_text[: -len(prefix)].rstrip()

        # 采样部分结果（保存原始完整输入和输出）
        if i < 3:
            # 完整对话文本（包含 system/user/assistant 所有角色标记）
            full_conversation = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            # 模型生成前的 prompt（包含 system + user 角色标记）
            model_input = tokenizer.apply_chat_template(
                messages[:-1], tokenize=False, add_generation_prompt=True
            )
            # 模型生成的完整响应（含角色标签前缀）
            full_output = prompt_text + generated_text

            samples.append({
                "question": user_turn,
                "reference": messages[-1]["content"] if messages[-1]["role"] == "assistant" else "",
                "generated": generated_text,
                "full_input": full_conversation,
                "model_input": model_input,
                "model_output": full_output,
            })

    num_samples = len(test_dataset)
    metrics["loss"] = round(metrics["loss"] / max(num_samples, 1), 4)
    metrics["ppl"] = round(math.exp(metrics["loss"]), 4)
    metrics["num_samples"] = num_samples
    metrics["samples"] = samples

    print_info(logger, f"  {model_name}: loss={metrics['loss']:.4f}, ppl={metrics['ppl']:.4f}")

    return metrics


# =============================================================================
# 评估报告输出函数
# =============================================================================

def write_evaluation_report(
    model_name: str,
    metrics: Dict[str, Any],
    output_path: str,
) -> None:
    """
    将评估结果写入 Markdown 报告文件。

    Args:
        model_name: 模型名称
        metrics: 评估指标字典
        output_path: 输出文件路径
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    report_lines = [
        f"# 模型评估报告: {model_name}",
        "",
        "## 评估概览",
        "",
        f"- **模型名称**: {model_name}",
        f"- **评估样本数**: {metrics.get('num_samples', 0)}",
        f"- **评估时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 评估指标",
        "",
        "| 指标 | 值 |",
        "|------|-----|",
        f"| Loss | {metrics.get('loss', 'N/A'):.4f} |",
        f"| Perplexity (PPL) | {metrics.get('ppl', 'N/A'):.4f} |",
        "",
        "## 生成样例",
        "",
    ]

    for idx, sample in enumerate(metrics.get("samples", []), 1):
        report_lines.extend([
            f"### 样例 {idx}",
            "",
            f"**问题**: {sample.get('question', '')}",
            "",
            f"**参考答案**: {sample.get('reference', '')}",
            "",
            f"**模型输入（含角色标签）**:",
            "",
            "```",
            sample.get('model_input', ''),
            "```",
            "",
            f"**模型输出（含角色标签）**:",
            "",
            "```",
            sample.get('model_output', ''),
            "```",
            "",
            "---",
            "",
        ])

    report_text = "\n".join(report_lines)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print_info(get_logger("evaluate"), f"  评估报告已保存至: {output_path}")


# =============================================================================
# 主函数
# =============================================================================

def main():
    """
    主函数：按步骤执行模型评估流程。

    步骤:
        1. 加载配置
        2. 加载测试数据
        3. 评估基座模型（可选）
        4. 评估微调后模型（可选）
    """
    logger = get_logger("evaluate")
    config = load_config("Config.yaml")

    model_cfg = config["ModelConfig"]
    eval_cfg = config["EvalConfig"]
    output_cfg = config["OutputConfig"]

    print_info(logger, "=" * 60)
    print_info(logger, "模型评估启动")
    print_info(logger, f"基座模型: {model_cfg['model_name_or_path']}")
    print_info(logger, "=" * 60)

    # 加载测试数据
    data_cfg = config["DataConfig"]
    print_info(logger, "加载测试数据集...")
    test_dataset = load_dataset(
        "json",
        data_files=data_cfg["test_data_path"],
        split=f"train[:{int(data_cfg['test_data_usage'] * 100)}%]"
        if data_cfg["test_data_usage"] < 1.0
        else "train",
    )
    # 加载 tokenizer（用于格式化数据）
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["model_name_or_path"],
        trust_remote_code=True,
        padding_side="right",
    )
    # 格式化数据（保留 messages 列以便评估函数访问）
    test_dataset = test_dataset.map(
        format_messages,
        desc="Formatting test",
        fn_kwargs={"tokenizer": tokenizer},
        batched=True,
    )
    print_info(logger, f"测试集样本数: {len(test_dataset)}")

    # =========================================================================
    # 步骤 3: 评估基座模型
    # =========================================================================
    if eval_cfg.get("evaluate_base_model", False):
        print_info(logger, "-" * 60)
        print_info(logger, "评估基座模型...")

        # 复用步骤 2 加载的 tokenizer（已在上方初始化）
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            model_cfg["model_name_or_path"],
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map="auto",
        )

        base_metrics = evaluate_model(
            base_model, tokenizer, test_dataset,
            model_name=model_cfg["model_name_or_path"],
            batch_size=eval_cfg.get("eval_batch_size", 2),
            max_length=model_cfg["max_seq_length"],
        )

        base_report_path = os.path.join(
            output_cfg["eval_dir"], "eval_base_model.md"
        )
        write_evaluation_report(
            model_cfg["model_name_or_path"], base_metrics, base_report_path
        )

        del base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # =========================================================================
    # 步骤 4: 评估微调后模型
    # =========================================================================
    if eval_cfg.get("evaluate_fine_tuned_model", False):
        print_info(logger, "-" * 60)
        print_info(logger, "评估微调后模型...")

        ft_model_path = eval_cfg.get("fine_tuned_model_path", "")
        if not ft_model_path or not os.path.exists(ft_model_path):
            ft_model_path = get_latest_checkpoint(output_cfg["checkpoint_dir"])

        if ft_model_path is None:
            print_info(logger, "未找到微调模型检查点，跳过评估")
        else:
            print_info(logger, f"  检查点路径: {ft_model_path}")

            # 复用步骤 2 加载的 tokenizer（已在上方初始化）
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            base_model = AutoModelForCausalLM.from_pretrained(
                model_cfg["model_name_or_path"],
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                device_map="auto",
            )
            ft_model = PeftModel.from_pretrained(
                base_model, ft_model_path, device_map="auto"
            )

            ft_metrics = evaluate_model(
                ft_model, tokenizer, test_dataset,
                model_name=f"fine_tuned({ft_model_path})",
                batch_size=eval_cfg.get("eval_batch_size", 2),
                max_length=model_cfg["max_seq_length"],
            )

            ft_report_path = os.path.join(
                output_cfg["eval_dir"], "eval_fine_tuned_model.md"
            )
            write_evaluation_report(
                ft_model_path.split("/")[-1] if ft_model_path else "fine_tuned",
                ft_metrics, ft_report_path,
            )

            del ft_model, base_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print_info(logger, "=" * 60)
    print_info(logger, "评估完成!")
    print_info(logger, f"  评估报告目录: {output_cfg['eval_dir']}")
    print_info(logger, "=" * 60)


if __name__ == "__main__":
    main()
