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
    评估模型在测试集上的性能（批量推理优化版）。

    将逐样本循环改为分批并行推理：
    - Loss 计算：按 batch_size 分批 tokenize 并前向传播，加权平均
    - 文本生成：按 batch_size 分批调用 model.generate()，充分利用 GPU 并行

    Args:
        model: 待评估的模型
        tokenizer: 对应的 tokenizer
        test_dataset: 测试数据集（Dataset 对象，需含 messages 列）
        model_name: 模型名称（用于报告）
        batch_size: 评估 batch size
        max_length: 最大序列长度

    Returns:
        包含评估指标和结果的字典:
        {
            "loss": float, "ppl": float, "num_samples": int,
            "samples": List[Dict]  # 前 3 条生成样例
        }
    """
    logger = get_logger("evaluate")
    model.eval()

    # ---- 收集所有 messages 和样例元数据 ----
    all_messages = []
    sample_meta = []  # 前 3 条用于报告

    for i, example in enumerate(test_dataset):
        messages = example["messages"]
        all_messages.append(messages)

        if i < 3:
            user_turn = None
            for msg in reversed(messages):
                if msg["role"] == "user":
                    user_turn = msg["content"]
                    break
            sample_meta.append({
                "messages": messages,
                "user_turn": user_turn,
                "reference": messages[-1]["content"] if messages[-1]["role"] == "assistant" else "",
            })

    num_samples = len(all_messages)

    # ---- 批量计算 Loss ----
    total_loss = 0.0
    total_tokens = 0

    for i in range(0, num_samples, batch_size):
        batch_msgs = all_messages[i:i + batch_size]
        batch_texts = [
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            for msgs in batch_msgs
        ]
        inputs = tokenizer(
            batch_texts, return_tensors="pt", truncation=True,
            max_length=max_length, padding=True,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            # outputs.loss 是 batch 内非填充 token 的平均损失
            batch_loss = outputs.loss.item()
            # 统计本 batch 的有效 token 数（非填充）
            batch_tokens = int(inputs["attention_mask"].sum().item())
            total_loss += batch_loss * batch_tokens
            total_tokens += batch_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(avg_loss) if avg_loss < 100 else float("inf")  # 防止溢出

    # ---- 批量生成文本 ----
    all_generated = []
    prompt_texts = [
        tokenizer.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
        for msgs in all_messages
    ]

    # 生成时必须使用 left padding（decoder-only 架构要求，否则影响生成质量）
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    for i in range(0, num_samples, batch_size):
        batch_prompts = prompt_texts[i:i + batch_size]
        prompt_inputs = tokenizer(
            batch_prompts, return_tensors="pt", truncation=True,
            max_length=max_length, padding=True,
        )
        prompt_inputs = {k: v.to(model.device) for k, v in prompt_inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(
                **prompt_inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        # 解码每条生成结果
        for j, ids in enumerate(generated_ids):
            # 用 attention_mask 获取实际 prompt 长度（非 padding 长度）
            input_len = int(prompt_inputs["attention_mask"][j].sum().item())
            # 先保留特殊 token 解码，再手动清理尾部 EOS/PAD
            gen_text = tokenizer.decode(ids[input_len:], skip_special_tokens=False)
            # 去除尾部 EOS token（模型正常结束标志）
            eos_token = tokenizer.eos_token
            if eos_token and gen_text.endswith(eos_token):
                gen_text = gen_text[: -len(eos_token)].rstrip()
            all_generated.append(gen_text)

    tokenizer.padding_side = original_padding_side

    # ---- 构建报告样例 ----
    samples = []
    for idx in range(min(3, len(sample_meta))):
        sd = sample_meta[idx]
        gen_text = all_generated[idx]
        prompt_text = prompt_texts[idx]

        # 后处理：移除尾部重复的角色标签
        for prefix in ("assistant", "Assistant", "user", "User"):
            while gen_text.lower().endswith(prefix.lower()):
                gen_text = gen_text[: -len(prefix)].rstrip()

        full_conversation = tokenizer.apply_chat_template(
            sd["messages"], tokenize=False, add_generation_prompt=False,
        )
        model_input = tokenizer.apply_chat_template(
            sd["messages"][:-1], tokenize=False, add_generation_prompt=True,
        )

        samples.append({
            "question": sd["user_turn"],
            "reference": sd["reference"],
            "generated": gen_text,
            "full_input": full_conversation,
            "model_input": model_input,
            "model_output": prompt_text + gen_text,
        })

    metrics = {
        "loss": round(avg_loss, 4),
        "ppl": round(ppl, 4),
        "num_samples": num_samples,
        "samples": samples,
    }

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
        split=f"train[:{int(data_cfg['test_data_ratio'] * 100)}%]"
        if data_cfg["test_data_ratio"] < 1.0
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
