# ====================================================
# GRPO 模型评估脚本
# 用途：对训练好的模型（LoRA adapter 或合并模型）在测试集上进行评估。
# 用法：
#   python evaluate_model.py                          # 默认配置
#   python evaluate_model.py --checkpoint PATH         # 指定 LoRA adapter 路径
#   python evaluate_model.py --model PATH              # 指定完整模型路径
#   python evaluate_model.py --num-generations 8       # 每个 prompt 生成数
# ====================================================

import os
import sys
import json
import argparse
import time
import torch
import traceback
from pathlib import Path
from datetime import datetime

# ============================================================
# 依赖导入区
# ★ Windows conda 环境 0xC0000005 crash 防护
# ============================================================
import pandas  # noqa: F401

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import yaml
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from tool_grpo import (
    ConfigLoader,
    load_dataset,
    setup_lora,
    make_reward_function,
    merge_and_unload,
    load_hf_model,
)
from reward_function import GRPORewardFunction


# ============================================================
# 辅助函数
# ============================================================

def safe_stat(values: list) -> float:
    """
    安全计算列表的平均值，空列表返回 0。

    Args:
        values: 数值列表

    Returns:
        列表平均值，空列表返回 0
    """
    return float(np.mean(values)) if values else 0.0


# ============================================================
# 评估主函数
# ============================================================

def evaluate_model(
    model,
    tokenizer,
    test_dataset,
    reward_fn,
    eval_params: dict,
    num_generations: int = 4,
    batch_size: int = 8,
    device: str = "auto",
) -> dict:
    """
    评估模型在测试集上的生成质量。

    流程：
    1. 对每个 prompt 进行 N 次生成
    2. 使用奖励函数计算每条生成的奖励
    3. 统计各项指标（avg_reward, Pass@k, reward 分布等）

    Args:
        model: 待评估的模型（支持 PEFT/LoRA 或全量模型）
        tokenizer: 分词器
        test_dataset: 测试数据集（需含 prompt/reference 列）
        reward_fn: GRPORewardFunction 实例
        eval_params: 生成参数（max_new_tokens, temperature, top_p 等）
        num_generations: 每个 prompt 的生成次数
        batch_size: 推理 batch 大小
        device: 设备（"auto"/"cuda"/"cpu"）

    Returns:
        评估结果字典（含 metrics 和 detailed_results）
    """

    print("=" * 70)
    print("  GRPO 模型评估")
    print("=" * 70)
    t_start = time.time()

    # -------------------------------------------
    # 准备测试数据
    # -------------------------------------------
    print(f"\n准备测试数据: {len(test_dataset)} 条样本")

    prompts = test_dataset["prompt"]
    references = test_dataset["reference"] if "reference" in test_dataset.column_names else test_dataset["answer"]

    # 计算 reference token 长度
    ref_token_lengths = []
    for ref_text in references:
        ref_tokens = tokenizer(ref_text, return_tensors="pt", add_special_tokens=False)
        ref_token_lengths.append(len(ref_tokens.input_ids[0]))

    # -------------------------------------------
    # 模型准备
    # -------------------------------------------
    is_peft = hasattr(model, "peft_config")
    print(f"\n模型类型: {'PEFT/LoRA' if is_peft else '全量模型'}")

    model.eval()
    if is_peft and hasattr(model, "set_adapter"):
        model.set_adapter("default")

    # 推理用 left padding
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"

    # 确定设备
    if device == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        dev = device
    model = model.to(dev)

    # -------------------------------------------
    # 批量生成
    # -------------------------------------------
    print(f"\n开始批量生成 (batch_size={batch_size}, 每个 prompt 生成 {num_generations} 次)")

    all_completions = []
    all_prompts = []
    all_references = []
    all_output_ids_list = []
    all_input_lens = []
    total_gen_tokens = 0

    total_prompts = len(prompts)
    total_gens = total_prompts * num_generations

    max_new_tokens = eval_params.get("max_new_tokens", eval_params.get("max_completion_length", 512))
    temperature = eval_params.get("temperature", 0.7)
    top_p = eval_params.get("top_p", 0.9)

    start_gen_time = time.time()

    with torch.no_grad():
        for i in range(0, total_prompts, batch_size):
            batch_end = min(i + batch_size, total_prompts)
            batch_prompts_raw = prompts[i:batch_end]
            batch_refs_raw = references[i:batch_end]
            current_batch = len(batch_prompts_raw)

            # 将每个 prompt 复制 num_generations 次
            flat_prompts = []
            for p in batch_prompts_raw:
                flat_prompts.extend([p] * num_generations)

            inputs = tokenizer(
                flat_prompts,
                padding=True,
                truncation=True,
                max_length=eval_params.get("max_length", 4096),
                return_tensors="pt",
            ).to(dev)

            try:
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=tokenizer.pad_token_id,
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\n[警告] Batch {i//batch_size} OOM，跳过")
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    continue
                raise

            input_len = inputs.input_ids.shape[1]

            for idx in range(len(output_ids)):
                orig_prompt_idx = idx // num_generations
                full_ids = output_ids[idx].detach().cpu()
                all_output_ids_list.append(full_ids)
                all_input_lens.append(input_len)

                completion_ids = full_ids[input_len:]
                completion = tokenizer.decode(completion_ids, skip_special_tokens=True)

                token_count = len(completion_ids)
                total_gen_tokens += token_count

                all_prompts.append(batch_prompts_raw[orig_prompt_idx])
                all_references.append(batch_refs_raw[orig_prompt_idx])
                all_completions.append(completion)

            del inputs, output_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    end_gen_time = time.time()
    gen_duration = end_gen_time - start_gen_time
    tokens_per_sec = total_gen_tokens / gen_duration if gen_duration > 0 else 0

    print(f"  ✓ 生成完成: {len(all_completions)} 条 (耗时 {gen_duration:.2f}s, {tokens_per_sec:.2f} tokens/s)")

    # -------------------------------------------
    # 批量计算奖励
    # -------------------------------------------
    print(f"\n开始批量计算奖励...")
    all_rewards = reward_fn(
        prompts=all_prompts,
        completions=all_completions,
        references=all_references,
    )

    total_rewards_list = all_rewards.tolist()
    length_arr = np.array(reward_fn.rewards.get("length_rewards", np.zeros(len(all_completions))))
    match_arr = np.array(reward_fn.rewards.get("match_rewards", np.zeros(len(all_completions))))

    # 计算每条生成的 token 长度
    completion_lengths = [
        int(len(all_output_ids_list[i]) - all_input_lens[i])
        for i in range(len(all_output_ids_list))
    ]

    # -------------------------------------------
    # 指标计算
    # -------------------------------------------
    print(f"\n计算评估指标...")

    avg_reward = float(np.mean(total_rewards_list))
    reward_std = float(np.std(total_rewards_list))

    # 奖励分布统计
    reward_distribution = {
        "min": float(np.min(total_rewards_list)),
        "p25": float(np.percentile(total_rewards_list, 25)),
        "median": float(np.median(total_rewards_list)),
        "p75": float(np.percentile(total_rewards_list, 75)),
        "max": float(np.max(total_rewards_list)),
    }

    # Pass@k 计算
    # 将每条生成的奖励排序，取 top-1 作为"正确"的标准
    # Pass@k = (k 次生成中至少有一次奖励 > 0.5 的比例)
    all_generations_correct = [r > 0.5 for r in total_rewards_list]
    pass_at_k = {}
    for k in [1, 3, 5, 10]:
        if total_prompts == 0:
            pass_at_k[f"pass@{k}"] = 0.0
            continue
        correct_count = 0
        for p in range(total_prompts):
            gens = all_generations_correct[p * num_generations : (p + 1) * num_generations]
            if any(gens):
                correct_count += 1
        pass_at_k[f"pass@{k}"] = correct_count / total_prompts

    # 每 prompt 平均奖励
    prompt_level_rewards = [
        float(np.mean(total_rewards_list[p * num_generations : (p + 1) * num_generations]))
        for p in range(total_prompts)
        if (p + 1) * num_generations <= len(total_rewards_list)
    ]

    # 每个 prompt 的奖励标准差
    prompt_level_std = [
        float(np.std(total_rewards_list[p * num_generations : (p + 1) * num_generations]))
        for p in range(total_prompts)
        if (p + 1) * num_generations <= len(total_rewards_list)
    ]
    avg_prompt_std = safe_stat(prompt_level_std)

    # 长度统计
    len_arr = np.array(completion_lengths) if completion_lengths else np.array([0])
    avg_length = float(np.mean(len_arr))
    min_length = int(np.min(len_arr)) if len_arr.size > 0 else 0
    max_length = int(np.max(len_arr)) if len_arr.size > 0 else 0

    # -------------------------------------------
    # 输出结果
    # -------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"  GRPO 评估结果")
    print(f"{'=' * 70}")
    print(f"\n  {'指标':<30} {'值':>20}")
    print(f"  {'-' * 55}")
    print(f"  {'平均奖励 (avg_reward)':<30} {avg_reward:>20.4f}")
    print(f"  {'奖励标准差 (reward_std)':<30} {reward_std:>20.4f}")
    print(f"  {'平均生成长度':<30} {avg_length:>20.2f} tokens")
    print(f"  {'长度范围':<30} {min_length:>5} - {max_length:>5} tokens")
    print(f"  {'每 prompt 平均奖励标准差':<30} {avg_prompt_std:>20.4f}")
    print(f"  {'生成速度':<30} {tokens_per_sec:>20.2f} tokens/s")

    print(f"\n  Pass@k 指标:")
    for k_str, val in sorted(pass_at_k.items(), key=lambda x: int(x[0].split("@")[1])):
        print(f"    {k_str:<25} {val * 100:>8.2f}%")

    print(f"\n  奖励分布:")
    for stat_name, stat_val in reward_distribution.items():
        print(f"    {stat_name:<10} {stat_val:>10.4f}")

    print(f"\n  奖励分项:")
    print(f"    长度奖励: 平均={float(np.mean(length_arr)):.4f} (±{float(np.std(length_arr)):.4f})")

    # -------------------------------------------
    # 构建结果字典
    # -------------------------------------------
    eval_result = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "parameters": {
            "test_size": len(prompts),
            "num_generations": num_generations,
            "total_generations": len(all_completions),
            "max_new_tokens": max_new_tokens,
            "batch_size": batch_size,
            "temperature": temperature,
            "top_p": top_p,
        },
        "metrics": {
            "avg_reward": avg_reward,
            "reward_std": reward_std,
            "pass_at_k": pass_at_k,
            "avg_length": avg_length,
            "min_length": min_length,
            "max_length": max_length,
            "reward_distribution": reward_distribution,
            "avg_prompt_std": avg_prompt_std,
            "generation_speed": tokens_per_sec,
            "reward_components": {
                "length": {
                    "mean": float(np.mean(length_arr)),
                    "std": float(np.std(length_arr)),
                },
            },
            "weights": reward_fn.weights,
        },
        "detailed_results": {
            "prompts": list(prompts),
            "references": list(references),
            "completions": all_completions,
            "rewards": [
                {
                    "total": total_rewards_list[i],
                    "length": float(length_arr[i]),
                    "prompt_idx": i // num_generations,
                    "generation_idx": i % num_generations,
                    "token_length": completion_lengths[i],
                }
                for i in range(len(all_completions))
            ],
            "prompt_level_avg_rewards": prompt_level_rewards,
            "prompt_level_std": prompt_level_std,
        },
    }

    t_elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"  评估完成！耗时: {t_elapsed:.1f} 秒")
    print(f"{'=' * 70}\n")

    return eval_result



# ============================================================
# 保存评估报告
# ============================================================

def save_eval_report(eval_result: dict, output_dir: str, tokenizer=None, system_prompt: str = "你是一个有用的 AI 助手。") -> str:
    """
    将评估结果保存为 JSON 和 Markdown 报告。

    文件命名对齐 SFT 规范（无时间戳，固定文件名）：
    - eval_result.json  — 完整评估指标
    - eval_report.md    — 人类可读报告

    Args:
        eval_result: 评估结果字典
        output_dir: 输出目录
        tokenizer: 分词器（用于渲染 chat template，可选）
        system_prompt: 系统提示（用于渲染 chat template，可选）

    Returns:
        Markdown 报告文件路径
    """
    os.makedirs(output_dir, exist_ok=True)

    # 保存 JSON（固定文件名）
    json_path = os.path.join(output_dir, "eval_result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(eval_result, f, ensure_ascii=False, indent=2)

    # 保存 Markdown（固定文件名，对齐 SFT 规范）
    md_path = os.path.join(output_dir, "eval_report.md")
    metrics = eval_result.get("metrics", {})
    params = eval_result.get("parameters", {})
    detailed = eval_result.get("detailed_results", {})

    # 获取 prompts（优先使用带 chat template 的 formatted_prompt，否则回退到原始 prompt）
    prompts = detailed.get("formatted_prompts", detailed.get("prompts", []))
    completions = detailed.get("completions", [])
    rewards_detail = detailed.get("rewards", [])
    references = detailed.get("references", [])
    num_gen = params.get("num_generations", 4)

    md_lines = [
        "# GRPO 模型评估报告",
        f"\n> 评估时间: {eval_result.get('timestamp', 'N/A')}\n",
        "## 评估参数",
        "",
        "| 参数 | 值 |",
        "|------|-----|",
        f"| 测试集大小 | {params.get('test_size', 0)} |",
        f"| 每个 Prompt 生成数 | {params.get('num_generations', 0)} |",
        f"| 生成总数 | {params.get('total_generations', 0)} |",
        f"| 最大生成长度 | {params.get('max_new_tokens', 0)} tokens |",
        f"| Batch Size | {params.get('batch_size', 0)} |",
        f"| Temperature | {params.get('temperature', 0)} |",
        f"| Top-P | {params.get('top_p', 0)} |",
        "",
        "## 评估指标",
        "",
        "| 指标 | 值 |",
        "|------|------|",
        f"| 平均奖励 | {metrics.get('avg_reward', 0):.4f} |",
        f"| 奖励标准差 | {metrics.get('reward_std', 0):.4f} |",
        f"| 平均长度 | {metrics.get('avg_length', 0):.2f} tokens |",
        f"| 长度范围 | {metrics.get('min_length', 0)} - {metrics.get('max_length', 0)} |",
        f"| 生成速度 | {metrics.get('generation_speed', 0):.2f} tokens/s |",
        "",
        "### Pass@k",
        "",
        "| 指标 | 通过率 |",
        "|------|------|",
    ]

    for k_key, k_val in sorted(
        metrics.get("pass_at_k", {}).items(),
        key=lambda x: int(x[0].split("@")[1]),
    ):
        md_lines.append(f"| {k_key} | {k_val * 100:.2f}% |")

    # 奖励分布
    dist = metrics.get("reward_distribution", {})
    md_lines.extend([
        "",
        "### 奖励分布",
        "",
        "| 统计项 | 值 |",
        "|--------|------|",
        f"| 最小值 | {dist.get('min', 0):.4f} |",
        f"| 25% | {dist.get('p25', 0):.4f} |",
        f"| 中位数 | {dist.get('median', 0):.4f} |",
        f"| 75% | {dist.get('p75', 0):.4f} |",
        f"| 最大值 | {dist.get('max', 0):.4f} |",
        "",
    ])

    # 奖励分项统计
    reward_comps = metrics.get("reward_components", {})
    length_comp = reward_comps.get("length", {})
    md_lines.extend([
        "### 奖励分项统计",
        "",
        "| 分项 | 平均值 | 标准差 |",
        "|------|--------|--------|",
        f"| 长度奖励 | {length_comp.get('mean', 0):.4f} | {length_comp.get('std', 0):.4f} |",
        "",
    ])
    syntax_comp = reward_comps.get("syntax", {})
    if syntax_comp:
        md_lines.append(f"| 语法奖励 | {syntax_comp.get('mean', 0):.4f} | {syntax_comp.get('std', 0):.4f} |")

    # 生成样例展示（工作文档第 3.7.78 条：展示完整 chat template 渲染结果）
    md_lines.extend([
        "## 详细生成结果 (前 5 条)",
        "",
    ])

    raw_prompts = detailed.get("prompts", [])

    for p_idx in range(min(5, len(prompts))):
        # 工作文档第 78 条：必须展示包含 system/user/assistant 角色标签的完整文本
        # 尝试应用 chat template 渲染
        raw_prompt = raw_prompts[p_idx] if p_idx < len(raw_prompts) else prompts[p_idx]
        if tokenizer and system_prompt:
            try:
                # 从原始 prompt 重建 messages，应用 chat template
                msg_list = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": raw_prompt},
                ]
                full_prompt_text = tokenizer.apply_chat_template(
                    msg_list,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                display_text = full_prompt_text
            except Exception:
                display_text = raw_prompt
        else:
            display_text = raw_prompt

        md_lines.extend([
            f"\n### Prompt #{p_idx + 1}",
            "",
            f"**完整输入（含角色标签）**:",
            f"```text",
            f"{display_text[:1500]}{'...' if len(display_text) > 1500 else ''}",
            f"```",
            f"**参考答案**:",
            f"```text",
            f"{references[p_idx][:300] if p_idx < len(references) else 'N/A'}",
            f"```",
        ])

        for g_idx in range(num_gen):
            idx = p_idx * num_gen + g_idx
            if idx < len(rewards_detail) and idx < len(completions):
                r = rewards_detail[idx]
                # 后处理：去除尾部多余的 assistant 标签前缀
                completion_text = completions[idx]
                # 保留 assistant 标签但 strip 尾部多余重复
                if completion_text.strip().startswith("<|assistant|>"):
                    completion_text = completion_text.strip()

                md_lines.extend([
                    f"\n#### 生成 #{g_idx + 1}",
                    "",
                    f"| 属性 | 值 |",
                    f"|------|------|",
                    f"| 总奖励 | {r['total']:.4f} |",
                    f"| 长度奖励 | {r.get('length', 0):.4f} |",
                    f"| Token 长度 | {r.get('token_length', 0)} |",
                    "",
                    f"**生成内容**:",
                    f"```text",
                    f"{completion_text[:500]}{'...' if len(completion_text) > 500 else ''}",
                    f"```",
                ])

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"  ✓ JSON 报告: {json_path}")
    print(f"  ✓ Markdown 报告: {md_path}")

    return json_path


# ============================================================
# 命令行入口
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="GRPO 模型评估脚本")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(_SCRIPT_DIR, "Config.yaml"),
        help="Config.yaml 路径（默认当前目录下的 Config.yaml）",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="LoRA adapter 路径（可选，不提供则使用基座模型）",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="完整模型路径（可选，优先级高于 checkpoint）",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=4,
        help="每个 prompt 的生成次数（默认 4）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="推理 batch 大小（默认 4）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="设备选择: auto/cuda/cpu（默认 auto）",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default=None,
        help="测试数据路径（默认从 Config.yaml 读取）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="评估输出目录（默认 output/eval_results）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # -------------------------------------------
    # 加载配置
    # -------------------------------------------
    config_path = args.config
    if not os.path.exists(config_path):
        print(f"✗ 配置文件不存在: {config_path}")
        return

    print("=" * 70)
    print("  GRPO 模型评估管线")
    print("=" * 70)

    loader = ConfigLoader(config_path)
    all_config = loader.load_all()

    model_cfg = all_config.get("ModelConfig", {})
    data_cfg = all_config.get("DataSetConfig", {})
    lora_cfg = all_config.get("LoraParamConfig", {})
    reward_cfg = all_config.get("RewardFuncConfig", {})
    eval_cfg = all_config.get("EvalConfig", {})

    # -------------------------------------------
    # 确定模型路径和加载
    # -------------------------------------------
    base_model_path = model_cfg.get("model_name_or_path", "")
    model_to_use = args.model or base_model_path

    print(f"\n基座模型: {base_model_path}")
    print(f"使用模型: {model_to_use}")

    # 加载分词器
    print(f"加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型
    if args.model:
        # 使用完整模型路径
        print(f"加载完整模型: {args.model}")
        model, _ = load_hf_model(args.model, trust_remote_code=True)
    elif args.checkpoint:
        # 加载基座模型 + LoRA adapter
        print(f"加载 LoRA adapter: {args.checkpoint}")
        base_model, _ = load_hf_model(
            base_model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(base_model, args.checkpoint)
    else:
        # 仅基座模型
        print(f"仅使用基座模型")
        model, _ = load_hf_model(base_model_path, trust_remote_code=True)

    # -------------------------------------------
    # 加载测试数据
    # -------------------------------------------
    # 兼容两种配置风格：平铺键名 test_data_path vs 嵌套字典 test.path
    test_path = getattr(args, "test_data", None)
    if not test_path:
        test_path = data_cfg.get("test_data_path", "")
    if not test_path:
        if isinstance(data_cfg.get("test"), dict):
            test_path = data_cfg["test"].get("path", "")
        elif data_cfg.get("test"):
            test_path = str(data_cfg["test"])

    if not test_path or not os.path.exists(test_path):
        print(f"✗ 测试数据不存在: {test_path}")
        return

    print(f"加载测试数据: {test_path}")
    prompts, answers, references = load_dataset(test_path)
    test_data = Dataset.from_dict({
        "prompt": prompts,
        "answer": answers,
        "reference": references,
    })

    # -------------------------------------------
    # 构建 chat template 格式的 prompt（工作文档第 78 条）
    # -------------------------------------------
    system_prompt = model_cfg.get("system_prompt", "你是一个有用的 AI 助手。")
    formatted_prompts = []
    for p in prompts:
        try:
            msg_list = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": p},
            ]
            formatted = tokenizer.apply_chat_template(
                msg_list,
                tokenize=False,
                add_generation_prompt=True,
            )
            formatted_prompts.append(formatted)
        except Exception:
            # 回退到原始 prompt
            formatted_prompts.append(p)

    # 将 formatted_prompts 存入 test_data
    test_data = test_data.add_column("formatted_prompt", formatted_prompts)

    # -------------------------------------------
    # 实例化奖励函数
    # -------------------------------------------
    reward_fn = make_reward_function(reward_cfg)
    print(f"奖励函数: weights={reward_fn.weights}")

    # -------------------------------------------
    # 构建评估参数
    # -------------------------------------------
    eval_params = {
        "max_new_tokens": eval_cfg.get("max_new_tokens", eval_cfg.get("max_completion_length", 512)),
        "max_completion_length": eval_cfg.get("max_completion_length", 512),
        "temperature": eval_cfg.get("temperature", 0.7),
        "top_p": eval_cfg.get("top_p", 0.9),
        "max_length": eval_cfg.get("max_length", 4096),
    }

    # -------------------------------------------
    # 执行评估
    # -------------------------------------------
    # 从 OutputConfig 读取评估输出路径
    output_dir = args.output_dir or output_cfg.get("eval_dir") or os.path.join(_SCRIPT_DIR, "output", "eval")

    try:
        eval_result = evaluate_model(
            model=model,
            tokenizer=tokenizer,
            test_dataset=test_data,
            reward_fn=reward_fn,
            eval_params=eval_params,
            num_generations=args.num_generations,
            batch_size=args.batch_size,
            device=args.device,
        )
    except Exception as e:
        traceback.print_exc()
        print(f"\n✗ 评估异常: {e}")
        return

    # -------------------------------------------
    # 保存报告
    # -------------------------------------------
    # 将 formatted_prompts 传入 eval_result 供报告使用
    eval_result["detailed_results"]["formatted_prompts"] = formatted_prompts

    save_eval_report(eval_result, output_dir, tokenizer=tokenizer, system_prompt=system_prompt)

    print(f"\n{'=' * 70}")
    print("评估管线完成！")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
