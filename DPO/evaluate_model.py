"""
DPO 模型评估脚本

完全遵循 DPO/工作文档.md 的各项要求：
  - 基于隐式奖励的 DPO 偏好判别评估
  - 相对参考模型视角 + 策略自身概率视角 双视角
  - 分 Batch 并行前向 + OOM 回退
  - 基座模型与微调后模型对照输出
  - 带完整 chat 模板渲染的样例展示
"""

import os
import sys
import json
import time
import math
from datetime import datetime
from collections import Counter

import torch
import torch.nn.functional as F
import yaml
import numpy as np
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# 辅助函数
# ============================================================

def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_tokenizer(model_path: str, trust_remote_code: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(model_path: str, torch_dtype: str = "bfloat16",
                    trust_remote_code: bool = False):
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                 "float32": torch.float32, "auto": "auto"}
    dtype = dtype_map.get(torch_dtype, torch.bfloat16)
    print(f"  加载基座模型: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype,
        trust_remote_code=trust_remote_code, device_map="auto",
    )
    model.eval()
    return model


def load_tuned_model(base_model_path: str, adapter_path: str,
                     torch_dtype: str = "bfloat16", trust_remote_code: bool = False):
    """加载基座 + LoRA 适配器作为微调模型。"""
    print(f"  加载微调模型: {base_model_path} + 适配器 {adapter_path}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16,
        trust_remote_code=trust_remote_code, device_map="auto",
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return model


# ============================================================
# 对数概率计算（核心评估逻辑）
# ============================================================

def compute_completion_log_probs(
    model, tokenizer, prompt_messages, chosen_messages, rejected_messages,
    batch_size: int = 4, max_length: int = 1024,
):
    """
    批量计算 chosen 和 rejected 回答的对数概率。

    对每个偏好对，分别计算 logπ(chosen|prompt) 和 logπ(rejected|prompt)，
    即给定 prompt 后，chosen/rejected 回答的对数条件概率。

    采用与 DPOTrainer 一致的 token 级处理方法：
      - prompt_ids = apply_chat_template(prompt, add_generation_prompt=True)
      - full_ids    = apply_chat_template(prompt + response)
      - completion  = full_ids[len(prompt_ids):]（保证前缀一致）

    支持分 Batch 前向 + OOM 自动回退。

    Returns:
        chosen_log_probs: list[float]
        rejected_log_probs: list[float]
        chosen_token_counts: list[int]
        rejected_token_counts: list[int]
        skipped_pairs: int
    """
    device = model.device
    n = len(prompt_messages)
    chosen_log_probs = []
    rejected_log_probs = []
    chosen_token_counts = []
    rejected_token_counts = []
    skipped = 0

    # 预计算每个样本的 prompt_ids / full_chosen_ids / full_rejected_ids
    sample_ids = []
    for i in range(n):
        prompt_result = tokenizer.apply_chat_template(
            prompt_messages[i], tokenize=True, add_generation_prompt=True,
            return_dict=True,
        )
        full_chosen_result = tokenizer.apply_chat_template(
            prompt_messages[i] + chosen_messages[i], tokenize=True,
            return_dict=True,
        )
        full_rejected_result = tokenizer.apply_chat_template(
            prompt_messages[i] + rejected_messages[i], tokenize=True,
            return_dict=True,
        )
        sample_ids.append((
            prompt_result["input_ids"],
            full_chosen_result["input_ids"],
            full_rejected_result["input_ids"],
        ))

    def _process_batch(ids_list, is_chosen):
        """对一批 full_ids 计算 completion 部分的对数概率之和。"""
        batch_logps = []
        batch_token_counts = []
        # 手动 padding（transformers 5.x 的 tokenizer.pad 入参格式已变化）
        # 根据 is_chosen 选择 chosen 或 rejected 的 ids
        seqs = [
            (full_chosen if is_chosen else full_rejected)[:max_length]
            for _, full_chosen, full_rejected in ids_list
        ]
        batch_prompt_lens = [
            min(len(prompt_ids), max_length)
            for prompt_ids, _, _ in ids_list
        ]
        max_len = max(len(s) for s in seqs)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        input_ids_t = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
        attention_mask_t = torch.zeros((len(seqs), max_len), dtype=torch.long)
        for i, s in enumerate(seqs):
            input_ids_t[i, :len(s)] = torch.tensor(s, dtype=torch.long)
            attention_mask_t[i, :len(s)] = 1

        input_ids_t = input_ids_t.to(device)
        attention_mask_t = attention_mask_t.to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids_t, attention_mask=attention_mask_t)
            logits = outputs.logits  # [batch, seq_len, vocab]
            log_probs = F.log_softmax(logits, dim=-1)

            for idx_in_batch in range(len(seqs)):
                prompt_len = batch_prompt_lens[idx_in_batch]
                shift_log_probs = log_probs[idx_in_batch]      # [seq_len, vocab]
                mask = attention_mask_t[idx_in_batch].bool()
                valid_positions = torch.where(mask)[0]
                # completion 位置：>= prompt_len
                comp_positions = valid_positions[valid_positions >= prompt_len]

                if len(comp_positions) == 0:
                    batch_logps.append(float("-inf"))
                    batch_token_counts.append(0)
                else:
                    # 位置 p 的概率由 shift_log_probs[p-1] 预测 input_ids[p]
                    pos_prev = comp_positions - 1
                    keep = pos_prev >= 0
                    pos_prev = pos_prev[keep]
                    target_tokens = input_ids_t[idx_in_batch][pos_prev + 1]
                    token_logps = shift_log_probs[pos_prev].gather(
                        -1, target_tokens.unsqueeze(-1)
                    ).squeeze(-1)
                    batch_logps.append(token_logps.sum().item())
                    batch_token_counts.append(len(pos_prev))

        return batch_logps, batch_token_counts

    current_batch_size = batch_size
    start = 0
    while start < n:
        end = min(start + current_batch_size, n)
        batch_items = sample_ids[start:end]

        try:
            chosen_logps_b, chosen_tokens_b = _process_batch(batch_items, True)
            rejected_logps_b, rejected_tokens_b = _process_batch(batch_items, False)
            chosen_log_probs.extend(chosen_logps_b)
            rejected_log_probs.extend(rejected_logps_b)
            chosen_token_counts.extend(chosen_tokens_b)
            rejected_token_counts.extend(rejected_tokens_b)
            start = end
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                if current_batch_size <= 1:
                    print(f"  OOM (batch_size=1) 跳过样本 {start}~{end - 1}")
                    for _ in range(end - start):
                        chosen_log_probs.append(float("-inf"))
                        rejected_log_probs.append(float("-inf"))
                        chosen_token_counts.append(0)
                        rejected_token_counts.append(0)
                    skipped += (end - start)
                    start = end
                    continue
                current_batch_size = max(1, current_batch_size // 2)
                print(f"  OOM: batch_size 减至 {current_batch_size}，重试")
                continue
            else:
                raise

    return chosen_log_probs, rejected_log_probs, chosen_token_counts, rejected_token_counts, skipped


# ============================================================
# 评估函数
# ============================================================

def evaluate_model(model, tokenizer, dataset, beta=0.1, batch_size=4,
                   max_length=1024, ref_log_probs=None):
    """
    对偏好数据集执行 DPO 评估。

    参数:
        model: 待评估的模型
        tokenizer: tokenizer
        dataset: list of dicts with keys "prompt", "chosen", "rejected" (messages list)
        beta: DPO beta 参数（用于隐式奖励计算）
        batch_size: 批量大小
        max_length: 最大序列长度
        ref_log_probs: 参考模型的对数概率（相对参考模型视角用）

    Returns:
        metrics: dict 包含所有评估指标
    """
    prompt_messages = [item["prompt"] for item in dataset]
    chosen_messages = [item["chosen"] for item in dataset]
    rejected_messages = [item["rejected"] for item in dataset]

    start_time = time.time()

    # 计算策略模型的对数概率
    chosen_logps, rejected_logps, chosen_tokens, rejected_tokens, skipped = \
        compute_completion_log_probs(
            model, tokenizer, prompt_messages, chosen_messages, rejected_messages,
            batch_size=batch_size, max_length=max_length,
        )

    elapsed = time.time() - start_time

    n = len(chosen_logps)
    valid_mask = [not (math.isinf(c) or math.isinf(r)) for c, r in zip(chosen_logps, rejected_logps)]
    n_valid = sum(valid_mask)

    if n_valid == 0:
        print("警告: 所有样本均无效（均为 -inf），请检查模型和数据")
        return {"error": "all_samples_invalid"}

    # 将 valid 样本转换为 array
    chosen_arr = np.array([chosen_logps[i] for i in range(n) if valid_mask[i]])
    rejected_arr = np.array([rejected_logps[i] for i in range(n) if valid_mask[i]])
    chosen_tok_arr = np.array([chosen_tokens[i] for i in range(n) if valid_mask[i]])
    rejected_tok_arr = np.array([rejected_tokens[i] for i in range(n) if valid_mask[i]])

    # ---- 策略自身概率视角 ----
    policy_margins = chosen_arr - rejected_arr  # logπ(chosen) - logπ(rejected)
    policy_accuracy = np.mean(policy_margins > 0)

    # ---- 相对参考模型视角（如提供 ref_log_probs）----
    # 语义说明：当待评估模型与参考模型为同一份权重（评估基座模型）时，
    # logπ_θ ≡ logπ_ref，隐式奖励恒为 0，该视角无判别信息，标记为不可用。
    is_self_reference = False
    if ref_log_probs is not None:
        ref_chosen = np.array([ref_log_probs["chosen"][i] for i in range(n) if valid_mask[i]])
        ref_rejected = np.array([ref_log_probs["rejected"][i] for i in range(n) if valid_mask[i]])
        # 判断是否为"自己参考自己"：对数概率逐条完全相同
        is_self_reference = bool(
            np.allclose(chosen_arr, ref_chosen) and np.allclose(rejected_arr, ref_rejected)
        )
        # 隐式奖励 = beta × (logπ_θ − logπ_ref)
        chosen_rewards = beta * (chosen_arr - ref_chosen)
        rejected_rewards = beta * (rejected_arr - ref_rejected)
        reward_margins = chosen_rewards - rejected_rewards
        reward_accuracy = np.mean(reward_margins > 0) if not is_self_reference else None
    else:
        chosen_rewards = None
        rejected_rewards = None
        reward_margins = None
        reward_accuracy = None

    # ---- 统计指标 ----
    metrics = {}

    # 样本信息
    prompt_token_lens = [
        len(tokenizer.apply_chat_template(p, tokenize=True, add_generation_prompt=True))
        for p in prompt_messages
    ]
    metrics["num_samples"] = n
    metrics["num_valid"] = n_valid
    metrics["num_skipped"] = skipped
    metrics["avg_prompt_token_len"] = float(np.mean(prompt_token_lens))
    metrics["avg_chosen_token_len"] = float(np.mean(chosen_tok_arr)) if len(chosen_tok_arr) > 0 else 0
    metrics["avg_rejected_token_len"] = float(np.mean(rejected_tok_arr)) if len(rejected_tok_arr) > 0 else 0
    metrics["eval_time_sec"] = round(elapsed, 2)
    metrics["eval_speed_pairs_per_sec"] = round(n / elapsed, 2) if elapsed > 0 else 0

    # 策略自身概率视角
    metrics["policy_accuracy"] = float(policy_accuracy)
    metrics["policy_avg_chosen_logp"] = float(np.mean(chosen_arr))
    metrics["policy_avg_rejected_logp"] = float(np.mean(rejected_arr))
    metrics["policy_avg_margin"] = float(np.mean(policy_margins))
    metrics["policy_margin_std"] = float(np.std(policy_margins))
    # margin 分布
    metrics["policy_margin_min"] = float(np.min(policy_margins))
    metrics["policy_margin_25p"] = float(np.percentile(policy_margins, 25))
    metrics["policy_margin_median"] = float(np.median(policy_margins))
    metrics["policy_margin_75p"] = float(np.percentile(policy_margins, 75))
    metrics["policy_margin_max"] = float(np.max(policy_margins))

    # 相对参考模型视角
    metrics["is_self_reference"] = is_self_reference
    if reward_margins is not None and not is_self_reference:
        metrics["ref_accuracy"] = float(reward_accuracy)
        metrics["ref_avg_chosen_reward"] = float(np.mean(chosen_rewards))
        metrics["ref_avg_rejected_reward"] = float(np.mean(rejected_rewards))
        metrics["ref_avg_margin"] = float(np.mean(reward_margins))
        metrics["ref_margin_std"] = float(np.std(reward_margins))
        metrics["ref_margin_min"] = float(np.min(reward_margins))
        metrics["ref_margin_25p"] = float(np.percentile(reward_margins, 25))
        metrics["ref_margin_median"] = float(np.median(reward_margins))
        metrics["ref_margin_75p"] = float(np.percentile(reward_margins, 75))
        metrics["ref_margin_max"] = float(np.max(reward_margins))

    # 判定分布（相对参考模型视角；自参考时该视角无信息，跳过）
    if reward_margins is not None and not is_self_reference:
        ties = 1e-8  # 极小阈值，认为 margin 绝对值 < 该值的为打平
        wins = np.sum(reward_margins > ties)
        losses = np.sum(reward_margins < -ties)
        ties_count = np.sum(np.abs(reward_margins) <= ties)
        metrics["ref_judgment_correct"] = int(wins)
        metrics["ref_judgment_wrong"] = int(losses)
        metrics["ref_judgment_tie"] = int(ties_count)

    # 存原始数据供样例展示
    _has_ref_view = (chosen_rewards is not None) and (not is_self_reference)
    metrics["_raw_data"] = {
        "prompt_messages": [prompt_messages[i] for i in range(n)],
        "chosen_messages": [chosen_messages[i] for i in range(n)],
        "rejected_messages": [rejected_messages[i] for i in range(n)],
        "chosen_logps": chosen_logps,
        "rejected_logps": rejected_logps,
        "chosen_tokens": chosen_tokens,
        "rejected_tokens": rejected_tokens,
        "ref_chosen_rewards": [float(x) for x in chosen_rewards] if _has_ref_view else None,
        "ref_rejected_rewards": [float(x) for x in rejected_rewards] if _has_ref_view else None,
        "ref_margins": [float(x) for x in reward_margins] if _has_ref_view else None,
        "policy_margins": [float(x) for x in policy_margins],
        "valid_mask": valid_mask,
    }
    return metrics


# ============================================================
# 评估报告输出
# ============================================================

def write_evaluation_report(model_name: str, metrics: dict, tokenizer,
                            output_path: str, num_samples_to_show: int = 3):
    """
    将评估指标和样例写入 markdown 报告。
    """
    lines = []
    lines.append(f"# DPO 模型评估报告\n")
    lines.append(f"**模型**: {model_name}")
    lines.append(f"**评估时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    if "error" in metrics:
        lines.append(f"## 评估失败\n{metrics['error']}")
        lines.append("\n---\n")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return

    # 概要
    lines.append(f"## 评估概要\n")
    lines.append(f"- 偏好对样本数: {metrics['num_samples']} (有效: {metrics['num_valid']})")
    lines.append(f"- 评估耗时: {metrics['eval_time_sec']} 秒")
    lines.append(f"- 评估速度: {metrics['eval_speed_pairs_per_sec']} pairs/s")
    lines.append(f"- 平均 Prompt 长度: {metrics.get('avg_prompt_token_len', metrics.get('avg_prompt_len', 0)):.1f} tokens")
    lines.append(f"- 平均 Chosen 回答长度: {metrics['avg_chosen_token_len']:.1f} tokens")
    lines.append(f"- 平均 Rejected 回答长度: {metrics['avg_rejected_token_len']:.1f} tokens")

    # 策略自身概率视角
    lines.append(f"\n## 策略自身概率视角\n")
    lines.append(f"- 偏好准确率: {metrics['policy_accuracy']:.4f} ({metrics['policy_accuracy']*100:.1f}%)")
    lines.append(f"- 平均 Chosen log_prob: {metrics['policy_avg_chosen_logp']:.4f}")
    lines.append(f"- 平均 Rejected log_prob: {metrics['policy_avg_rejected_logp']:.4f}")
    lines.append(f"- 平均概率间隔: {metrics['policy_avg_margin']:.4f}")
    lines.append(f"- 概率间隔标准差: {metrics['policy_margin_std']:.4f}")
    lines.append(f"- 概率间隔分布: min={metrics['policy_margin_min']:.4f}, "
                 f"25%={metrics['policy_margin_25p']:.4f}, "
                 f"50%={metrics['policy_margin_median']:.4f}, "
                 f"75%={metrics['policy_margin_75p']:.4f}, "
                 f"max={metrics['policy_margin_max']:.4f}")

    # 相对参考模型视角
    if "ref_accuracy" in metrics:
        lines.append(f"\n## 相对参考模型视角 (beta={metrics.get('_beta', '?')})\n")
        lines.append(f"- 偏好准确率: {metrics['ref_accuracy']:.4f} ({metrics['ref_accuracy']*100:.1f}%)")
        lines.append(f"- 平均 Chosen 隐式奖励: {metrics['ref_avg_chosen_reward']:.6f}")
        lines.append(f"- 平均 Rejected 隐式奖励: {metrics['ref_avg_rejected_reward']:.6f}")
        lines.append(f"- 平均奖励间隔: {metrics['ref_avg_margin']:.6f}")
        lines.append(f"- 奖励间隔标准差: {metrics['ref_margin_std']:.6f}")
        lines.append(f"- 奖励间隔分布: min={metrics['ref_margin_min']:.6f}, "
                     f"25%={metrics['ref_margin_25p']:.6f}, "
                     f"50%={metrics['ref_margin_median']:.6f}, "
                     f"75%={metrics['ref_margin_75p']:.6f}, "
                     f"max={metrics['ref_margin_max']:.6f}")

        # 判定分布
        lines.append(f"\n## 判定分布 (相对参考模型视角)\n")
        total = metrics['ref_judgment_correct'] + metrics['ref_judgment_wrong'] + metrics['ref_judgment_tie']
        lines.append(f"- 判定正确: {metrics['ref_judgment_correct']} ({metrics['ref_judgment_correct']/total*100:.1f}%)")
        lines.append(f"- 判定错误: {metrics['ref_judgment_wrong']} ({metrics['ref_judgment_wrong']/total*100:.1f}%)")
        lines.append(f"- 判定持平: {metrics['ref_judgment_tie']} ({metrics['ref_judgment_tie']/total*100:.1f}%)")
    elif metrics.get("is_self_reference"):
        # 待评估模型与参考模型同权重（基座模型），相对参考模型视角无判别信息
        lines.append(f"\n## 相对参考模型视角\n")
        lines.append("- 说明: 当前模型与参考模型为同一份权重（基座模型），隐式奖励恒为 0，该视角无判别信息；")
        lines.append("  偏好判别能力以「策略自身概率视角」为准。")

    # 样例展示
    lines.append(f"\n## 偏好对样例展示\n")
    raw = metrics.get("_raw_data", {})
    if raw.get("prompt_messages"):
        # 按 margin 绝对值排序，取代表性样例
        if raw.get("ref_margins"):
            margins = raw["ref_margins"]
        else:
            margins = raw["policy_margins"]

        valid_mask = raw.get("valid_mask", [True] * len(margins))
        indices = [
            i for i, (m, v) in enumerate(zip(margins, valid_mask))
            if v and not math.isinf(m)
        ]
        # 排序：正确判定(大正margin)和错误判定(大负margin)各展示一些
        indices.sort(key=lambda i: abs(margins[i]), reverse=True)
        # 取一半正确一半错误的
        correct = [i for i in indices if margins[i] > 0]
        wrong = [i for i in indices if margins[i] < 0]
        sampled = correct[:num_samples_to_show] + wrong[:num_samples_to_show]
        # 如果不够再从尾部补
        if len(sampled) < num_samples_to_show:
            sampled = indices[:num_samples_to_show * 2]
        sampled = sampled[:num_samples_to_show * 2]

        for idx in sampled:
            margin_val = margins[idx]
            decision = "✅ 正确" if margin_val > 0 else "❌ 错误"
            chosen_r = raw["ref_chosen_rewards"][idx] if raw.get("ref_chosen_rewards") else "N/A"
            rejected_r = raw["ref_rejected_rewards"][idx] if raw.get("ref_rejected_rewards") else "N/A"

            # 渲染完整文本
            prompt_text = tokenizer.apply_chat_template(
                raw["prompt_messages"][idx], tokenize=False, add_generation_prompt=True
            )
            chosen_text = tokenizer.apply_chat_template(
                raw["prompt_messages"][idx] + raw["chosen_messages"][idx], tokenize=False
            )
            rejected_text = tokenizer.apply_chat_template(
                raw["prompt_messages"][idx] + raw["rejected_messages"][idx], tokenize=False
            )

            lines.append(f"### 偏好对 #{idx} ({decision})\n")
            lines.append(f"- 奖励间隔: {margin_val:.6f}")
            if raw.get("ref_chosen_rewards"):
                lines.append(f"- Chosen 隐式奖励: {chosen_r:.6f}")
                lines.append(f"- Rejected 隐式奖励: {rejected_r:.6f}")
            lines.append(f"- Chosen log_prob: {raw['chosen_logps'][idx]:.4f}")
            lines.append(f"- Rejected log_prob: {raw['rejected_logps'][idx]:.4f}")
            lines.append(f"- Chosen token 数: {raw['chosen_tokens'][idx]}")
            lines.append(f"- Rejected token 数: {raw['rejected_tokens'][idx]}")

            lines.append(f"\n**Prompt (完整模板渲染):**\n```text\n{prompt_text}\n```\n")
            lines.append(f"**Chosen (完整模板渲染):**\n```text\n{chosen_text}\n```\n")
            lines.append(f"**Rejected (完整模板渲染):**\n```text\n{rejected_text}\n```\n")

    lines.append("\n---\n")
    lines.append("*报告由 evaluate_model.py 自动生成*\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"评估报告已保存: {output_path}")


# ============================================================
# 对比报告输出
# ============================================================

def write_comparison_report(base_metrics, tuned_metrics, output_path,
                            model_name="微调后模型", base_name="基座模型",
                            beta=0.1):
    """
    输出基座模型与微调后模型的指标对照表。
    """
    lines = []
    lines.append(f"# DPO 模型对比评估报告\n")
    lines.append(f"**基座模型**: {base_name}")
    lines.append(f"**微调后模型**: {model_name}")
    lines.append(f"**评估时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**DPO beta**: {beta}\n")

    if "error" in base_metrics and "error" in tuned_metrics:
        lines.append("两个模型评估均失败，请检查数据与模型路径。")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return

    # 对照表
    lines.append(f"## 指标对照表\n")
    lines.append("| 指标 | 基座模型 | 微调后模型 | Δ 变化 |")
    lines.append("|---|---|---|---|")

    # 策略自身概率视角
    def add_row(name, base_val, tuned_val, fmt=".4f"):
        base_s = f"{base_val:{fmt}}" if base_val is not None else "N/A"
        tuned_s = f"{tuned_val:{fmt}}" if tuned_val is not None else "N/A"
        if base_val is not None and tuned_val is not None:
            delta = tuned_val - base_val
            delta_s = f"{delta:{fmt}}"
            # 用箭头表示方向
            if delta > 0:
                delta_s += " ↑"
            elif delta < 0:
                delta_s += " ↓"
            else:
                delta_s += "  —"
            lines.append(f"| {name} | {base_s} | {tuned_s} | {delta_s} |")
        else:
            lines.append(f"| {name} | {base_s} | {tuned_s} | N/A |")

    # 策略自身视角
    lines.append("| **策略自身概率视角** | | | |")
    if "policy_accuracy" in base_metrics and "policy_accuracy" in tuned_metrics:
        add_row("偏好准确率", base_metrics["policy_accuracy"], tuned_metrics["policy_accuracy"], ".4f")
        add_row("平均 Chosen log_prob", base_metrics.get("policy_avg_chosen_logp"),
                tuned_metrics.get("policy_avg_chosen_logp"), ".4f")
        add_row("平均 Rejected log_prob", base_metrics.get("policy_avg_rejected_logp"),
                tuned_metrics.get("policy_avg_rejected_logp"), ".4f")
        add_row("平均概率间隔", base_metrics.get("policy_avg_margin"),
                tuned_metrics.get("policy_avg_margin"), ".4f")

    # 相对参考模型视角（只有微调后模型有）
    if "ref_accuracy" in tuned_metrics:
        lines.append("| **相对参考模型视角** | | | |")
        # 基座模型与参考模型同权重时该视角无判别信息（隐式奖励恒为 0）
        if base_metrics.get("is_self_reference") or "ref_accuracy" not in base_metrics:
            base_ref_acc = None
        else:
            base_ref_acc = base_metrics.get("ref_accuracy")
        add_row("偏好准确率（参考模型视角）", base_ref_acc, tuned_metrics["ref_accuracy"], ".4f")
        add_row("平均 Chosen 隐式奖励",
                base_metrics.get("ref_avg_chosen_reward") if not base_metrics.get("is_self_reference") else None,
                tuned_metrics.get("ref_avg_chosen_reward"), ".6f")
        add_row("平均 Rejected 隐式奖励",
                base_metrics.get("ref_avg_rejected_reward") if not base_metrics.get("is_self_reference") else None,
                tuned_metrics.get("ref_avg_rejected_reward"), ".6f")
        add_row("平均奖励间隔",
                base_metrics.get("ref_avg_margin") if not base_metrics.get("is_self_reference") else None,
                tuned_metrics.get("ref_avg_margin"), ".6f")
        if base_metrics.get("is_self_reference"):
            lines.append("")
            lines.append("> 注：基座模型与参考模型为同一份权重，其「相对参考模型视角」隐式奖励恒为 0，")
            lines.append("> 该视角无判别信息（上表显示 N/A），基座判别能力以「策略自身概率视角」为准。")

    # 分布指标
    lines.append("| **奖励间隔分布 (min/25%/50%/75%/max)** | | | |")
    if "ref_margin_median" in tuned_metrics:
        if "ref_margin_median" in base_metrics:
            b_p = (f"{base_metrics['ref_margin_min']:.4f}/{base_metrics['ref_margin_25p']:.4f}/"
                   f"{base_metrics['ref_margin_median']:.4f}/{base_metrics['ref_margin_75p']:.4f}/"
                   f"{base_metrics['ref_margin_max']:.4f}")
        else:
            b_p = "N/A"
        t_p = (f"{tuned_metrics['ref_margin_min']:.4f}/{tuned_metrics['ref_margin_25p']:.4f}/"
               f"{tuned_metrics['ref_margin_median']:.4f}/{tuned_metrics['ref_margin_75p']:.4f}/"
               f"{tuned_metrics['ref_margin_max']:.4f}")
        lines.append(f"| 奖励间隔分布 | {b_p} | {t_p} | — |")

    lines.append("\n---\n")
    lines.append("*对比报告由 evaluate_model.py 自动生成*\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"对比报告已保存: {output_path}")


# ============================================================
# 主函数
# ============================================================

def main():
    """
    评估主函数：分两步（基座模型、微调后模型）执行 DPO 评估。
    """
    # 加载配置
    config_path = "Config.yaml"
    if not os.path.exists(config_path):
        for candidate in ["./DPO/Config.yaml", "../Config.yaml"]:
            if os.path.exists(candidate):
                config_path = candidate
                break

    config = load_config(config_path)
    eval_cfg = config.get("EvalConfig", {})
    model_cfg = config.get("ModelConfig", {})
    dataset_cfg = config.get("DatasetConfig", {})
    output_cfg = config.get("OutputConfig", {})
    output_dir = output_cfg.get("output_dir", "./output")
    eval_dir = os.path.join(output_dir, output_cfg.get("eval_subdir", "eval"))
    os.makedirs(eval_dir, exist_ok=True)

    model_path = eval_cfg.get("model_name_or_path", model_cfg["model_name_or_path"])
    ref_model_path = eval_cfg.get("ref_model_name_or_path", model_path)
    test_data_path = eval_cfg.get("test_data_path", dataset_cfg.get("test_data_path"))
    beta = eval_cfg.get("beta", 0.1)
    batch_size = eval_cfg.get("eval_batch_size", 4)
    max_length = config.get("TrainConfig", {}).get("max_length", 1024)
    num_samples_show = eval_cfg.get("num_samples_to_show", 3)
    data_usage = eval_cfg.get("data_usage", 1.0)

    # 加载测试数据集
    print(f"加载测试数据集: {test_data_path}")
    dataset = load_dataset("json", data_files=test_data_path, split="train")
    if data_usage < 1.0:
        n_use = int(len(dataset) * data_usage)
        dataset = dataset.select(range(n_use))
    print(f"测试集样本数: {len(dataset)}")

    # 加载 tokenizer
    tokenizer = load_tokenizer(model_path)

    # 转换为 messages 列表
    data_list = list(dataset)  # [{"prompt": [...], "chosen": [...], "rejected": [...]}]

    # 先计算参考模型的对数概率（基座权重）
    torch_dtype = model_cfg.get("torch_dtype", "bfloat16")
    trust_rc = model_cfg.get("trust_remote_code", False)

    print("\n计算参考模型（基座）的对数概率...")
    ref_model = load_base_model(ref_model_path, torch_dtype, trust_rc)
    ref_metrics = evaluate_model(
        ref_model, tokenizer, data_list,
        beta=beta, batch_size=batch_size, max_length=max_length,
        ref_log_probs=None,  # 计算参考模型自身作为"策略"的概率
    )
    # 将参考模型自身的 logps 作为 ref_log_probs 源
    ref_log_probs_source = {
        "chosen": ref_metrics["_raw_data"]["chosen_logps"],
        "rejected": ref_metrics["_raw_data"]["rejected_logps"],
    }
    del ref_model
    torch.cuda.empty_cache()

    base_metrics = None
    tuned_metrics = None

    # ---- 第一步：评估基座模型 ----
    if eval_cfg.get("eval_base_model", True):
        print("\n" + "=" * 60)
        print("第一步：评估基座模型")
        print("=" * 60)
        base_model = load_base_model(model_path, torch_dtype, trust_rc)
        base_metrics = evaluate_model(
            base_model, tokenizer, data_list,
            beta=beta, batch_size=batch_size, max_length=max_length,
            ref_log_probs=ref_log_probs_source,
        )
        base_metrics["_beta"] = beta

        # 输出报告
        base_report_path = os.path.join(eval_dir, "base_model_report.md")
        write_evaluation_report(
            f"基座模型 ({model_path})", base_metrics, tokenizer,
            base_report_path, num_samples_show,
        )
        del base_model
        torch.cuda.empty_cache()
        print("基座模型评估完成\n")

    # ---- 第二步：评估微调后模型 ----
    if eval_cfg.get("eval_tuned_model", True):
        tuned_model_path = eval_cfg.get("tuned_model_path",
                                         os.path.join(output_dir, "final_model"))
        if os.path.exists(tuned_model_path):
            print("\n" + "=" * 60)
            print("第二步：评估微调后模型")
            print("=" * 60)
            tuned_model = load_tuned_model(model_path, tuned_model_path,
                                            torch_dtype, trust_rc)
            tuned_metrics = evaluate_model(
                tuned_model, tokenizer, data_list,
                beta=beta, batch_size=batch_size, max_length=max_length,
                ref_log_probs=ref_log_probs_source,
            )
            tuned_metrics["_beta"] = beta

            # 输出报告
            tuned_report_path = os.path.join(eval_dir, "tuned_model_report.md")
            write_evaluation_report(
                f"微调后模型 ({model_path} + LoRA)", tuned_metrics, tokenizer,
                tuned_report_path, num_samples_show,
            )
            del tuned_model
            torch.cuda.empty_cache()
            print("微调后模型评估完成\n")
        else:
            print(f"微调模型路径不存在: {tuned_model_path}，跳过微调模型评估")

    # ---- 对比报告 ----
    if base_metrics is not None and tuned_metrics is not None:
        comparison_path = os.path.join(eval_dir, "comparison_report.md")
        write_comparison_report(
            base_metrics, tuned_metrics, comparison_path,
            model_name="微调后模型", base_name="基座模型", beta=beta,
        )

    print("\n所有评估完成。")


if __name__ == "__main__":
    main()