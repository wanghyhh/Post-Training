"""
DPO 训练过程绘图脚本

从训练日志 jsonl 提取指标，生成多子图训练曲线图。
遵循 DPO/工作文档.md §3.6 的映射约定和子图布局。
"""

import os
import sys
import json
import math

import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_log_data(log_path: str) -> list:
    """加载训练日志 jsonl，返回字典列表。"""
    records = []
    if not os.path.exists(log_path):
        print(f"日志文件不存在: {log_path}")
        # 尝试子目录下的 train_log.jsonl
        alt_path = os.path.join("output", "logs", "train_log.jsonl")
        if os.path.exists(alt_path):
            log_path = alt_path
        else:
            print(f"尝试备用路径: {alt_path} 也不存在")
            return records
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"加载日志记录: {len(records)} 条")
    return records


def safe_extract(records, key, use_step=True):
    """
    从日志记录中提取指标序列。

    提取规则：
    - 训练指标（不带 eval_）：提取所有 step 对应的记录
    - eval 指标（带 eval_ 前缀）：仅提取包含该键的记录（来自 on_evaluate 的日志行）

    Args:
        records: 日志记录列表
        key: 要提取的指标键名
        use_step: 是否使用 step 作为 x 轴（默认 True）

    Returns:
        (x_values, y_values) 元组
    """
    x_vals = []
    y_vals = []
    for rec in records:
        val = rec.get(key)
        if val is not None and not (isinstance(val, float) and math.isnan(val)):
            if use_step:
                x_vals.append(rec.get("step", len(x_vals)))
            else:
                x_vals.append(len(x_vals))
            y_vals.append(val)
    return np.array(x_vals), np.array(y_vals)


def sliding_average(x, y, stride=1):
    """滑动窗口平均（stride>1 时启用平滑）。"""
    if stride <= 1 or len(y) == 0:
        return x, y
    smoothed = []
    smoothed_x = []
    for i in range(0, len(y) - stride + 1, stride):
        smoothed.append(np.mean(y[i:i + stride]))
        smoothed_x.append(np.mean(x[i:i + stride]))
    # 剩余尾部
    if len(y) % stride != 0:
        remainder = len(y) % stride
        smoothed.append(np.mean(y[-remainder:]))
        smoothed_x.append(np.mean(x[-remainder:]))
    return np.array(smoothed_x), np.array(smoothed)


def plot_subplot(ax, records, train_key, eval_key=None, title="",
                 xlabel="Step", ylabel="Value", stride=1,
                 start_step=0, is_step_time=False):
    """
    绘制单个子图。

    支持从第几步开始绘制，支持平滑双层曲线。

    Args:
        ax: matplotlib Axes 对象
        records: 日志记录列表
        train_key: 训练指标的键名（如 "loss"）
        eval_key: eval 指标的键名（如 "eval_loss"）
        title: 子图标题
        xlabel: X 轴标签
        ylabel: Y 轴标签
        stride: 采样步长（>1 时启用滑动窗口平滑）
        start_step: 跳过前 N 步
        is_step_time: 是否为 step_time（累计耗时模式）
    """
    if is_step_time:
        # step_time 特殊处理：累加为分钟
        x_raw, y_raw = safe_extract(records, "step_time")
        if len(y_raw) == 0:
            ax.text(0.5, 0.5, "step_time 缺失", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10, color="gray")
            ax.set_title(title)
            return
        # 只画累计耗时
        y_cum = np.cumsum(y_raw) / 60.0  # 秒→分钟
        mask = x_raw >= start_step
        if mask.any():
            ax.plot(x_raw[mask], y_cum[mask], color="tab:purple",
                    linewidth=1.5, label="Cumulative Time")
            total_time = y_cum[-1]
            total_steps = x_raw[-1]
            ax.annotate(f"Total: {total_time:.1f} min\nSteps: {int(total_steps)}",
                        xy=(0.95, 0.95), xycoords="axes fraction",
                        ha="right", va="top", fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor="yellow", alpha=0.3))
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper left", fontsize=8)
        return

    # 训练指标
    x_train, y_train = safe_extract(records, train_key)
    if len(x_train) > 0:
        mask = x_train >= start_step
        x_train, y_train = x_train[mask], y_train[mask]
        if stride > 1:
            # 原始曲线（底）
            ax.plot(x_train, y_train, alpha=0.2, linewidth=1,
                    color="tab:blue", label=f"{train_key} (raw)")
            # 平滑曲线（顶）
            x_smooth, y_smooth = sliding_average(x_train, y_train, stride)
            ax.plot(x_smooth, y_smooth, alpha=0.8, linewidth=1.5,
                    color="tab:blue", label=train_key)
        else:
            ax.plot(x_train, y_train, alpha=0.8, linewidth=1.5,
                    color="tab:blue", label=train_key)

    # Eval 指标（散点/虚线叠加）
    if eval_key:
        x_eval, y_eval = safe_extract(records, eval_key)
        if len(x_eval) > 0:
            mask = x_eval >= start_step
            x_eval, y_eval = x_eval[mask], y_eval[mask]
            ax.scatter(x_eval, y_eval, color="tab:orange", s=20, zorder=5,
                       marker="o", label=eval_key)
            if len(x_eval) > 1:
                ax.plot(x_eval, y_eval, color="tab:orange", linewidth=1,
                        linestyle="--", alpha=0.7)

    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)

    # 只有在至少有一条曲线时才调用 legend
    has_data = (len(x_train) > 0) or (eval_key and any(
        rec.get(eval_key) is not None for rec in records
    ))
    if has_data:
        try:
            ax.legend(loc="upper left", fontsize=7)
        except Exception:
            pass


def main():
    # 加载配置
    config_path = "Config.yaml"
    if not os.path.exists(config_path):
        for candidate in ["./DPO/Config.yaml", "../Config.yaml"]:
            if os.path.exists(candidate):
                config_path = candidate
                break

    config = load_config(config_path)
    plot_cfg = config.get("PlotConfig", {})
    output_cfg = config.get("OutputConfig", {})

    output_dir = output_cfg.get("output_dir", "./output")
    plots_dir = os.path.join(output_dir, output_cfg.get("plots_subdir", "plots"))
    os.makedirs(plots_dir, exist_ok=True)

    # 绘图参数
    subplot_cols = plot_cfg.get("subplot_cols", 2)
    hspace = plot_cfg.get("hspace", 0.35)
    wspace = plot_cfg.get("wspace", 0.3)
    title_y = plot_cfg.get("title_y", 1.02)
    start_step = plot_cfg.get("start_step", 0)
    stride = plot_cfg.get("stride", 1)
    output_filename = plot_cfg.get("output_filename", "train_curves.png")
    output_dpi = plot_cfg.get("output_dpi", 150)

    # 加载日志
    log_dir = os.path.join(output_dir, output_cfg.get("logs_subdir", "logs"))
    log_path = os.path.join(log_dir, "train_log.jsonl")
    records = load_log_data(log_path)

    if not records:
        print("无日志数据，无法绘图。")
        return

    print(f"绘图参数: subplot_cols={subplot_cols}, stride={stride}, "
          f"start_step={start_step}")

    # 子图规划（遵循文档 §3.6 约定）
    plot_specs = []
    # ① loss
    plot_specs.append(("loss", "eval_loss", "Loss", "Loss", False))
    # ② 奖励曲线
    plot_specs.append(("rewards/chosen", "eval_rewards/chosen",
                       "Reward (Chosen)", "Reward", False))
    plot_specs.append(("rewards/rejected", "eval_rewards/rejected",
                       "Reward (Rejected)", "Reward", False))
    plot_specs.append(("rewards/margins", "eval_rewards/margins",
                       "Reward Margin", "Margin", False))
    # ③ 偏好准确率
    plot_specs.append(("rewards/accuracies", "eval_rewards/accuracies",
                       "Preference Accuracy", "Accuracy", False))
    # ④ 对数概率
    plot_specs.append(("logps/chosen", "eval_logps/chosen",
                       "Log Prob (Chosen)", "log p", False))
    plot_specs.append(("logps/rejected", "eval_logps/rejected",
                       "Log Prob (Rejected)", "log p", False))
    # ⑤ nll_loss
    plot_specs.append(("nll_loss", "eval_nll_loss", "NLL Loss", "NLL", False))
    # ⑥ 梯度与学习率（双 y 轴）
    plot_specs.append(("grad_norm", None, "Grad Norm & LR", "Value", False))
    # ⑦ 累计耗时
    plot_specs.append((None, None, "Training Time", "Cumulative (min)", True))

    # 过滤空指标
    active_specs = []
    for spec in plot_specs:
        train_k, eval_k, title, ylabel, is_time = spec
        if is_time:
            # step_time：检查是否有数据
            has_step_time = any(rec.get("step_time") is not None for rec in records)
            if has_step_time:
                active_specs.append(spec)
            else:
                print(f"跳过子图 [{title}]: step_time 缺失")
            continue
        has_train = any(rec.get(train_k) is not None for rec in records) if train_k else False
        has_eval = any(rec.get(eval_k) is not None for rec in records) if eval_k else False
        if not has_train and not has_eval:
            print(f"跳过子图 [{title}]: 数据全空 ({train_k}/{eval_k})")
            continue
        if train_k == "nll_loss" and not has_train and not has_eval:
            print(f"跳过子图 [NLL Loss]: nll_loss 缺失（未启用 rpo_alpha）")
            continue
        if train_k == "learning_rate":
            # 与 grad_norm 合并到同一子图（双 y 轴），所以检查 grad_norm
            has_grad = any(rec.get("grad_norm") is not None for rec in records)
            if not has_grad and not has_train:
                print(f"跳过子图 [Grad Norm & LR]: 数据全空")
                continue
        active_specs.append(spec)

    print(f"将绘制 {len(active_specs)} 个子图")

    # 计算子图行列
    total_plots = len(active_specs)
    n_cols = subplot_cols
    n_rows = math.ceil(total_plots / n_cols)

    # 创建画布
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten()

    fig.suptitle("DPO Training Curves", fontsize=14, y=title_y + 0.02)
    fig.subplots_adjust(hspace=hspace, wspace=wspace)

    # 按序绘制每个活动子图
    for idx, spec in enumerate(active_specs):
        train_k, eval_k, title, ylabel, is_time = spec
        ax = axes_flat[idx]

        if is_time:
            # 累计耗时子图
            x_raw, y_raw = safe_extract(records, "step_time")
            if len(y_raw) > 0:
                y_cum = np.cumsum(y_raw) / 60.0  # 秒→分钟
                mask = x_raw >= start_step
                ax.plot(x_raw[mask], y_cum[mask], color="tab:purple",
                        linewidth=1.5, label="Cumulative Time (min)")
                total_time = y_cum[-1]
                total_steps = x_raw[-1]
                ax.annotate(
                    f"Total: {total_time:.1f} min\nSteps: {int(total_steps)}",
                    xy=(0.95, 0.95), xycoords="axes fraction",
                    ha="right", va="top", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3",
                            facecolor="lightyellow", alpha=0.8),
                )
            else:
                ax.text(0.5, 0.5, "step_time missing",
                        transform=ax.transAxes, ha="center", va="center",
                        color="gray")
        elif train_k == "grad_norm" and eval_k is None:
            # 在 grad_norm 子图中叠加 learning_rate（双 y 轴）
            # grad_norm 左轴
            x_gn, y_gn = safe_extract(records, "grad_norm")
            if len(x_gn) > 0:
                mask = x_gn >= start_step
                x_gn, y_gn = x_gn[mask], y_gn[mask]
                if stride > 1:
                    ax.plot(x_gn, y_gn, alpha=0.2, linewidth=1,
                            color="tab:blue", label="grad_norm (raw)")
                    x_sm, y_sm = sliding_average(x_gn, y_gn, stride)
                    ax.plot(x_sm, y_sm, alpha=0.8, linewidth=1.5,
                            color="tab:blue", label="grad_norm")
                else:
                    ax.plot(x_gn, y_gn, alpha=0.8, linewidth=1.5,
                            color="tab:blue", label="grad_norm")

            # learning_rate 右轴
            ax2 = ax.twinx()
            x_lr, y_lr = safe_extract(records, "learning_rate")
            if len(x_lr) > 0:
                mask = x_lr >= start_step
                x_lr, y_lr = x_lr[mask], y_lr[mask]
                ax2.plot(x_lr, y_lr, alpha=0.7, linewidth=1.2,
                         color="tab:red", label="learning_rate", linestyle="--")
                ax2.set_ylabel("Learning Rate", fontsize=9, color="tab:red")
                ax2.tick_params(axis="y", labelsize=8, colors="tab:red")
                ax2.legend(loc="upper right", fontsize=7)

            ax.set_ylabel("Grad Norm", fontsize=9, color="tab:blue")
            ax.tick_params(axis="y", labelsize=8, colors="tab:blue")
        else:
            # 常规子图
            plot_subplot(ax, records, train_k, eval_k, title=title,
                         xlabel="Step", ylabel=ylabel, stride=stride,
                         start_step=start_step)

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Step", fontsize=9)

    # 隐藏多余的子图
    for idx in range(len(active_specs), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # 保存输出
    output_path = os.path.join(plots_dir, output_filename)
    plt.savefig(output_path, dpi=output_dpi, bbox_inches="tight")
    plt.close()
    print(f"训练曲线图已保存: {output_path}")


if __name__ == "__main__":
    main()