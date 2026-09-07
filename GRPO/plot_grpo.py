# ====================================================
# GRPO 训练曲线绘图脚本
# 用途：从训练日志（JSONL）绘制训练/评估指标曲线。
# 用法：
#   python plot_grpo.py                          # 使用默认配置
#   python plot_grpo.py --log-file PATH          # 指定日志文件
#   python plot_grpo.py --config Config.yaml     # 指定配置文件
#   python plot_grpo.py --stride 20 --ncols 4    # 自定义平滑步长和列数
# ====================================================

import os
import sys
import json
import argparse
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional

# Windows GBK 编码修复
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ============================================================
# 依赖导入区
# ★ Windows conda 环境 0xC0000005 crash 防护
# ============================================================
import pandas  # noqa: F401

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 非交互式后端（无 GUI 环境）
import matplotlib.pyplot as plt
import yaml

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)


# ============================================================
# 日志读取
# ============================================================

def _read_logs(file_path: str) -> List[Dict[str, Any]]:
    """
    读取 JSONL 格式的日志文件。

    支持两种格式：
    - JSONL：每行一个 JSON 对象
    - JSON：单个 JSON 对象，含 "logs" 数组

    Args:
        file_path: 日志文件路径

    Returns:
        日志行列表，每个元素为字典
    """
    logs = []

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 尝试解析为完整 JSON
    try:
        data = json.loads(content)
        if isinstance(data, dict) and "logs" in data:
            print(f"✓ 成功读取完整 JSON 格式，共 {len(data['logs'])} 条日志")
            return data["logs"]
    except json.JSONDecodeError:
        pass

    # 按行解析 JSONL
    lines = content.strip().split("\n")
    i_step = 0
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            line_data = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"警告：第 {i + 1} 行解析失败: {e}")
            continue

        # 跳过同时包含 loss 和 eval_loss 的行（混合行）
        if line_data.get("loss") is not None and line_data.get("eval_loss") is not None:
            continue

        if line_data.get("loss") is not None:
            i_step += 1

        if not line_data.get("global_step"):
            line_data["global_step"] = i_step

        logs.append(line_data)

    print(f"✓ 成功读取 JSONL 格式，共 {len(logs)} 条日志")
    return logs


# ============================================================
# 绘图函数
# ============================================================

def plot_training_logs(
    jsonl_path: str,
    output_dir: str = "./output/plots",
    n_cols: int = 3,
    hspace: float = 0.4,
    wspace: float = 0.3,
    y_title: float = 0.96,
    start_index: int = 0,
    stride: int = 10,
    title: str = "",
) -> str:
    """
    从训练日志绘制训练/评估指标曲线。

    Args:
        jsonl_path: 日志 JSONL 文件路径
        output_dir: 输出目录
        n_cols: 子图列数
        hspace: 子图垂直间距
        wspace: 子图水平间距
        y_title: suptitle 的 y 位置
        start_index: 从第几步开始绘制
        stride: 采样步长（滑动窗口大小），> 1 时平滑降采样
        title: 图表标题（留空则自动生成）

    Returns:
        输出文件路径
    """
    output_path_obj = Path(output_dir)
    output_path_obj.mkdir(parents=True, exist_ok=True)

    logs = _read_logs(jsonl_path)

    if not logs:
        print("✗ 未读取到任何日志数据")
        return ""

    # 处理 start_index
    if start_index < 0:
        start_index = 0
    if start_index >= len(logs):
        start_index = 0

    logs = logs[start_index:]

    # 参数映射：训练指标名 → 评估指标名（None 表示无对应评估指标）
    param_mappings = {
        "loss": "eval_loss",
        "grad_norm": None,
        "learning_rate": None,
        "completions/min_length": "eval_completions/min_length",
        "completions/mean_length": "eval_completions/mean_length",
        "completions/max_length": "eval_completions/max_length",
        "completions/clipped_ratio": "eval_completions/clipped_ratio",
        "reward": "eval_reward",
        "reward_std": "eval_reward_std",
        "mean_error": "eval_mean_error",
        "zero_ratio": "eval_zero_ratio",
        "kl": "eval_kl",
        "entropy": "eval_entropy",
        "step_time": None,
    }

    # 按参数组织数据
    data = {
        key: {"step": [], "epoch": [], "train": [], "eval": []}
        for key in param_mappings.keys()
    }

    for log in logs:
        step = log.get("step")
        epoch = log.get("epoch")
        if step is None and epoch is None:
            continue

        for train_key, eval_key in param_mappings.items():
            train_val = log.get(train_key)
            eval_val = log.get(eval_key) if eval_key else None

            if train_val is not None or eval_val is not None:
                if step is not None:
                    data[train_key]["step"].append(step)
                if epoch is not None:
                    data[train_key]["epoch"].append(epoch)
                data[train_key]["train"].append(train_val)
                data[train_key]["eval"].append(eval_val)

    # 计算子图布局
    n_params = len(param_mappings)
    n_rows = (n_params + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 5, n_rows * 4),
        squeeze=False,
    )
    fig_title = title or f"GRPO Training Logs: {Path(jsonl_path).stem}"
    fig.suptitle(fig_title, fontsize=16, fontweight="bold", y=y_title)

    for idx, (train_key, eval_key) in enumerate(param_mappings.items()):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        x_vals = data[train_key]["epoch"]
        train_vals = data[train_key]["train"]
        eval_vals = data[train_key]["eval"]

        # step_time 特殊处理：累计曲线
        if train_key == "step_time":
            cumulative_times = []
            steps = []
            cumsum = 0.0
            for log in logs:
                st = log.get("step_time")
                if st is not None:
                    st_min = st / 60.0  # 转为分钟
                    cumsum += st_min
                    cumulative_times.append(cumsum)
                    steps.append(log.get("global_step"))

            train_epochs = steps
            train_clean = cumulative_times
            display_train_key = "cost_time / minutes"
            ax.set_xlabel("global_step", fontsize=10)

            # 标注最后一个点
            if train_epochs:
                last_x, last_y = train_epochs[-1], train_clean[-1]
                ax.plot(last_x, last_y, "ro", markersize=8)
                ax.annotate(
                    f"{last_x} steps\n{last_y:.1f} min",
                    xy=(last_x, last_y),
                    xytext=(-50, -60),
                    textcoords="offset points",
                    arrowprops=dict(arrowstyle="->", color="red", lw=1),
                    fontsize=10,
                    color="red",
                )
        else:
            # 正常指标：过滤 None 值
            train_epochs = [x for x, v in zip(x_vals, train_vals) if v is not None]
            train_clean = [v for v in train_vals if v is not None]
            display_train_key = train_key
            ax.set_xlabel("Epoch", fontsize=10)

        eval_epochs = [x for x, v in zip(x_vals, eval_vals) if v is not None]
        eval_clean = [v for v in eval_vals if v is not None]

        # 绘制 Train 曲线
        if train_clean:
            if stride > 1:
                # 原始曲线（极浅色）
                ax.plot(
                    train_epochs, train_clean,
                    "b-", linewidth=1, label="Train (Raw)", alpha=0.2,
                )
                # 滑动窗口平滑
                arr = np.array(train_clean)
                window_size = stride
                smoothed_vals = []
                smoothed_epochs = []

                for i in range(0, len(arr), window_size):
                    window = arr[i : i + window_size]
                    if len(window) > 0:
                        smoothed_vals.append(np.mean(window))
                        smoothed_epochs.append(train_epochs[i])

                ax.plot(
                    smoothed_epochs, smoothed_vals,
                    "b-", linewidth=1.5, label=f"Train (Stride={stride})", alpha=0.8,
                )
            else:
                ax.plot(
                    train_epochs, train_clean,
                    "b-", linewidth=1.5, label="Train", alpha=0.8,
                )

        # 绘制 Eval 曲线
        if eval_clean:
            ax.plot(
                eval_epochs, eval_clean,
                "r--.", linewidth=1.5, label="Eval", alpha=0.8,
            )

        ax.set_title(display_train_key, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    # 隐藏多余子图
    for idx in range(n_params, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        fig.delaxes(axes[row, col])

    plt.subplots_adjust(hspace=hspace, wspace=wspace)

    # 生成输出文件名
    stem = Path(jsonl_path).stem
    output_path = output_path_obj / f"training_curves_{stem}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")

    print(f"✓ 图表已保存至: {output_path}")
    print(f"  采样步长 (stride): {stride}")
    print(f"  布局: {n_rows} 行 × {n_cols} 列")
    plt.close(fig)

    return str(output_path)


# ============================================================
# 命令行入口
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="GRPO 训练曲线绘图脚本")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(_SCRIPT_DIR, "Config.yaml"),
        help="Config.yaml 路径（用于读取 PlotConfig）",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="训练日志 JSONL 路径（默认自动查找 output/logs/train_log.jsonl 或最新时间戳文件）",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="绘图输出目录（默认从 OutputConfig.plot_dir 读取）",
    )
    parser.add_argument(
        "--n-cols",
        type=int,
        default=None,
        help="子图列数（默认从 Config.yaml 读取）",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="平滑降采样步长（默认从 Config.yaml 读取）",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="从第几步开始绘制（默认 0）",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="",
        help="图表标题",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("  GRPO 训练曲线绘图")
    print("=" * 70)

    # -------------------------------------------
    # 加载 PlotConfig 和 OutputConfig
    # -------------------------------------------
    plot_cfg = {}
    output_cfg = {}
    if os.path.exists(args.config):
        try:
            with open(args.config, "r", encoding="utf-8") as f:
                all_cfg = yaml.safe_load(f)
            plot_cfg = all_cfg.get("PlotConfig", {})
            output_cfg = all_cfg.get("OutputConfig", {})
        except Exception as e:
            print(f"警告: 读取配置失败: {e}")

    # -------------------------------------------
    # 确定日志文件路径
    # -------------------------------------------
    log_file = args.log_file
    if not log_file:
        # 从 OutputConfig 获取日志目录，或回退到默认路径
        log_dir = output_cfg.get("train_log_dir") or args.output_path or os.path.join(_SCRIPT_DIR, "output", "logs")
        unified_log = os.path.join(log_dir, "train_log.jsonl")

        if os.path.exists(unified_log):
            log_file = unified_log
        else:
            # 查找最新时间戳文件
            timestamp_files = sorted(Path(log_dir).glob("train_log_*.jsonl"))
            if timestamp_files:
                log_file = str(timestamp_files[-1])
            else:
                print(f"✗ 未找到训练日志文件")
                print(f"  请检查 output/logs/ 目录或 --log-file 参数")
                return

    print(f"日志文件: {log_file}")

    # -------------------------------------------
    # 确定绘图参数
    # -------------------------------------------
    n_cols = args.n_cols or plot_cfg.get("ncols", 3)
    stride = args.stride or plot_cfg.get("stride", 10)
    hspace = plot_cfg.get("hspace", 0.4)
    wspace = plot_cfg.get("wspace", 0.3)
    start_index = args.start_index or plot_cfg.get("start_index", 0)
    output_path = args.output_path or output_cfg.get("plot_dir") or plot_cfg.get("output_dir") or os.path.join(_SCRIPT_DIR, "output", "plots")

    print(f"  列数: {n_cols}")
    print(f"  平滑步长: {stride}")
    print(f"  起始步: {start_index}")

    # -------------------------------------------
    # 绘图
    # -------------------------------------------
    try:
        output_file = plot_training_logs(
            jsonl_path=log_file,
            output_dir=output_path,
            n_cols=n_cols,
            hspace=hspace,
            wspace=wspace,
            start_index=start_index,
            stride=stride,
            title=args.title,
        )
        if output_path:
            print(f"\n✓ 绘图完成: {output_path}")
        else:
            print(f"\n✗ 绘图失败")
    except Exception as e:
        traceback.print_exc()
        print(f"\n✗ 绘图异常: {e}")


if __name__ == "__main__":
    main()
