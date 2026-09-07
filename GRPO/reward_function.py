# ====================================================
# GRPO 奖励函数模块
# 用途：实现 GRPO 训练和多阶段评估的奖励计算逻辑
# 说明：通过 reward_funcs 参数传入 GRPOTrainer
# ====================================================

import re
import torch
from typing import List, Dict, Optional


class GRPORewardFunction:
    """
    GRPO 奖励函数类。

    设计思想：
        - 支持多个奖励分量（长度、匹配等），每个分量独立计算后通过权重加权组合。
        - 各分量归一化到 [0, 1] 范围。
        - 支持参考回答（references）用于计算分量奖励。
        - 内部维护 rewards 字典，便于训练日志回调和评估脚本提取。
        - 支持批量输入，对异常样本降级为 0 奖励。

    示例：
        >>> reward_fn = GRPORewardFunction(weights={"length": 0.5, "match": 0.5})
        >>> rewards = reward_fn(prompts=["问题"], completions=["答案"], references=["参考答案"])
        >>> print(rewards.shape)  # torch.Size([1])
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        answer_tag: str = "",
        forbidden_strings: Optional[List[str]] = None,
    ):
        """
        初始化奖励函数配置参数。

        Args:
            weights: 各奖励分量的权重字典。例如 {"length": 0.5, "match": 0.5}。
                     默认 {"length": 0.5, "match": 0.5}。
            answer_tag: 答案提取标签（从生成文本中按此标签提取实际答案）。
                        默认为空字符串（不使用标签过滤）。
            forbidden_strings: 违禁字符串列表。如果生成文本命中任何违禁串，该样本
                               的总奖励将被强制设为 0。默认为空列表。
        """
        self.__name__ = "grpo_reward_function"
        self.answer_tag = answer_tag
        self.forbidden_strings = forbidden_strings or []

        # 权重分配
        self.weights = weights or {
            "length": 0.5,
            "match": 0.5,
        }

        # 历史奖励缓存（供评估报告使用）
        self.rewards: Dict[str, Optional[torch.Tensor]] = {
            "total_rewards": None,
            "length_rewards": None,
            "match_rewards": None,
        }

        # 原始输入缓存
        self.prompts: Optional[List[str]] = None
        self.completions: Optional[List[str]] = None
        self.references: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------

    def _contains_forbidden_string(self, text: str) -> bool:
        """
        检查文本中是否包含任何违禁字符串。

        Args:
            text: 待检查的生成文本。

        Returns:
            如果包含违禁串返回 True，否则返回 False。
        """
        for s in self.forbidden_strings:
            if s in text:
                return True
        return False

    def _extract_answer(self, completion: str) -> str:
        """
        从生成文本中提取实际答案。

        如果配置了 answer_tag，则提取该标签之后的内容；
        否则返回整个生成文本（去除首尾空白）。

        Args:
            completion: 模型的完整生成文本。

        Returns:
            提取后的答案字符串。
        """
        if self.answer_tag and self.answer_tag in completion:
            idx = completion.index(self.answer_tag) + len(self.answer_tag)
            return completion[idx:].strip()
        return completion.strip()

    def _calculate_length_reward(self, gen_len: int, ref_len: int) -> float:
        """
        计算长度奖励分量。

        公式：reward = 1 - |gen_len - ref_len| / (gen_len + ref_len)
        值越接近 1 表示生成内容与参考答案长度越接近。

        Args:
            gen_len: 生成文本的字符长度。
            ref_len: 参考答案的字符长度。

        Returns:
            归一化到 [0, 1] 的长度奖励值。
        """
        total = gen_len + ref_len
        if total == 0:
            return 1.0
        return 1.0 - abs(gen_len - ref_len) / total

    def _calculate_match_reward(self, gen_text: str, ref_text: str) -> float:
        """
        计算文字匹配奖励分量。

        规则：若生成文本去除空白后的首字符与参考答案首字符相同，奖励为 1.0，
        否则为 0.0。

        Args:
            gen_text: 生成文本（去除首尾空白后）。
            ref_text: 参考答案（去除首尾空白后）。

        Returns:
            匹配奖励值（0.0 或 1.0）。
        """
        gen_stripped = gen_text.strip()
        ref_stripped = ref_text.strip()
        if not gen_stripped or not ref_stripped:
            return 0.0
        return 1.0 if gen_stripped[0] == ref_stripped[0] else 0.0

    # ------------------------------------------------------------------
    # 公开调用入口
    # ------------------------------------------------------------------

    def __call__(
        self,
        prompts: List[str],
        completions: List[str],
        references: Optional[List[str]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        GRPO Trainer 调用入口。

        对批量输入的每个样本逐个计算奖励分量，加权组合后返回奖励张量。

        Args:
            prompts: prompt 文本列表，长度为 batch_size。
            completions: 模型生成的文本列表，长度与 prompts 相同。
            references: 参考回答列表（可选），长度与 prompts 相同。
                        如果不提供，则跳过依赖 reference 的分量计算。

        Returns:
            奖励张量，shape 为 [batch_size]，dtype 为 float32。
        """
        self.prompts = prompts.copy()
        self.completions = completions.copy()
        self.references = list(references) if references else ["" for _ in prompts]

        batch_size = len(prompts)

        length_rewards = []
        match_rewards = []

        for i in range(batch_size):
            completion = completions[i]
            reference = self.references[i] if i < len(self.references) else ""
            answer = self._extract_answer(completion)

            # 违禁词检查：命中后该样本所有分量为 0
            if self._contains_forbidden_string(completion):
                length_rewards.append(0.0)
                match_rewards.append(0.0)
                continue

            # 长度奖励
            gen_len = len(answer)
            ref_len = len(reference) if reference else gen_len
            length_rewards.append(self._calculate_length_reward(gen_len, ref_len))

            # 文字匹配奖励
            match_rewards.append(self._calculate_match_reward(answer, reference))

        # 转换为 Tensor
        length_tensor = torch.tensor(length_rewards, dtype=torch.float32)
        match_tensor = torch.tensor(match_rewards, dtype=torch.float32)

        # 加权组合
        w_length = self.weights.get("length", 0.5)
        w_match = self.weights.get("match", 0.5)
        total_rewards = w_length * length_tensor + w_match * match_tensor

        # 保存缓存
        self.rewards = {
            "total_rewards": total_rewards.detach().cpu(),
            "length_rewards": length_tensor.detach().cpu(),
            "match_rewards": match_tensor.detach().cpu(),
        }

        return total_rewards
