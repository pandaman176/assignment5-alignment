import pathlib
import socket
from typing import Dict, List, Callable, Literal

import torch
from transformers import PreTrainedTokenizerBase, AutoTokenizer, PreTrainedModel
from torch.nn.utils.rnn import pad_sequence
from einops import rearrange

hostname = socket.gethostname()
is_local: bool = "MacBook" in hostname

MODEL_PATH = pathlib.Path(__file__).parent.parent / "model" if is_local else pathlib.Path("/root/autodl-fs")
PROMPT_PATH = pathlib.Path(__file__).parent / "prompts"
LOG_PATH = pathlib.Path(__file__).parent.parent / "logs" if is_local else pathlib.Path("/root/autodl-fs")
DATA_PATH = pathlib.Path(__file__).parent.parent / "data"


def tokenize_prompt_and_output(
    prompt_strs: List[str],
    output_strs: List[str],
    tokenizer: PreTrainedTokenizerBase,
) -> Dict[str, torch.Tensor]:
    """将提示语与输出分别分词、拼接并构造回复掩码。

    功能概述：
        - 分别对 `prompt_strs` 与 `output_strs` 进行分词；
        - 将每个样本的提示与输出的 token 序列首尾相接；
        - 依据输出部分的位置，生成 `response_mask`（输出 token 位置为 1，其余为 0）；
        - 形成语言建模所需的 `input_ids` 与 `labels`（移位后的输入）。

    参数：
        prompt_strs: List[str]
            提示语字符串列表。
        output_strs: List[str]
            输出字符串列表。
        tokenizer: PreTrainedTokenizerBase
            用于分词的 tokenizer。

    返回：
        Dict[str, torch.Tensor]
            - "input_ids": 形状为 (batch_size, max_len - 1) 的张量，
              表示拼接后的 token 序列去掉最后一个 token。
            - "labels": 形状为 (batch_size, max_len - 1) 的张量，
              为 `input_ids` 向右移位一位后的标签。
            - "response_mask": 形状为 (batch_size, max_len - 1) 的张量，
              在标签对应的输出 token 位置为 1，其余位置为 0（提示或补齐）。
    """
    # 对提示和输出分别进行编码
    input_ids = []
    response_mask = []
    for prompt, output in zip(prompt_strs, output_strs):
        prompt_ids = tokenizer.encode(prompt)
        output_ids = tokenizer.encode(output)
        input_ids.append(torch.tensor(prompt_ids + output_ids))
        response_mask.append(torch.tensor([False] * len(prompt_ids) + [True] * len(output_ids)))

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id, padding_side="right")
    response_mask = pad_sequence(response_mask, batch_first=True, padding_value=0, padding_side="right")


    # 创建labels（input_ids向左移一位）
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    response_mask[:, :-1] = response_mask[:, 1:] # 和label对齐


    # 调整长度以匹配
    max_len = input_ids.shape[1] - 1
    input_ids = input_ids[:, :max_len]
    labels = labels[:, :max_len]
    response_mask = response_mask[:, :max_len]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask
    }


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """计算下一 token 预测在词表维度上的逐位置熵。

    功能：
        给定未归一化的 logits（形状为 (batch_size, sequence_length, vocab_size)），
        计算每个时间步的预测分布熵，并返回形状为
        (batch_size, sequence_length) 的张量。

    参数：
        logits: torch.Tensor
            未归一化 logits，形状为 (batch_size, sequence_length, vocab_size)。

    返回：
        torch.Tensor
            形状为 (batch_size, sequence_length) 的逐位置熵。
    """
    
    softmax_logits = torch.softmax(logits, dim=-1)
    entropy = -torch.sum(softmax_logits * (logits - torch.logsumexp(logits, dim=-1, keepdim=True)), dim=-1)
    return entropy


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> Dict[str, torch.Tensor]:
    """获取因果语言模型在给定前缀下的逐 token 条件对数概率，并可选返回下一 token 分布的熵。

    功能概述：
        - 使用放置在正确设备上的 HuggingFace `PreTrainedModel` 进行打分；
        - 对输入 `input_ids` 计算逐位置的条件对数概率 log p(x_t | x_{<t})；
        - 使用 `labels` 指定目标 token（通常为 `input_ids` 右移一位）；
        - 当 `return_token_entropy=True` 时，额外返回每个位置的下一 token 分布熵。

    参数：
        model: PreTrainedModel
            用于打分的 HuggingFace 模型，对应评测应在推理模式/不计算梯度下使用。
        input_ids: torch.Tensor
            形状为 (batch_size, sequence_length) 的整型张量，拼接后的 prompt+response token 序列。
        labels: torch.Tensor
            形状为 (batch_size, sequence_length) 的整型张量，由分词与右移得到的标签序列。
        return_token_entropy: bool
            若为 True，则计算并返回逐 token 的熵。

    返回：
        Dict[str, torch.Tensor]
            - "log_probs": 形状为 (batch_size, sequence_length) 的张量，表示条件对数概率；
            - "token_entropy": 可选，形状为 (batch_size, sequence_length) 的张量，仅当
              `return_token_entropy=True` 时存在。

    备注：
        实现提示：可通过 `model(input_ids).logits` 获得 logits，随后配合 `labels` 计算。
    """
    logits = model(input_ids).logits # (batch_size, sequence_length, vocab_size)
    label_logits = logits.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1) # (batch_size, sequence_length)
    log_probs = label_logits - torch.logsumexp(logits, dim=-1)
    if return_token_entropy:
        entropy = compute_entropy(logits)
        return {
            "log_probs": log_probs,
            "token_entropy": entropy
        }
    else:
        return {
            "log_probs": log_probs
        }


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """按掩码在给定维度求和并按常数归一化。

    功能：
        仅对`mask == 1`（或 True）的位置累加`tensor`的值；
        在维度`dim`上进行求和（若`dim is None`则对全部维度求和），
        然后将求和结果除以`normalize_constant`得到归一化结果。

    参数：
        tensor: torch.Tensor
            要进行掩码求和与归一化的张量。
        mask: torch.Tensor
            与`tensor`同形状的布尔/0-1张量；值为1/True的位置参与求和。
        normalize_constant: float
            归一化常数，最终结果将被该常数除以。
        dim: int | None
            求和维度；若为 None，则对所有维度求和。

    返回：
        torch.Tensor: 掩码求和后再除以`normalize_constant`得到的归一化张量。
    """
    if dim is None:
        dim = tuple(range(tensor.ndim))
    return tensor.masked_fill(~mask, 0).sum(dim=dim) / normalize_constant


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """计算朴素策略梯度（每 token）损失的占位函数。

    功能：
        在强化学习微调中，给定每个样本的标量回报或已归一化优势
        `raw_rewards_or_advantages`（形状为 (batch_size, 1)），以及策略在每个
        token 位置的对数概率 `policy_log_probs`（形状为 (batch_size, sequence_length)），
        该函数需返回逐 token 的策略梯度损失张量（形状为 (batch_size, sequence_length)）。

    参数：
        raw_rewards_or_advantages: torch.Tensor
            形状为 (batch_size, 1) 的张量，代表每条响应的原始回报或已计算好的优势。
        policy_log_probs: torch.Tensor
            形状为 (batch_size, sequence_length) 的张量，代表策略在各 token 上的对数概率。

    返回：
        tuple[torch.Tensor, dict[str, torch.Tensor]]
            - 第一个元素：形状为 (batch_size, sequence_length) 的逐 token 策略梯度损失。
            - 第二个元素：包含统计信息的字典（如无可返回空字典）。

    实现提示：
        - 需要将 (batch_size, 1) 的回报/优势在 sequence_length 维上进行广播；
        - 损失的符号约定通常为 `- advantage * log_prob` 的逐位置形式。
    """
    loss = -raw_rewards_or_advantages * policy_log_probs
    return loss, {
        "raw_rewards_or_advantages": raw_rewards_or_advantages,
        "policy_log_probs": policy_log_probs,
    }


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """计算 GRPO-Clip 的逐 token 损失（占位实现）。

    功能：
        - 给定每个样本的优势（形状为 (batch_size, 1)），以及新旧策略在每个 token 上的对数概率
          `policy_log_probs` 与 `old_log_probs`（形状均为 (batch_size, sequence_length)），
          按照 GRPO-Clip 公式计算逐 token 的损失；
        - 返回逐 token 的损失张量与用于日志记录的元数据（例如是否发生裁剪等）。

    参数：
        advantages: torch.Tensor
            形状为 (batch_size, 1) 的张量，表示每个响应的优势 A。
        policy_log_probs: torch.Tensor
            形状为 (batch_size, sequence_length) 的张量，正在训练策略的逐 token 对数概率。
        old_log_probs: torch.Tensor
            形状为 (batch_size, sequence_length) 的张量，旧策略的逐 token 对数概率。
        cliprange: float
            裁剪范围 ε（例如 0.2）。

    返回：
        tuple[torch.Tensor, dict[str, torch.Tensor]]
            - 第一个元素：形状为 (batch_size, sequence_length) 的张量，逐 token 的 GRPO-Clip 损失；
            - 第二个元素：包含用于日志记录的张量字典（例如裁剪标记、比例等）。
    """
    ratio = torch.exp(policy_log_probs - old_log_probs)
    clip_ratio = torch.clamp(ratio, 1 - cliprange, 1 + cliprange)
    loss = -torch.min(advantages * ratio, advantages * clip_ratio)
    return loss, {
        "ratio": ratio,
        "clip_ratio": clip_ratio,
    }


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """执行一次 SFT 微批量（microbatch）的前向与反向步骤（占位实现）。

    功能：
        - 基于输入的逐 token 对数概率 `policy_log_probs` 与 `response_mask` 计算交叉熵损失；
        - 仅在回复 token 位置（`response_mask == 1`）处累计损失；
        - 按 `normalize_constant` 进行归一化；
        - 结合 `gradient_accumulation_steps` 对损失进行缩放，并执行 `loss.backward()`；
        - 返回标量损失与用于日志记录的元数据字典。

    参数：
        policy_log_probs: torch.Tensor
            形状为 (batch_size, sequence_length) 的逐 token 对数概率，来自正在训练的 SFT 策略。
        response_mask: torch.Tensor
            形状为 (batch_size, sequence_length) 的 0/1 掩码；1 表示回复 token，0 表示提示或补齐。
        gradient_accumulation_steps: int
            每次优化器 `step` 前要累计的微批量数。
        normalize_constant: float
            用于将掩码求和后的总和进行归一化的常数，通常为 1.0。

    返回：
        tuple[torch.Tensor, dict[str, torch.Tensor]]
            - 第一个元素为标量损失（已按梯度累积进行缩放，便于日志记录）；
            - 第二个元素为包含底层统计量的字典（例如未缩放损失、token 级统计等）。

    """
    per_example_loss = masked_normalize(
        -policy_log_probs,
        response_mask,
        normalize_constant,
        dim=1,
    )
    unscaled_loss = per_example_loss.mean()
    loss = unscaled_loss / gradient_accumulation_steps
    loss.backward()
    return loss, {
        "unscaled_loss": unscaled_loss,
        "per_example_loss": per_example_loss,
    }

def log_generation(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    device: torch.device,
) -> None:
    """
    Log the generation results.
    1. The input prompt.
    2. The response generated by the SFT/RL model.
    3. The ground-truth answer.
    4. The reward information, including format, answer, and total reward.
    5. The average token entropy of the response.
    6. The average response length, average response length for correct responses, and average response length
    for incorrect responses.
    """
    model.eval()
    model.to(device)
    outputs = model.generate(
        input_ids=input_ids,
    )
    # TODO
    print(outputs)


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: List[str],
    repeated_ground_truths: List[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """按组计算与归一化回报（占位实现）。

    功能：
        - 对每个 rollouts 响应与对应的重复 ground-truth 进行打分，得到原始回报 `raw_rewards`；
        - 以 `group_size` 为分组粒度，在每个组内对回报做归一化（减去组均值，
          若 `normalize_by_std=True` 则再除以组标准差与 `advantage_eps` 的和，
          否则仅做中心化），得到 `advantages`；
        - 返回组归一化后的 `advantages`、未归一化的 `raw_rewards` 以及可选的统计信息 `metadata`。

    参数：
        reward_fn: Callable[[str, str], dict[str, float]]
            评分函数：输入 (response, ground_truth)；输出需包含键 "reward"、
            "format_reward"、"answer_reward" 对应的浮点分数。
        rollout_responses: list[str]
            策略生成的响应序列，长度为 rollout_batch_size = n_prompts_per_rollout_batch * group_size。
        repeated_ground_truths: list[str]
            与 `rollout_responses` 等长的 ground-truth 列表（每个样本的 ground-truth 被重复 `group_size` 次）。
        group_size: int
            每个问题的分组大小（每组内的样本数）。
        advantage_eps: float
            防止除零的微小常数，参与标准差归一化时与组内标准差相加。
        normalize_by_std: bool
            若为 True，则在减去组均值后再除以 (组标准差 + advantage_eps)，否则仅减去组均值。

    返回：
        tuple[torch.Tensor, torch.Tensor, dict[str, float]]
            - advantages: 形状为 (rollout_batch_size, ) 的张量，按组归一化后的回报。
            - raw_rewards: 形状为 (rollout_batch_size, ) 的张量，未归一化的原始回报（未减去均值）。
            - metadata: 额外日志统计（如各组的均值、标准差、min/max 等）。

    """
    # results = reward_fn(rollout_responses, repeated_ground_truths)
    results = [reward_fn(response, ground_truth)["reward"] for response, ground_truth in zip(rollout_responses, repeated_ground_truths)]
    flat_rewards = torch.tensor(results) # (n_prompts_per_rollout_batch * group_size, )
    print(flat_rewards.shape, "flat_rewards")
    rewards = rearrange(flat_rewards, "(n g) -> n g", g=group_size)
    print(rewards)
    print(rewards.shape, "rewards")
    mean = rewards.mean(dim=-1, keepdim=True) # (n_prompts_per_rollout_batch, 1)
    print(mean.shape, "mean")
    std = rewards.std(dim=-1, keepdim=True) # (n_prompts_per_rollout_batch, 1)
    raw_rewards = rewards - mean # (n_prompts_per_rollout_batch, group_size)
    advantages = raw_rewards
    if normalize_by_std:
        advantages = advantages / (std + advantage_eps)
    return advantages.flatten(), flat_rewards, {
        "mean": mean,
        "std": std,
        "min": rewards.min(dim=-1).values,
        "max": rewards.max(dim=-1).values,
    }


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """选择并计算所需的策略梯度损失（逐 token），并返回辅助统计信息。

    功能：
        - 作为便捷包装器，根据 `loss_type` 分派到对应的损失例程：
          - "no_baseline": 使用原始回报 `raw_rewards` 与 `policy_log_probs` 计算；
          - "reinforce_with_baseline": 使用（已归一化）优势 `advantages` 与 `policy_log_probs` 计算；
          - "grpo_clip": 使用 `advantages`、`policy_log_probs`、`old_log_probs` 与 `cliprange` 计算 GRPO-Clip。
        - 返回逐 token 的损失张量与底层统计字典。

    参数：
        policy_log_probs: torch.Tensor
            形状为 (batch_size, sequence_length)，正在训练策略的逐 token 对数概率。
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"]
            指定损失类型。
        raw_rewards: torch.Tensor | None
            当 `loss_type == "no_baseline"` 时必须提供；形状为 (batch_size, 1)。
        advantages: torch.Tensor | None
            当 `loss_type in {"reinforce_with_baseline", "grpo_clip"}` 时必须提供；形状为 (batch_size, 1)。
        old_log_probs: torch.Tensor | None
            当 `loss_type == "grpo_clip"` 时必须提供；形状为 (batch_size, sequence_length)。
        cliprange: float | None
            当 `loss_type == "grpo_clip"` 时必须提供；GRPO-Clip 的裁剪系数 ε。

    返回：
        tuple[torch.Tensor, dict[str, torch.Tensor]]
            - 第一个元素：形状为 (batch_size, sequence_length) 的逐 token 损失。
            - 第二个元素：底层统计信息（如比例、裁剪比例等），如无可返回空字典。

    备注：
        - 具体计算逻辑由你实现，可直接调用 `compute_naive_policy_gradient_loss` 或 `compute_grpo_clip_loss`。
        - 建议在内部做参数断言（例如必需张量是否为 None、形状是否匹配等），并将底层元数据汇总为一个字典返回。
    """
    if loss_type == "no_baseline":
        assert raw_rewards is not None
        return compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None
        return compute_naive_policy_gradient_loss(advantages, policy_log_probs)
    elif loss_type == "grpo_clip":
        assert advantages is not None
        assert old_log_probs is not None
        assert cliprange is not None
        return compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")

def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    """计算张量在给定维度上的掩码均值。

    功能：
        计算张量在指定维度上的均值，但只考虑掩码为1（或True）的元素。
        如果dim为None，则计算所有掩码元素的均值。

    参数：
        tensor: torch.Tensor
            要进行均值计算的数据张量。
        mask: torch.Tensor
            与tensor同形状的布尔张量或0-1张量；值为1/True的位置参与均值计算。
        dim: int | None
            计算均值的维度。如果为None，则计算所有掩码元素的均值。

    返回：
        torch.Tensor
            掩码均值；形状遵循tensor.mean(dim)的语义。
    """
    return tensor.masked_fill(~mask, 0).sum(dim=dim) / mask.sum(dim=dim)

if __name__ == "__main__":
    prompt_strs = ["Hello, world"]
    output_strs = ["Goodbye, world"]
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH / "Qwen2.5-Math-1.5B"))
    result = tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)
    print(result)
