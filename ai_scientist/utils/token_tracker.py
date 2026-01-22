"""
Token 追踪与成本计算模块
======================

本模块用于追踪 LLM API 调用的 Token 使用量，并根据预定义的价格表计算成本。
支持多种 OpenAI 模型，并能记录每次交互的详细信息（提示词、响应、时间戳等）。

主要功能：
1. TokenTracker 类: 核心追踪器，管理 Token 计数、交互记录和成本计算。
2. track_token_usage 装饰器: 用于自动追踪被装饰函数的 Token 使用情况。

作者: AI Scientist Team
日期: 2025-01-22
"""

from functools import wraps
from typing import Dict, Optional, List
import tiktoken
from collections import defaultdict
import asyncio
from datetime import datetime
import logging


class TokenTracker:
    """
    Token 追踪器类。

    用于跟踪 Prompt、Completion、Reasoning 和 Cached Token 的数量。
    Reasoning Token 包含在 Completion Token 中。
    Cached Token 包含在 Prompt Token 中。
    同时记录所有的交互历史（Prompt, Response, Timestamp）。
    假设这些信息直接从 LLM 的响应中获取，而不是本地计算。
    """

    def __init__(self):
        """
        初始化 TokenTracker。

        设置默认的 Token 计数器和交互记录列表。
        定义不同模型的定价表（每百万 Token 的价格）。
        """
        self.token_counts = defaultdict(
            lambda: {"prompt": 0, "completion": 0, "reasoning": 0, "cached": 0}
        )
        self.interactions = defaultdict(list)

        self.MODEL_PRICES = {
            "gpt-4o-2024-11-20": {
                "prompt": 2.5 / 1000000,  # $2.50 per 1M tokens
                "cached": 1.25 / 1000000,  # $1.25 per 1M tokens
                "completion": 10 / 1000000,  # $10.00 per 1M tokens
            },
            "gpt-4o-2024-08-06": {
                "prompt": 2.5 / 1000000,  # $2.50 per 1M tokens
                "cached": 1.25 / 1000000,  # $1.25 per 1M tokens
                "completion": 10 / 1000000,  # $10.00 per 1M tokens
            },
            "gpt-4o-2024-05-13": {  # this ver does not support cached tokens
                "prompt": 5.0 / 1000000,  # $5.00 per 1M tokens
                "completion": 15 / 1000000,  # $15.00 per 1M tokens
            },
            "gpt-4o-mini-2024-07-18": {
                "prompt": 0.15 / 1000000,  # $0.15 per 1M tokens
                "cached": 0.075 / 1000000,  # $0.075 per 1M tokens
                "completion": 0.6 / 1000000,  # $0.60 per 1M tokens
            },
            "o1-2024-12-17": {
                "prompt": 15 / 1000000,  # $15.00 per 1M tokens
                "cached": 7.5 / 1000000,  # $7.50 per 1M tokens
                "completion": 60 / 1000000,  # $60.00 per 1M tokens
            },
            "o1-preview-2024-09-12": {
                "prompt": 15 / 1000000,  # $15.00 per 1M tokens
                "cached": 7.5 / 1000000,  # $7.50 per 1M tokens
                "completion": 60 / 1000000,  # $60.00 per 1M tokens
            },
            "o3-mini-2025-01-31": {
                "prompt": 1.1 / 1000000,  # $1.10 per 1M tokens
                "cached": 0.55 / 1000000,  # $0.55 per 1M tokens
                "completion": 4.4 / 1000000,  # $4.40 per 1M tokens
            },
        }

    def add_tokens(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        reasoning_tokens: int,
        cached_tokens: int,
    ):
        """
        添加 Token 使用记录。

        Args:
            model (str): 模型名称。
            prompt_tokens (int): 提示词 Token 数量。
            completion_tokens (int): 完成词 Token 数量。
            reasoning_tokens (int): 推理 Token 数量（包含在 completion_tokens 中）。
            cached_tokens (int): 缓存 Token 数量（包含在 prompt_tokens 中）。
        """
        self.token_counts[model]["prompt"] += prompt_tokens
        self.token_counts[model]["completion"] += completion_tokens
        self.token_counts[model]["reasoning"] += reasoning_tokens
        self.token_counts[model]["cached"] += cached_tokens

    def add_interaction(
        self,
        model: str,
        system_message: str,
        prompt: str,
        response: str,
        timestamp: datetime,
    ):
        """
        记录一次与模型的交互。

        Args:
            model (str): 模型名称。
            system_message (str): 系统提示词。
            prompt (str): 用户提示词。
            response (str): 模型响应内容。
            timestamp (datetime): 交互发生的时间戳。
        """
        self.interactions[model].append(
            {
                "system_message": system_message,
                "prompt": prompt,
                "response": response,
                "timestamp": timestamp,
            }
        )

    def get_interactions(self, model: Optional[str] = None) -> Dict[str, List[Dict]]:
        """
        获取交互记录。

        Args:
            model (Optional[str]): 可选的模型名称过滤。如果为 None，则返回所有模型的交互记录。

        Returns:
            Dict[str, List[Dict]]: 交互记录字典，键为模型名称，值为交互列表。
        """
        if model:
            return {model: self.interactions[model]}
        return dict(self.interactions)

    def reset(self):
        """重置所有的 Token 计数和交互记录。"""
        self.token_counts = defaultdict(
            lambda: {"prompt": 0, "completion": 0, "reasoning": 0, "cached": 0}
        )
        self.interactions = defaultdict(list)
        # self._encoders = {}

    def calculate_cost(self, model: str) -> float:
        """
        计算特定模型的预估成本。

        Args:
            model (str): 模型名称。

        Returns:
            float: 计算出的成本（美元）。如果模型价格未知，则返回 0.0。
        """
        if model not in self.MODEL_PRICES:
            logging.warning(f"Price information not available for model {model}")
            return 0.0

        prices = self.MODEL_PRICES[model]
        tokens = self.token_counts[model]

        # 计算提示词和完成词的成本
        if "cached" in prices:
            # 如果支持缓存价格，则分别计算非缓存和缓存部分的成本
            prompt_cost = (tokens["prompt"] - tokens["cached"]) * prices["prompt"]
            cached_cost = tokens["cached"] * prices["cached"]
        else:
            # 否则全部按提示词价格计算
            prompt_cost = tokens["prompt"] * prices["prompt"]
            cached_cost = 0
        completion_cost = tokens["completion"] * prices["completion"]

        return prompt_cost + cached_cost + completion_cost

    def get_summary(self) -> Dict[str, Dict[str, int]]:
        """
        获取所有模型的 Token 使用和成本汇总。

        Returns:
            Dict[str, Dict[str, int]]: 汇总字典，包含每个模型的 Token 计数和成本。
        """
        # return dict(self.token_counts)
        summary = {}
        for model, tokens in self.token_counts.items():
            summary[model] = {
                "tokens": tokens.copy(),
                "cost (USD)": self.calculate_cost(model),
            }
        return summary


# 全局 Token 追踪器实例
token_tracker = TokenTracker()


def track_token_usage(func):
    """
    装饰器：用于追踪异步或同步函数的 Token 使用情况。

    该装饰器会拦截函数的返回值，提取 usage 信息，并更新全局 token_tracker。
    要求被装饰函数的返回值必须包含 usage 属性（通常是 LLM 的响应对象）。
    同时要求调用时提供 'prompt' 或 'system_message' 参数。

    Args:
        func: 被装饰的函数。

    Returns:
        wrapper: 包装后的函数。
    """
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        prompt = kwargs.get("prompt")
        system_message = kwargs.get("system_message")
        if not prompt and not system_message:
            raise ValueError(
                "Either 'prompt' or 'system_message' must be provided for token tracking"
            )

        logging.info("args: ", args)
        logging.info("kwargs: ", kwargs)

        result = await func(*args, **kwargs)
        model = result.model
        timestamp = result.created

        if hasattr(result, "usage") and result.usage.completion_tokens_details is not None:
            token_tracker.add_tokens(
                model,
                result.usage.prompt_tokens,
                result.usage.completion_tokens,
                result.usage.completion_tokens_details.reasoning_tokens,
                (
                    result.usage.prompt_tokens_details.cached_tokens
                    if hasattr(result.usage, "prompt_tokens_details")
                    else 0
                ),
            )
            # 添加交互详情
            token_tracker.add_interaction(
                model,
                system_message,
                prompt,
                result.choices[
                    0
                ].message.content,  # 假设响应内容在 choices[0].message.content 中
                timestamp,
            )
        return result

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        prompt = kwargs.get("prompt")
        system_message = kwargs.get("system_message")
        if not prompt and not system_message:
            raise ValueError(
                "Either 'prompt' or 'system_message' must be provided for token tracking"
            )
        result = func(*args, **kwargs)
        model = result.model
        timestamp = result.created
        logging.info("args: ", args)
        logging.info("kwargs: ", kwargs)

        if hasattr(result, "usage") and result.usage.completion_tokens_details is not None:
            token_tracker.add_tokens(
                model,
                result.usage.prompt_tokens,
                result.usage.completion_tokens,
                result.usage.completion_tokens_details.reasoning_tokens,
                (
                    result.usage.prompt_tokens_details.cached_tokens
                    if hasattr(result.usage, "prompt_tokens_details")
                    else 0
                ),
            )
            # 添加交互详情
            token_tracker.add_interaction(
                model,
                system_message,
                prompt,
                result.choices[
                    0
                ].message.content,  # 假设响应内容在 choices[0].message.content 中
                timestamp,
            )
        return result

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
