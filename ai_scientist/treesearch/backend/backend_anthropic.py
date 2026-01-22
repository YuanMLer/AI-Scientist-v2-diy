"""
Anthropic 后端模块
==================

本模块提供了与 Anthropic API (通过 Bedrock 或直接调用) 交互的功能。
用于发送聊天请求并获取响应，支持 Claude 模型。

主要功能：
1. get_ai_client: 获取 Anthropic 客户端实例。
2. query: 向 Anthropic 模型发送请求，处理系统消息和用户消息。

作者: AI Scientist Team
日期: 2025-01-22
"""

import time
import os

from .utils import FunctionSpec, OutputType, opt_messages_to_list, backoff_create
from funcy import notnone, once, select_values
import anthropic
from ai_scientist.config_loader import get_llm_config


ANTHROPIC_TIMEOUT_EXCEPTIONS = (
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.InternalServerError,
    anthropic.APIStatusError,
)

def get_ai_client(model : str, max_retries=2) -> anthropic.AnthropicBedrock:
    """
    获取 Anthropic 客户端实例。

    目前配置为使用 AnthropicBedrock 客户端。

    Args:
        model (str): 模型名称（此处未使用，但保留以保持接口一致性）。
        max_retries (int, optional): 最大重试次数。默认为 2。

    Returns:
        anthropic.AnthropicBedrock: Anthropic Bedrock 客户端实例。
    """
    client = anthropic.AnthropicBedrock(max_retries=max_retries)
    return client

def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    """
    向 Anthropic 模型发送查询请求。

    处理消息格式（Anthropic 将 system 消息作为独立参数），支持自动重试。
    目前不支持函数调用 (Function Calling)。

    Args:
        system_message (str | None): 系统消息内容。
        user_message (str | None): 用户消息内容。
        func_spec (FunctionSpec | None, optional): 函数规范。Anthropic 后端暂不支持，若提供将抛出 NotImplementedError。
        **model_kwargs: 传递给 API 的其他参数（如 model, temperature, max_tokens 等）。

    Returns:
        tuple: 包含以下元素的元组：
            - OutputType: 模型响应内容（字符串）。
            - float: 请求耗时（秒）。
            - int: 输入 Token 数量。
            - int: 输出 Token 数量。
            - dict: 包含响应元数据的字典（停止原因）。

    Raises:
        NotImplementedError: 如果提供了 func_spec。
    """
    client = get_ai_client(model_kwargs.get("model"), max_retries=0)

    filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore
    if "max_tokens" not in filtered_kwargs:
        filtered_kwargs["max_tokens"] = 8192  # default for Claude models

    if func_spec is not None:
        raise NotImplementedError(
            "Anthropic does not support function calling for now."
        )

    # Anthropic doesn't allow not having a user messages
    # if we only have system msg -> use it as user msg
    if system_message is not None and user_message is None:
        system_message, user_message = user_message, system_message

    # Anthropic passes the system messages as a separate argument
    if system_message is not None:
        filtered_kwargs["system"] = system_message

    messages = opt_messages_to_list(None, user_message)

    t0 = time.time()
    message = backoff_create(
        client.messages.create,
        ANTHROPIC_TIMEOUT_EXCEPTIONS,
        messages=messages,
        **filtered_kwargs,
    )
    req_time = time.time() - t0
    # print(filtered_kwargs)

    if "thinking" in filtered_kwargs:
        assert (
            len(message.content) == 2
            and message.content[0].type == "thinking"
            and message.content[1].type == "text"
        )
        output: str = message.content[1].text
    else:
        assert len(message.content) == 1 and message.content[0].type == "text"
        output: str = message.content[0].text

    in_tokens = message.usage.input_tokens
    out_tokens = message.usage.output_tokens

    info = {
        "stop_reason": message.stop_reason,
    }

    return output, req_time, in_tokens, out_tokens, info
