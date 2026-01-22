"""
OpenAI 后端模块
===============

本模块提供了与 OpenAI API 交互的功能，用于发送聊天请求并获取响应。
支持自动重试机制（处理速率限制和超时）以及函数调用（Tool Use）。

主要功能：
1. get_ai_client: 获取配置好的 OpenAI 客户端实例。
2. query: 向 OpenAI 模型发送请求，支持系统消息、用户消息和函数定义。

作者: AI Scientist Team
日期: 2025-01-22
"""

import json
import logging
import time
import os

from .utils import FunctionSpec, OutputType, opt_messages_to_list, backoff_create
from funcy import notnone, once, select_values
import openai
from rich import print
from ai_scientist.config_loader import get_llm_config

logger = logging.getLogger("ai-scientist")


OPENAI_TIMEOUT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)

def get_ai_client(model: str, max_retries=2) -> openai.OpenAI:
    """
    获取 OpenAI 客户端实例。

    根据模型名称加载配置（API Key 和 Base URL），并创建 OpenAI 客户端。

    Args:
        model (str): 模型名称，用于查找配置。
        max_retries (int, optional): 最大重试次数。默认为 2。

    Returns:
        openai.OpenAI: 配置好的 OpenAI 客户端实例。

    Raises:
        ValueError: 如果未找到模型配置。
    """
    conf = get_llm_config(model)
    if conf:
        api_key = conf.get("api_key")
        base_url = conf.get("base_url")
        return openai.OpenAI(api_key=api_key, base_url=base_url, max_retries=max_retries) 
    else:
        raise ValueError("conf is None in conf")


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    """
    向 OpenAI 模型发送查询请求。

    支持发送系统消息和用户消息，并可选择性地提供函数规范以启用工具调用。
    使用指数退避策略处理 API 超时和速率限制。

    Args:
        system_message (str | None): 系统消息内容。
        user_message (str | None): 用户消息内容。
        func_spec (FunctionSpec | None, optional): 函数规范，用于工具调用。默认为 None。
        **model_kwargs: 传递给 API 的其他参数（如 model, temperature 等）。

    Returns:
        tuple: 包含以下元素的元组：
            - OutputType: 模型响应内容（字符串或解析后的 JSON）。
            - float: 请求耗时（秒）。
            - int: 输入 Token 数量。
            - int: 输出 Token 数量。
            - dict: 包含响应元数据的字典（系统指纹、模型名、创建时间）。
    """
    client = get_ai_client(model_kwargs.get("model"), max_retries=0)
    filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore

    messages = opt_messages_to_list(system_message, user_message)

    if func_spec is not None:
        filtered_kwargs["tools"] = [func_spec.as_openai_tool_dict]
        # force the model to use the function
        filtered_kwargs["tool_choice"] = func_spec.openai_tool_choice_dict

    # Determine provider to handle model name formatting
    model = model_kwargs.get("model")
    try:
        conf = get_llm_config(model)
        provider = conf.get("provider")
    except Exception:
        raise ValueError("provider is None in conf in query function")

    if provider == "ollama" and filtered_kwargs.get("model", "").startswith("ollama/"):
       filtered_kwargs["model"] = filtered_kwargs["model"].replace("ollama/", "")

    t0 = time.time()
    completion = backoff_create(
        client.chat.completions.create,
        OPENAI_TIMEOUT_EXCEPTIONS,
        messages=messages,
        **filtered_kwargs,
    )
    req_time = time.time() - t0

    choice = completion.choices[0]

    if func_spec is None:
        output = choice.message.content
    else:
        assert (
            choice.message.tool_calls
        ), f"function_call is empty, it is not a function call: {choice.message}"
        assert (
            choice.message.tool_calls[0].function.name == func_spec.name
        ), "Function name mismatch"
        try:
            print(f"[cyan]Raw func call response: {choice}[/cyan]")
            output = json.loads(choice.message.tool_calls[0].function.arguments)
        except json.JSONDecodeError as e:
            logger.error(
                f"Error decoding the function arguments: {choice.message.tool_calls[0].function.arguments}"
            )
            raise e

    in_tokens = completion.usage.prompt_tokens
    out_tokens = completion.usage.completion_tokens

    info = {
        "system_fingerprint": completion.system_fingerprint,
        "model": completion.model,
        "created": completion.created,
    }

    return output, req_time, in_tokens, out_tokens, info
