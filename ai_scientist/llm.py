"""
LLM 交互模块
============

本模块处理与大型语言模型 (LLM) 的交互。
支持 OpenAI、Anthropic 和 Ollama 等多种提供商。
包含重试逻辑、批量响应生成、令牌跟踪等功能。

主要功能：
1. get_batch_responses_from_llm: 批量获取 LLM 响应。
2. make_llm_call: 执行单个 LLM API 调用。
3. get_response_from_llm: 获取单个 LLM 响应（带重试和历史记录管理）。
4. create_client: 根据模型配置创建相应的 API 客户端。
5. extract_json_between_markers: 从 LLM 输出中提取 JSON 内容。

作者: AI Scientist Team
日期: 2025-01-22
"""

import json
import os
import re
from typing import Any
from ai_scientist.utils.token_tracker import track_token_usage
from ai_scientist.config_loader import get_config, get_llm_config

import anthropic
import backoff
import openai
import requests

MAX_NUM_TOKENS = 4096

# Load available models from configuration
try:
    _conf = get_config()
    AVAILABLE_LLMS = list(_conf.llm_config.models.keys())
except Exception:
    AVAILABLE_LLMS = []


# Get N responses from a single message, used for ensembling.
@backoff.on_exception(
    backoff.expo,
    (
        openai.RateLimitError,
        openai.APITimeoutError,
        openai.InternalServerError,
        anthropic.RateLimitError,
    ),
)
@track_token_usage
def get_batch_responses_from_llm(
    prompt,
    client,
    model,
    system_message,
    print_debug=False,
    msg_history=None,
    temperature=0.7,
    n_responses=1,
) -> tuple[list[str], list[list[dict[str, Any]]]]:
    """
    从 LLM 批量获取响应。

    用于集成学习或生成多个候选项。

    Args:
        prompt (str): 用户输入的提示词。
        client (Any): LLM API 客户端。
        model (str): 模型名称。
        system_message (str): 系统提示词。
        print_debug (bool, optional): 是否打印调试信息。默认为 False。
        msg_history (list, optional): 消息历史记录。默认为 None。
        temperature (float, optional): 采样温度。默认为 0.7。
        n_responses (int, optional): 需要生成的响应数量。默认为 1。

    Returns:
        tuple[list[str], list[list[dict[str, Any]]]]:
            - 响应内容列表。
            - 每个响应对应的更新后的消息历史列表。
    """
    msg = prompt
    if msg_history is None:
        msg_history = []

    # Get provider info
    try:
        conf = get_llm_config(model)
        provider = conf.get("provider")
        # Handle model override if specified in config (e.g. for Llama via OpenRouter)
        # But for batch responses, we usually pass the model name directly unless client is pre-configured?
        # Actually create_client returns (client, model_name). The model_name passed here is likely that one.
    except Exception:
        raise ValueError(
            f"Provider {provider} not supported for model {model} in get_batch_responses_from_llm."
        )

    if provider == "ollama":
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=model.replace("ollama/", ""),
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=n_responses,
            stop=None,
        )
        content = [r.message.content for r in response.choices]
        new_msg_history = [
            new_msg_history + [{"role": "assistant", "content": c}] for c in content
        ]
    elif provider == "openai":
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=n_responses,
            stop=None,
            seed=0,
        )
        content = [r.message.content for r in response.choices]
        new_msg_history = [
            new_msg_history + [{"role": "assistant", "content": c}] for c in content
        ]
    # Add deepseek special handling if needed, or treat as openai
    else:
        raise ValueError(
            f"Provider {provider} not supported for model {model} in get_batch_responses_from_llm."
        )

    if print_debug:
        # Just print the first one.
        print()
        print("*" * 20 + " LLM START " + "*" * 20)
        for j, msg in enumerate(new_msg_history[0]):
            print(f'{j}, {msg["role"]}: {msg["content"]}')
        print(content)
        print("*" * 21 + " LLM END " + "*" * 21)
        print()

    return content, new_msg_history


@track_token_usage
def make_llm_call(client, model, temperature, system_message, prompt):
    """
    执行底层的 LLM API 调用。

    根据不同的提供商（Ollama, OpenAI）调用相应的 API。

    Args:
        client (Any): LLM API 客户端。
        model (str): 模型名称。
        temperature (float): 采样温度。
        system_message (str): 系统提示词。
        prompt (list): 消息列表（包含历史记录和当前用户输入）。

    Returns:
        Any: API 响应对象。
    """
    try:
        conf = get_llm_config(model)
        provider = conf.get("provider")
    except Exception:
        provider = "openai"

    if provider == "ollama":
        return client.chat.completions.create(
            model=model.replace("ollama/", ""),
            messages=[
                {"role": "system", "content": system_message},
                *prompt,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=1,
            stop=None,
        )
    elif provider == "openai":
        # Check for O1/O3 models which have different API params
        if "o1" in model or "o3" in model:
            return client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": system_message},
                    *prompt,
                ],
                temperature=1,  # O1/O3 often fixed temp
                n=1,
                seed=0,
            )
        else:
            return client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    *prompt,
                ],
                temperature=temperature,
                max_tokens=MAX_NUM_TOKENS,
                n=1,
                stop=None,
                seed=0,
            )
    else:
        raise ValueError(
            f"Model {model} with provider {provider} not supported in make_llm_call."
        )


@backoff.on_exception(
    backoff.expo,
    (
        openai.RateLimitError,
        openai.APITimeoutError,
        openai.InternalServerError,
        anthropic.RateLimitError,
    ),
)
def get_response_from_llm(
    prompt,
    client,
    model,
    system_message,
    print_debug=False,
    msg_history=None,
    temperature=0.7,
) -> tuple[str, list[dict[str, Any]]]:
    """
    获取单个 LLM 响应。

    包含异常处理、重试逻辑和消息历史管理。

    Args:
        prompt (str): 用户输入的提示词。
        client (Any): LLM API 客户端。
        model (str): 模型名称。
        system_message (str): 系统提示词。
        print_debug (bool, optional): 是否打印调试信息。默认为 False。
        msg_history (list, optional): 消息历史记录。默认为 None。
        temperature (float, optional): 采样温度。默认为 0.7。

    Returns:
        tuple[str, list[dict[str, Any]]]:
            - 响应内容字符串。
            - 更新后的消息历史列表。
    """
    msg = prompt
    if msg_history is None:
        msg_history = []

    try:
        conf = get_llm_config(model)
        provider = conf.get("provider")
    except Exception as e:
        # logger.warning(f"模型{model}未在配置中找到（{e}），开始自动识别provider")
        raise RuntimeError(f"模型{model}未在配置中找到（{e}），开始自动识别provider")

    if provider == "anthropic":
        new_msg_history = msg_history + [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": msg,
                    }
                ],
            }
        ]
        response = client.messages.create(
            model=model,
            max_tokens=MAX_NUM_TOKENS,
            temperature=temperature,
            system=system_message,
            messages=new_msg_history,
        )
        content = response.content[0].text
        new_msg_history = new_msg_history + [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": content,
                    }
                ],
            }
        ]
    elif provider == "ollama":
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=model.replace("ollama/", ""),
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=1,
            stop=None,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    elif provider == "openai":
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = make_llm_call(
            client,
            model,
            temperature,
            system_message=system_message,
            prompt=new_msg_history,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    else:
        raise ValueError(
            f"Model {model} with provider {provider} not supported in get_response_from_llm."
        )

    if print_debug:
        print()
        print("*" * 20 + " LLM START " + "*" * 20)
        for j, msg in enumerate(new_msg_history):
            print(f'{j}, {msg["role"]}: {msg["content"]}')
        print(content)
        print("*" * 21 + " LLM END " + "*" * 21)
        print()

    return content, new_msg_history


def extract_json_between_markers(llm_output: str) -> dict | None:
    """
    从 LLM 输出中提取 JSON 内容。

    查找 ```json ... ``` 代码块，如果未找到则尝试查找 { ... }。
    包含基本的错误修复逻辑（如去除控制字符）。

    Args:
        llm_output (str): LLM 输出的文本。

    Returns:
        dict | None: 解析后的 JSON 字典，如果失败则返回 None。
    """
    # Regular expression pattern to find JSON content between ```json and ```
    json_pattern = r"```json(.*?)```"
    matches = re.findall(json_pattern, llm_output, re.DOTALL)

    if not matches:
        # Fallback: Try to find any JSON-like content in the output
        json_pattern = r"\{.*?\}"
        matches = re.findall(json_pattern, llm_output, re.DOTALL)

    for json_string in matches:
        json_string = json_string.strip()
        try:
            parsed_json = json.loads(json_string)
            return parsed_json
        except json.JSONDecodeError:
            # Attempt to fix common JSON issues
            try:
                # Remove invalid control characters
                json_string_clean = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
                parsed_json = json.loads(json_string_clean)
                return parsed_json
            except json.JSONDecodeError:
                continue  # Try next match

    return None  # No valid JSON found


def create_client(model) -> tuple[Any, str]:
    """
    根据模型配置创建 API 客户端。

    支持 Anthropic, Ollama, OpenAI 提供商。

    Args:
        model (str): 模型名称。

    Returns:
        tuple[Any, str]: (客户端实例, 模型名称)。
    """
    conf = get_llm_config(model)
    provider = conf.get("provider")
    api_key = conf.get("api_key")
    base_url = conf.get("base_url")

    if provider == "anthropic":
        print(f"Using Anthropic API with model {model}.")
        return anthropic.Anthropic(api_key=api_key), model

    elif provider == "ollama":
        print(f"Using Ollama with model {model}.")
        return openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
        ), model

    elif provider == "openai":
        print(f"Using OpenAI API with model {model}.")
        return openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
        ), model

    else:
        raise ValueError(
            f"Provider {provider} not supported for model {model} in create_client."
        )
