"""
视觉语言模型 (VLM) 交互模块
===========================

本模块处理与视觉语言模型 (VLM) 的交互，支持图像输入。
主要用于图像分析、绘图评估等任务。

主要功能：
1. encode_image_to_base64: 将图像文件转换为 Base64 编码字符串。
2. make_llm_call: 执行 VLM 的纯文本调用（虽然名字叫 make_llm_call，但在本模块中主要用于 VLM 的文本部分或兼容性）。
3. make_vlm_call: 执行包含图像输入的 VLM API 调用。
4. get_response_from_vlm: 获取单个 VLM 响应（包含图像处理和重试逻辑）。
5. get_batch_responses_from_vlm: 批量获取 VLM 响应。
6. create_client: 创建 VLM API 客户端。

作者: AI Scientist Team
日期: 2025-01-22
"""

import base64
from typing import Any
import re
import json
import backoff
import openai
import os
from PIL import Image
from ai_scientist.utils.token_tracker import track_token_usage
from ai_scientist.config_loader import get_config, get_llm_config

MAX_NUM_TOKENS = 4096

# Load available models from configuration
try:
    _conf = get_config()
    AVAILABLE_VLMS = list(_conf.llm_config.models.keys())
except Exception:
    AVAILABLE_VLMS = []


def encode_image_to_base64(image_path: str) -> str:
    """
    将图像转换为 Base64 字符串。

    用于将图像数据嵌入到 API 请求中。
    如果图像是 RGBA 模式，会自动转换为 RGB。

    Args:
        image_path (str): 图像文件路径。

    Returns:
        str: Base64 编码的图像字符串。
    """
    with Image.open(image_path) as img:
        # Convert RGBA to RGB if necessary
        if img.mode == "RGBA":
            img = img.convert("RGB")

        # Save to bytes
        import io

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()

    return base64.b64encode(image_bytes).decode("utf-8")


@track_token_usage
def make_llm_call(client, model, temperature, system_message, prompt):
    """
    执行纯文本的 LLM/VLM 调用。

    此函数主要用于不需要图像输入的场景，或者作为某些 VLM 的兼容层。

    Args:
        client (Any): API 客户端。
        model (str): 模型名称。
        temperature (float): 采样温度。
        system_message (str): 系统提示词。
        prompt (list): 消息列表。

    Returns:
        Any: API 响应对象。
    """
    try:
        conf = get_llm_config(model)
        provider = conf.get("provider")
    except Exception:
        raise ValueError(
            f"Provider {provider} not supported for model {model} in make_llm_call."
        )

    if provider == "ollama":
        print(f"Using Ollama API with model {model}.")
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
            seed=0,
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
                temperature=1,
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
            f"Provider {provider} not supported for model {model} in make_llm_call."
        )


@track_token_usage
def make_vlm_call(client, model, temperature, system_message, prompt):
    """
    执行包含图像输入的 VLM API 调用。

    支持 Ollama 和 OpenAI 格式的 API。

    Args:
        client (Any): API 客户端。
        model (str): 模型名称。
        temperature (float): 采样温度。
        system_message (str): 系统提示词。
        prompt (list): 消息列表（包含图像数据）。

    Returns:
        Any: API 响应对象。
    """
    try:
        conf = get_llm_config(model)
        provider = conf.get("provider")
    except Exception:
        raise ValueError(
            f"Provider {provider} not supported for model {model} in make_vlm_call."
        )

    if provider == "ollama":
        return client.chat.completions.create(
            model=model.replace("ollama/", ""),
            messages=[
                {"role": "system", "content": system_message},
                *prompt,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
        )
    elif provider == "openai":
        return client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                *prompt,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
        )
    else:
        raise ValueError(f"Model {model} not supported.")


def prepare_vlm_prompt(msg, image_paths, max_images):
    """
    准备 VLM 提示词（占位符函数）。

    目前未实现具体逻辑。
    """
    pass


@backoff.on_exception(
    backoff.expo,
    (
        openai.RateLimitError,
        openai.APITimeoutError,
    ),
)
def get_response_from_vlm(
    msg: str,
    image_paths: str | list[str],
    client: Any,
    model: str,
    system_message: str,
    print_debug: bool = False,
    msg_history: list[dict[str, Any]] | None = None,
    temperature: float = 0.7,
    max_images: int = 25,
) -> tuple[str, list[dict[str, Any]]]:
    """
    从 VLM 获取单个响应。

    处理图像加载、Base64 编码、消息构建和 API 调用。
    支持多张图像输入。

    Args:
        msg (str): 用户输入的文本消息。
        image_paths (str | list[str]): 图像文件路径或路径列表。
        client (Any): API 客户端。
        model (str): 模型名称。
        system_message (str): 系统提示词。
        print_debug (bool, optional): 是否打印调试信息。默认为 False。
        msg_history (list, optional): 消息历史记录。默认为 None。
        temperature (float, optional): 采样温度。默认为 0.7。
        max_images (int, optional): 最大图像数量。默认为 25。

    Returns:
        tuple[str, list[dict[str, Any]]]:
            - 响应内容字符串。
            - 更新后的消息历史列表。
    """
    if msg_history is None:
        msg_history = []

    # Check model availability - now using config so we trust the input or check config
    # We allow the call if we can create a client for it.

    # Convert single image path to list for consistent handling
    if isinstance(image_paths, str):
        image_paths = [image_paths]

    # Create content list starting with the text message
    content = [{"type": "text", "text": msg}]

    # Add each image to the content list
    for image_path in image_paths[:max_images]:
        base64_image = encode_image_to_base64(image_path)
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "low",
                },
            }
        )
    # Construct message with all images
    new_msg_history = msg_history + [{"role": "user", "content": content}]

    response = make_vlm_call(
        client,
        model,
        temperature,
        system_message=system_message,
        prompt=new_msg_history,
    )

    content = response.choices[0].message.content
    new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]

    if print_debug:
        print()
        print("*" * 20 + " VLM START " + "*" * 20)
        for j, msg in enumerate(new_msg_history):
            print(f'{j}, {msg["role"]}: {msg["content"]}')
        print(content)
        print("*" * 21 + " VLM END " + "*" * 21)
        print()

    return content, new_msg_history


def create_client(model: str) -> tuple[Any, str]:
    """
    创建 VLM API 客户端。

    Args:
        model (str): 模型名称。

    Returns:
        tuple[Any, str]: (客户端实例, 模型名称)。
    """
    conf = get_llm_config(model)
    if conf is not None:
        provider = conf.get("provider")
        api_key = conf.get("api_key")
        base_url = conf.get("base_url")
        print(f"Using {provider} API with model {model}.")
    else:
        raise ValueError(f"conf is None")

    return openai.OpenAI(api_key=api_key, base_url=base_url), model


def extract_json_between_markers(llm_output: str) -> dict | None:
    """
    从 VLM 输出中提取 JSON 内容。

    Args:
        llm_output (str): VLM 输出文本。

    Returns:
        dict | None: 解析后的 JSON 字典。
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


@backoff.on_exception(
    backoff.expo,
    (
        openai.RateLimitError,
        openai.APITimeoutError,
    ),
)
def get_batch_responses_from_vlm(
    msg: str,
    image_paths: str | list[str],
    client: Any,
    model: str,
    system_message: str,
    print_debug: bool = False,
    msg_history: list[dict[str, Any]] | None = None,
    temperature: float = 0.7,
    n_responses: int = 1,
    max_images: int = 200,
) -> tuple[list[str], list[list[dict[str, Any]]]]:
    """
    从 VLM 批量获取响应。

    用于获取多个不同的分析结果或建议。

    Args:
        msg (str): 用户输入的文本消息。
        image_paths (str | list[str]): 图像文件路径或路径列表。
        client (Any): API 客户端。
        model (str): 模型名称。
        system_message (str): 系统提示词。
        print_debug (bool, optional): 是否打印调试信息。默认为 False。
        msg_history (list, optional): 消息历史记录。默认为 None。
        temperature (float, optional): 采样温度。默认为 0.7。
        n_responses (int, optional): 需要生成的响应数量。默认为 1。
        max_images (int, optional): 最大图像数量。默认为 200。

    Returns:
        tuple[list[str], list[list[dict[str, Any]]]]:
            - 响应内容列表。
            - 每个响应对应的更新后的消息历史列表。
    """
    if msg_history is None:
        msg_history = []

    # Convert single image path to list
    if isinstance(image_paths, str):
        image_paths = [image_paths]

    # Create content list with text and images
    content = [{"type": "text", "text": msg}]
    for image_path in image_paths[:max_images]:
        base64_image = encode_image_to_base64(image_path)
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "low",
                },
            }
        )

    # Construct message with all images
    new_msg_history = msg_history + [{"role": "user", "content": content}]

    try:
        conf = get_llm_config(model)
        provider = conf.get("provider")
    except Exception:
        raise ValueError("provider is not provided!")

    if provider == "ollama":
        response = client.chat.completions.create(
            model=model.replace("ollama/", ""),
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=n_responses,
            seed=0,
        )
    else:
        # Get multiple responses
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=n_responses,
            seed=0,
        )

    # Extract content from all responses
    contents = [r.message.content for r in response.choices]
    new_msg_histories = [
        new_msg_history + [{"role": "assistant", "content": c}] for c in contents
    ]

    if print_debug:
        # Just print the first response
        print()
        print("*" * 20 + " VLM START " + "*" * 20)
        for j, msg in enumerate(new_msg_histories[0]):
            print(f'{j}, {msg["role"]}: {msg["content"]}')
        print(contents[0])
        print("*" * 21 + " VLM END " + "*" * 21)
        print()

    return contents, new_msg_histories
