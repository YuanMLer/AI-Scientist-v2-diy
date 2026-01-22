"""
后端工具模块
============

本模块提供与 LLM 后端交互的通用工具函数。
包含重试机制、消息格式化、Prompt 编译以及函数调用规格定义。

主要功能：
1. backoff_create: 带有指数退避重试机制的 API 调用包装器。
2. opt_messages_to_list: 将系统消息和用户消息转换为列表格式。
3. compile_prompt_to_md: 将不同类型的 Prompt (字符串、列表、字典) 编译为 Markdown 格式。
4. FunctionSpec: 定义 LLM 可调用的工具函数规格。

作者: AI Scientist Team
日期: 2025-01-22
"""

from dataclasses import dataclass

import jsonschema
from dataclasses_json import DataClassJsonMixin

PromptType = str | dict | list
FunctionCallType = dict
OutputType = str | FunctionCallType


import backoff
import logging
from typing import Callable

logger = logging.getLogger("ai-scientist")


@backoff.on_predicate(
    wait_gen=backoff.expo,
    max_value=60,
    factor=1.5,
)
def backoff_create(
    create_fn: Callable, retry_exceptions: list[Exception], *args, **kwargs
):
    """
    带有指数退避重试机制的 API 调用包装器。

    如果发生指定的异常，将自动重试，等待时间按指数增长。

    Args:
        create_fn (Callable): 要调用的 API 创建函数。
        retry_exceptions (list[Exception]): 需要重试的异常类型列表。
        *args: 传递给 create_fn 的位置参数。
        **kwargs: 传递给 create_fn 的关键字参数。

    Returns:
        Any: create_fn 的返回值，如果失败则可能返回 False。
    """
    try:
        return create_fn(*args, **kwargs)
    except retry_exceptions as e:
        logger.info(f"Backoff exception: {e}")
        return False


def opt_messages_to_list(
    system_message: str | None, user_message: str | None
) -> list[dict[str, str]]:
    """
    将可选的系统消息和用户消息转换为消息列表。

    Args:
        system_message (str | None): 系统提示消息，如果为 None 则忽略。
        user_message (str | None): 用户提示消息，如果为 None 则忽略。

    Returns:
        list[dict[str, str]]: 包含角色和内容的消息字典列表。
    """
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    if user_message:
        messages.append({"role": "user", "content": user_message})
    return messages


def compile_prompt_to_md(prompt: PromptType, _header_depth: int = 1) -> str:
    """
    将 Prompt 转换为 Markdown 格式。

    支持字符串、列表和字典类型的 Prompt。
    - 字符串：直接返回。
    - 列表：转换为 Markdown 列表项。
    - 字典：递归转换为 Markdown 标题和内容。

    Args:
        prompt (PromptType): 输入的 Prompt，可以是字符串、列表或字典。
        _header_depth (int): 当前 Markdown 标题深度，用于递归调用。默认为 1。

    Returns:
        str: 转换后的 Markdown 字符串。

    Raises:
        ValueError: 如果 Prompt 类型不支持。
    """
    try:
        logger.debug(f"compile_prompt_to_md input: type={type(prompt)}")
        if isinstance(prompt, (list, dict)):
            logger.debug(f"prompt content: {prompt}")

        if prompt is None:
            return ""

        if isinstance(prompt, str):
            return prompt.strip() + "\n"

        if isinstance(prompt, list):
            # Handle empty list case
            if not prompt:
                return ""
            # Special handling for multi-modal messages
            if all(isinstance(item, dict) and "type" in item for item in prompt):
                # For multi-modal messages, just pass through without modification
                return prompt

            try:
                result = "\n".join([f"- {s.strip()}" for s in prompt] + ["\n"])
                return result
            except Exception as e:
                logger.error(f"Error processing list items: {e}")
                logger.error("List contents:")
                for i, item in enumerate(prompt):
                    logger.error(f"  Item {i}: type={type(item)}, value={item}")
                raise

        if isinstance(prompt, dict):
            # Check if this is a single multi-modal message
            if "type" in prompt:
                return prompt

            # Regular dict processing
            try:
                out = []
                header_prefix = "#" * _header_depth
                for k, v in prompt.items():
                    logger.debug(f"Processing dict key: {k}")
                    out.append(f"{header_prefix} {k}\n")
                    out.append(compile_prompt_to_md(v, _header_depth=_header_depth + 1))
                return "\n".join(out)
            except Exception as e:
                logger.error(f"Error processing dict: {e}")
                logger.error(f"Dict contents: {prompt}")
                raise

        raise ValueError(f"Unsupported prompt type: {type(prompt)}")

    except Exception as e:
        logger.error("Error in compile_prompt_to_md:")
        logger.error(f"Input type: {type(prompt)}")
        logger.error(f"Input content: {prompt}")
        logger.error(f"Error: {str(e)}")
        raise


@dataclass
class FunctionSpec(DataClassJsonMixin):
    """
    函数规格说明类。

    用于定义 LLM 可调用的工具函数（Tools/Functions），包含名称、JSON Schema 和描述。
    支持转换为 OpenAI API 所需的工具格式。

    Args:
        name (str): 函数名称。
        json_schema (dict): 函数参数的 JSON Schema。
        description (str): 函数描述。
    """
    name: str
    json_schema: dict  # JSON schema
    description: str

    def __post_init__(self):
        """
        初始化后验证。

        检查提供的 JSON Schema 是否有效。
        """
        # validate the schema
        jsonschema.Draft7Validator.check_schema(self.json_schema)

    @property
    def as_openai_tool_dict(self):
        """
        转换为 OpenAI 工具定义字典。

        Returns:
            dict: OpenAI API 格式的工具定义。
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.json_schema,
            },
        }

    @property
    def openai_tool_choice_dict(self):
        """
        获取 OpenAI 工具选择字典。

        用于指定强制调用该函数。

        Returns:
            dict: OpenAI API 格式的工具选择字典。
        """
        return {
            "type": "function",
            "function": {"name": self.name},
        }
