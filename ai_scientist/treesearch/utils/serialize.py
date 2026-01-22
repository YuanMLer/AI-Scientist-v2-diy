"""
序列化工具模块
==============

本模块提供了对象（特别是 Journal 和 Node）的 JSON 序列化和反序列化功能。
处理自定义 Dataclasses 的转换，并管理节点间的父子关系引用。

主要功能：
1. dumps_json: 将对象序列化为 JSON 字符串。
2. dump_json: 将对象序列化并保存到文件。
3. loads_json: 从 JSON 字符串反序列化为对象。
4. load_json: 从文件读取并反序列化为对象。
5. parse_markdown_to_dict: 从 Markdown 内容中解析键值对字典。

作者: AI Scientist Team
日期: 2025-01-22
"""

import copy
import json
from pathlib import Path
from typing import Type, TypeVar
import re

import dataclasses_json
from ..journal import Journal, Node


def dumps_json(obj: dataclasses_json.DataClassJsonMixin):
    """
    将 Dataclasses（如 Journal）序列化为 JSON 字符串。

    处理 Journal 对象中的循环引用（节点父子关系），将其转换为 ID 引用。

    Args:
        obj (dataclasses_json.DataClassJsonMixin): 要序列化的对象。

    Returns:
        str: JSON 字符串。
    """
    if isinstance(obj, Journal):
        obj = copy.deepcopy(obj)
        node2parent = {}
        for n in obj.nodes:
            if n.parent is not None:
                # Handle both Node objects and string IDs
                parent_id = n.parent.id if isinstance(n.parent, Node) else n.parent
                node2parent[n.id] = parent_id
        for n in obj.nodes:
            n.parent = None
            n.children = set()

    obj_dict = obj.to_dict()

    if isinstance(obj, Journal):
        obj_dict["node2parent"] = node2parent
        obj_dict["__version"] = "2"

    return json.dumps(obj_dict, separators=(",", ":"))


def dump_json(obj: dataclasses_json.DataClassJsonMixin, path: Path):
    """
    将对象序列化并保存到 JSON 文件。

    Args:
        obj (dataclasses_json.DataClassJsonMixin): 要序列化的对象。
        path (Path): 目标文件路径。
    """
    with open(path, "w") as f:
        f.write(dumps_json(obj))


G = TypeVar("G", bound=dataclasses_json.DataClassJsonMixin)


def loads_json(s: str, cls: Type[G]) -> G:
    """
    将 JSON 字符串反序列化为 AIDE Dataclasses。

    如果是 Journal 对象，还会重建节点间的父子关系。

    Args:
        s (str): JSON 字符串。
        cls (Type[G]): 目标类。

    Returns:
        G: 反序列化后的对象实例。
    """
    obj_dict = json.loads(s)
    obj = cls.from_dict(obj_dict)

    if isinstance(obj, Journal):
        id2nodes = {n.id: n for n in obj.nodes}
        for child_id, parent_id in obj_dict["node2parent"].items():
            id2nodes[child_id].parent = id2nodes[parent_id]
            id2nodes[child_id].__post_init__()
    return obj


def load_json(path: Path, cls: Type[G]) -> G:
    """
    从 JSON 文件加载并反序列化对象。

    Args:
        path (Path): JSON 文件路径。
        cls (Type[G]): 目标类。

    Returns:
        G: 反序列化后的对象实例。
    """
    with open(path, "r") as f:
        return loads_json(f.read(), cls)


def parse_markdown_to_dict(content: str):
    """
    解析类似 JSON 的 Markdown 内容为字典。

    读取包含 "Key": "Value" 格式行的文件内容，支持多行值。

    Args:
        content (str): Markdown 文本内容。

    Returns:
        dict: 解析出的字典。
    """

    pattern = r'"([^"]+)"\s*:\s*"([^"]*?)"(?:,\s*|\s*$)'

    matches = re.findall(pattern, content, flags=re.DOTALL)

    data_dict = {}
    for key, value in matches:
        data_dict[key] = value

    return data_dict
