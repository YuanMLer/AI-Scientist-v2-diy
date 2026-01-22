"""
基础工具模块
============

定义了所有工具的抽象基类 BaseTool。
任何集成到 AI Scientist 系统中的工具都应继承此类并实现 `use_tool` 方法。

主要功能：
1. BaseTool: 定义工具的通用接口（名称、描述、参数）。

作者: AI Scientist Team
日期: 2025-01-22
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseTool(ABC):
    """
    自定义工具的抽象基类。

    Attributes:
        name (str): 工具的名称。
        description (str): 工具功能的简要描述。
        parameters (list): 工具所需的参数列表，每个参数应为一个包含 'name', 'type', 'description' 的字典。

    Usage:
        要使用此类，请继承它并实现 `use_tool` 抽象方法。
    """

    def __init__(self, name: str, description: str, parameters: List[Dict[str, Any]]):
        self.name = name
        self.description = description
        self.parameters = parameters

    @abstractmethod
    def use_tool(self, **kwargs) -> Any:
        """
        抽象方法，子类应实现此方法以定义工具的具体功能。

        Args:
            **kwargs: 传递给工具的参数。

        Returns:
            Any: 工具执行的结果。
        """
        pass
