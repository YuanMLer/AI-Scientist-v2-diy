"""
Semantic Scholar 搜索工具模块
=============================

本模块提供与 Semantic Scholar API 交互的工具，用于搜索学术文献。
支持通过 SemanticScholarSearchTool 类或独立的 search_for_papers 函数进行搜索。

主要功能：
1. SemanticScholarSearchTool: 封装了搜索功能的工具类，继承自 BaseTool。
2. search_for_papers: 执行搜索请求、处理重试逻辑和格式化结果。

作者: AI Scientist Team
日期: 2025-01-22
"""

import os
import requests
import time
import warnings
from typing import Dict, List, Optional, Union

import backoff

from ai_scientist.tools.base_tool import BaseTool


def on_backoff(details: Dict) -> None:
    """
    重试等待时的回调函数。

    打印重试信息，包括等待时间和尝试次数。

    Args:
        details (Dict): 包含重试详情的字典（wait, tries, target）。
    """
    print(
        f"Backing off {details['wait']:0.1f} seconds after {details['tries']} tries "
        f"calling function {details['target'].__name__} at {time.strftime('%X')}"
    )


class SemanticScholarSearchTool(BaseTool):
    """
    用于在 Semantic Scholar 上搜索相关文献的工具。

    Attributes:
        max_results (int): 每次搜索返回的最大结果数。默认为 10。
        S2_API_KEY (str): Semantic Scholar API 密钥，从环境变量 S2_API_KEY 获取。
    """
    def __init__(
        self,
        name: str = "SearchSemanticScholar",
        description: str = (
            "Search for relevant literature using Semantic Scholar. "
            "Provide a search query to find relevant papers."
        ),
        max_results: int = 10,
    ):
        parameters = [
            {
                "name": "query",
                "type": "str",
                "description": "The search query to find relevant papers.",
            }
        ]
        super().__init__(name, description, parameters)
        self.max_results = max_results
        self.S2_API_KEY = os.getenv("S2_API_KEY")
        if not self.S2_API_KEY:
            warnings.warn(
                "No Semantic Scholar API key found. Requests will be subject to stricter rate limits. "
                "Set the S2_API_KEY environment variable for higher limits."
            )

    def use_tool(self, query: str) -> Optional[str]:
        """
        执行搜索并返回格式化后的结果字符串。

        Args:
            query (str): 搜索查询字符串。

        Returns:
            Optional[str]: 格式化的论文列表字符串，如果没有找到论文则返回 "No papers found."。
        """
        papers = self.search_for_papers(query)
        if papers:
            return self.format_papers(papers)
        else:
            return "No papers found."

    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.HTTPError, requests.exceptions.ConnectionError),
        on_backoff=on_backoff,
    )
    def search_for_papers(self, query: str) -> Optional[List[Dict]]:
        """
        发送 HTTP 请求到 Semantic Scholar API 搜索论文。

        包含自动重试机制（backoff）。

        Args:
            query (str): 搜索查询字符串。

        Returns:
            Optional[List[Dict]]: 论文信息字典列表，按引用次数降序排列。如果未找到或出错返回 None。
        """
        if not query:
            return None
        
        headers = {}
        if self.S2_API_KEY:
            headers["X-API-KEY"] = self.S2_API_KEY
        
        rsp = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            headers=headers,
            params={
                "query": query,
                "limit": self.max_results,
                "fields": "title,authors,venue,year,abstract,citationCount",
            },
        )
        print(f"Response Status Code: {rsp.status_code}")
        print(f"Response Content: {rsp.text[:500]}")
        rsp.raise_for_status()
        results = rsp.json()
        total = results.get("total", 0)
        if total == 0:
            return None

        papers = results.get("data", [])
        # Sort papers by citationCount in descending order
        papers.sort(key=lambda x: x.get("citationCount", 0), reverse=True)
        return papers

    def format_papers(self, papers: List[Dict]) -> str:
        """
        将论文列表格式化为易读的字符串。

        Args:
            papers (List[Dict]): 论文信息字典列表。

        Returns:
            str: 格式化后的字符串。
        """
        paper_strings = []
        for i, paper in enumerate(papers):
            authors = ", ".join(
                [author.get("name", "Unknown") for author in paper.get("authors", [])]
            )
            paper_strings.append(
                f"""{i + 1}: {paper.get("title", "Unknown Title")}. {authors}. {paper.get("venue", "Unknown Venue")}, {paper.get("year", "Unknown Year")}.
Number of citations: {paper.get("citationCount", "N/A")}
Abstract: {paper.get("abstract", "No abstract available.")}"""
            )
        return "\n\n".join(paper_strings)


@backoff.on_exception(
    backoff.expo, requests.exceptions.HTTPError, on_backoff=on_backoff
)
def search_for_papers(query, result_limit=10) -> Union[None, List[Dict]]:
    """
    独立的搜索函数（非 Tool 类方法）。

    Args:
        query (str): 搜索查询字符串。
        result_limit (int): 最大返回结果数。默认为 10。

    Returns:
        Union[None, List[Dict]]: 论文信息字典列表。
    """
    S2_API_KEY = os.getenv("S2_API_KEY")
    headers = {}
    if not S2_API_KEY:
        warnings.warn(
            "No Semantic Scholar API key found. Requests will be subject to stricter rate limits."
        )
    else:
        headers["X-API-KEY"] = S2_API_KEY
    
    if not query:
        return None
    
    rsp = requests.get(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        headers=headers,
        params={
            "query": query,
            "limit": result_limit,
            "fields": "title,authors,venue,year,abstract,citationStyles,citationCount",
        },
    )
    print(f"Response Status Code: {rsp.status_code}")
    print(
        f"Response Content: {rsp.text[:500]}"
    )  # Print the first 500 characters of the response content
    rsp.raise_for_status()
    results = rsp.json()
    total = results["total"]
    time.sleep(1.0)
    if not total:
        return None

    papers = results["data"]
    return papers
