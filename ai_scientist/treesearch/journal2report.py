"""
实验报告生成模块
================

本模块负责将实验日志（Journal）转换为 Markdown 格式的技术报告。
利用 LLM 对实验过程、设计尝试及其结果进行总结和分析。

主要功能：
1. journal2report: 生成包含引言、预处理、方法、结果讨论和未来工作的实验报告。

作者: AI Scientist Team
日期: 2025-01-22
"""

from .backend import query
from .journal import Journal
from .utils.config import StageConfig


def journal2report(journal: Journal, task_desc: dict, rcfg: StageConfig):
    """
    从实验日志生成 Markdown 格式的技术报告。

    整理实验日志摘要和任务描述，构建 Prompt 调用 LLM 生成报告。

    Args:
        journal (Journal): 包含实验记录的日志对象。
        task_desc (dict): 任务描述信息（包含研究目标等）。
        rcfg (StageConfig): 报告生成阶段的配置（包含模型、温度等）。

    Returns:
        tuple: LLM 查询的返回结果 (OutputType, req_time, in_tokens, out_tokens, info)。
               Output content 通常为 Markdown 格式的字符串。
    """
    report_input = journal.generate_summary(include_code=True)
    system_prompt_dict = {
        "Role": "You are a research assistant that always uses concise language.",
        "Goal": "The goal is to write a technical report summarising the empirical findings and technical decisions.",
        "Input": "You are given a raw research journal with list of design attempts and their outcomes, and a research idea description.",
        "Output": [
            "Your output should be a single markdown document.",
            "Your report should have the following sections: Introduction, Preprocessing, Methods, Results Discussion, Future Work",
            "You can include subsections if needed.",
        ],
    }
    context_prompt = (
        f"Here is the research journal of the agent: <journal>{report_input}<\\journal>, "
        f"and the research idea description is: <research_proposal>{task_desc}<\\research_proposal>."
    )
    return query(
        system_message=system_prompt_dict,
        user_message=context_prompt,
        model=rcfg.model,
        temperature=rcfg.temp,
        max_tokens=4096,
    )
