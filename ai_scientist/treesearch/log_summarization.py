"""
日志摘要模块
============

本模块负责对实验日志进行总结和分析。
利用 LLM 对不同阶段（Draft, Baseline, Research, Ablation）的实验结果进行聚合、对比和总结，
生成具有科学价值的见解报告。

主要功能：
- 节点信息提取：从 Journal 节点中提取关键信息（计划、分析、指标、图表等）。
- 阶段摘要生成：使用 LLM 生成特定阶段的实验摘要。
- 历史记录标注：为节点标注其父节点的计划和当前计划的综合摘要。
- 整体摘要生成：并行处理所有阶段的日志，生成最终的实验报告。

作者: AI Scientist Team
日期: 2025-01-22
"""

import json
import os
import sys

from .journal import Node, Journal

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, parent_dir)
from ai_scientist.llm import get_response_from_llm, extract_json_between_markers
from ai_scientist.treesearch.backend import get_ai_client


# 报告摘要生成器的系统消息
report_summarizer_sys_msg = """You are an expert machine learning researcher.
You are given multiple experiment logs, each representing a node in a stage of exploring scientific ideas and implementations.
Your task is to aggregate these logs and provide scientifically insightful information.

Important instructions:
- Do NOT hallucinate or fabricate information that is not present in the logs.
- Do NOT introduce errors when repeating information from the logs.
- Identify notable insights or differences across the nodes without repeating the same information.
"""

# 输出格式控制指令
output_format_control = """Respond in the following format:

THOUGHT:
<THOUGHT>

JSON:
```json
<JSON>
```

In <THOUGHT>, thoroughly reason as an expert researcher. First, reason about each node, and then reason carefully by combining all the information. It is okay to be very detailed.

In <JSON>, provide the review in JSON format with the following fields in exactly this order:
- "Experiment_description": a string describing the conducted experiments
- "Significance": a string explaining why these experiments are important and what impact their findings might have
- "Description": a string describing the methods, steps taken, and any pertinent context needed to understand the experiments
- "List_of_included_plots": a list of plots that should be included. Each entry should include:
  • "path" (the plot path)
  • "description" (its original description)
  • "analysis" (your analysis of its scientific insights)
- "Key_numerical_results": a list of all important numerical results. Be selective about results that contribute to scientific insights. Each entry should include:
  • "result" (float number)
  • "description" (your short description of the result)
  • "analysis" (your analysis of its scientific insights)

Ensure the JSON is valid and properly formatted, as it will be automatically parsed."""

# 报告摘要生成提示词模板
report_summarizer_prompt = (
    """You are given multiple experiment logs from different "nodes". Each node represents attempts and experiments exploring various scientific ideas.

One key point is that these nodes collectively illustrate a stage of testing different methods or approaches. The crucial task is to identify the scientific insights gleaned from this stage. For example, if one node tries method A and another node tries method B, you should compare any observed differences in performance or outcomes. Summarize both experiments in "Experiment_description", explain the processes in "Description", and place any key numerical findings (such as accuracy metrics, loss values, or runtime comparisons) in "Key_numerical_results."

Be concise and avoid repeating the same information from different nodes. You are encouraged to be thorough, but you do not need to include information from every node. Reason carefully about which results from which nodes are scientifically insightful.

The name of this stage of the experiment: {stage_name}

Here are the experiment logs of the nodes:

{node_infos}
"""
    + output_format_control
)

# 阶段聚合提示词模板
stage_aggregate_prompt = """You are given:

1) The summary of all previous experiment stages:
{prev_summary}

2) The name of the current experiment stage:
{stage_name}

3) The summary of the current stage:
{current_summary}


Your task is to produce an **updated comprehensive summary** of all experiment stages, including the newly introduced results from the current stage.

**Key Requirements:**
1. **No Loss of Critical Information**
   - Preserve valuable insights from the summary of all previous experiment stages. Do not remove or alter crucial texts.
   - Absolutely no hallucinations: if something does not appear in the logs or summaries, do not invent it. If something appears in the previous summary, do not make any mistakes when repeating it.
2. **Merge New Stage Data**
   - Integrate relevant results from the current stage into the existing summary.
   - Identify any overlap or repetition between new and old content, and remove only that which is clearly redundant or no longer scientifically insightful.
   - Be very careful if you want to remove or shorten the old content. By default, you can keep most of it and append new text.
   - Highlight how new findings connect to or differ from previous findings.
3. **Numerical Results and Visuals**
   - Carefully maintain the most insightful plots, figures, and numerical results.
   - Do not delete crucial quantitative findings or meaningful visual references.
4. **Length and Format**
   - The final summary will likely be **very long**. That is acceptable.
   - Present the updated summary in a format consistent with the style of the previous summaries (e.g., same section headings or structure).

Respond in the following format:

THOUGHT:
<THOUGHT>

JSON:
```json
<JSON>
```
Ensure the JSON is valid and properly formatted, as it will be automatically parsed.
"""


def get_nodes_infos(nodes):
    """
    获取节点的详细信息字符串。

    格式化节点的主要信息，包括计划、分析、数值结果和绘图分析，用于构建 LLM 提示词。

    Args:
        nodes (List[Node]): 节点列表。

    Returns:
        str: 格式化的节点信息字符串。
    """
    node_infos = ""
    for n in nodes:
        node_info = f"Node ID: {n.id}\n"
        node_info += (
            f"Plan: {n.overall_plan}\n"
            if hasattr(n, "overall_plan")
            else "Plan: Not available\n"
        )
        node_info += (
            f"Analysis: {n.analysis}\n"
            if hasattr(n, "analysis")
            else "Analysis: Not available\n"
        )
        node_info += (
            f"Numerical Results: {n.metric}\n"
            if hasattr(n, "metric")
            else "Numerical Results: Not available\n"
        )
        node_info += "Plot Analyses:\n"
        if hasattr(n, "plot_analyses") and n.plot_analyses:
            for plot in n.plot_analyses:
                node_info += f"- Plot Path: {plot.get('plot_path', 'Not available')}, Description: {plot.get('analysis', 'Not available')}\n"
        else:
            node_info += "No plot analyses available\n"
        node_infos += node_info + "\n"
    return node_infos


def get_summarizer_prompt(journal, stage_name):
    """
    生成摘要生成器的提示词。

    选择好的叶子节点，并使用 get_nodes_infos 格式化其信息，构建完整的提示词。

    Args:
        journal (Journal): 实验日志。
        stage_name (str): 阶段名称。

    Returns:
        tuple: (系统消息, 完整提示词)。
    """
    good_leaf_nodes = [n for n in journal.good_nodes if n.is_leaf]
    if not good_leaf_nodes:
        print("NO GOOD LEAF NODES!!!")
        good_leaf_nodes = [n for n in journal.good_nodes]
    node_infos = get_nodes_infos(good_leaf_nodes)
    return report_summarizer_sys_msg, report_summarizer_prompt.format(
        node_infos=node_infos, stage_name=stage_name
    )


def get_stage_summary(journal, stage_name, model, client):
    """
    获取指定阶段的实验摘要。

    调用 LLM 生成摘要并解析 JSON 结果。

    Args:
        journal (Journal): 实验日志。
        stage_name (str): 阶段名称。
        model (str): 使用的 LLM 模型名称。
        client: LLM 客户端实例。

    Returns:
        dict: 解析后的摘要 JSON 对象。
    """
    sys_msg, prompt = get_summarizer_prompt(journal, stage_name)
    response = get_response_from_llm(prompt, client, model, sys_msg)
    summary_json = extract_json_between_markers(response[0])
    return summary_json


def get_node_log(node):
    """
    获取节点的日志信息字典。

    提取节点中与日志记录和分析相关的关键字段，处理结果目录路径。

    Args:
        node (Node): 实验节点。

    Returns:
        dict: 包含关键信息的字典。
    """
    node_dict = node.to_dict()
    # Only include keys that are relevant for logging/analysis
    keys_to_include = [
        "overall_plan",
        "analysis",
        "metric",
        "code",
        "plot_code",
        "plot_plan",
        "plot_analyses",
        "plot_paths",
        "vlm_feedback_summary",
        "exp_results_dir",
        "ablation_name",
    ]
    ret = {
        key: node_dict[key]
        for key in keys_to_include
        if key in node_dict and node_dict[key] is not None
    }
    if "exp_results_dir" in ret:
        original_dir_path = ret["exp_results_dir"]
        # Remove leading path segments before "experiment_results"
        idx = original_dir_path.find("experiment_results")
        short_dir_path = original_dir_path
        if idx != -1:
            short_dir_path = original_dir_path[idx:]

        ret["exp_results_dir"] = short_dir_path

        if os.path.isdir(original_dir_path):
            npy_files = [f for f in os.listdir(original_dir_path) if f.endswith(".npy")]
            # Prepend the shortened path to each .npy filename
            ret["exp_results_npy_files"] = [
                os.path.join(short_dir_path, f) for f in npy_files
            ]
        else:
            ret["exp_results_npy_files"] = []
    return ret


def update_summary(
    prev_summary, cur_stage_name, cur_journal, cur_summary, model, client, max_retry=5
):
    """
    更新实验总体摘要。

    将当前阶段的摘要与之前的摘要合并，生成更新后的综合摘要。

    Args:
        prev_summary (str): 之前的总体摘要。
        cur_stage_name (str): 当前阶段名称。
        cur_journal (Journal): 当前阶段的实验日志。
        cur_summary (str): 当前阶段的摘要。
        model (str): 使用的 LLM 模型名称。
        client: LLM 客户端实例。
        max_retry (int, optional): 最大重试次数。默认为 5。

    Returns:
        dict: 更新后的摘要 JSON 对象。

    Raises:
        Exception: 如果在多次重试后仍无法生成摘要。
    """
    good_leaf_nodes = [n for n in cur_journal.good_nodes if n.is_leaf]
    node_infos = get_nodes_infos(good_leaf_nodes)
    prompt = stage_aggregate_prompt.format(
        prev_summary=prev_summary,
        stage_name=cur_stage_name,
        current_summary=cur_summary,
    )
    try:
        response = get_response_from_llm(
            prompt, client, model, "You are an expert machine learning researcher."
        )
        summary_json = extract_json_between_markers(response[0])
        assert summary_json
    except Exception as e:
        if max_retry > 0:
            print(f"Error occurred: {e}. Retrying... ({max_retry} attempts left)")
            return update_summary(
                prev_summary,
                cur_stage_name,
                cur_journal,
                cur_summary,
                model,
                client,
                max_retry - 1,
            )
        else:
            print(f"Failed to update summary after multiple attempts. Error: {e}")
            raise
    return summary_json


# 总体计划摘要提示词模板
overall_plan_summarizer_prompt = """You have been provided with the plans for both the parent node and the current node. Your task is to synthesize a comprehensive summary of the overall plan by integrating details from both the parent and current node plans.
The summary should be thorough and clearly articulate the underlying motivations.
For example, if in your previous overall plan you were experimenting with a new idea, and now your current plan is to fix certain bugs in the previous implementation, your returned overall plan should focus on your previous overall plan, and briefly mention that the current plan includes bug fixes. If your current plan is more about implementing new ideas, then you should summarize that thoroughly along with the previous overall plan.
The goal is to create a comprehensive summary of all historical plans, focusing on the main scientific planning and objectives.

Previous overall plan:
{prev_overall_plan}

Current plan:
{current_plan}

Respond in the following format:

THOUGHT:
<THOUGHT>

JSON:
```json
<JSON>
```

In <THOUGHT>, thoroughly reason as an expert researcher. First, reason over each node, and then carefully combine all information. It is okay to be very detailed.

In <JSON>, provide the review in JSON format with the following field in exactly this order:
- "overall_plan": a string that describes the overall plan based on the current and previous overall plans

Ensure the JSON is valid and properly formatted, as it will be automatically parsed.
"""


def annotate_history(journal, cfg=None):
    """
    为日志中的节点标注历史计划摘要。

    遍历日志中的节点，使用 LLM 将其计划与父节点的计划进行综合，生成 overall_plan。

    Args:
        journal (Journal): 实验日志。
        cfg (Config, optional): 配置对象。
    """
    for node in journal.nodes:
        if node.parent:
            max_retries = 3
            retry_count = 0
            while retry_count < max_retries:
                try:
                    if cfg.agent.get("summary", None) is not None:
                        model = cfg.agent.summary.model
                    else:
                        model = "gpt-4o-2024-08-06"
                    client = get_ai_client(model)
                    response = get_response_from_llm(
                        overall_plan_summarizer_prompt.format(
                            prev_overall_plan=node.parent.overall_plan,
                            current_plan=node.plan,
                        ),
                        client,
                        model,
                        report_summarizer_sys_msg,
                    )
                    node.overall_plan = extract_json_between_markers(response[0])[
                        "overall_plan"
                    ]
                    break
                except Exception as e:
                    retry_count += 1
                    if retry_count == max_retries:
                        print(f"Failed after {max_retries} attempts. Error: {e}")
                        raise
                    print(
                        f"Error occurred: {e}. Retrying... ({max_retries - retry_count} attempts left)"
                    )
        else:
            node.overall_plan = node.plan


def overall_summarize(journals, cfg=None):
    """
    生成所有实验阶段的综合摘要。

    并行处理每个阶段的日志，生成 Draft, Baseline, Research, Ablation 四个部分的摘要。

    Args:
        journals (List[Tuple[str, Journal]]): 包含 (阶段名, 日志) 元组的列表。
        cfg (Config, optional): 配置对象。

    Returns:
        tuple: (draft_summary, baseline_summary, research_summary, ablation_summary)
    """
    from concurrent.futures import ThreadPoolExecutor

    def process_stage(idx, stage_tuple):
        """
        处理单个阶段的日志。
        """
        stage_name, journal = stage_tuple
        annotate_history(journal, cfg=cfg)
        if idx in [1, 2]:
            best_node = journal.get_best_node(cfg=cfg)
            # get multi-seed results and aggregater node
            child_nodes = best_node.children
            multi_seed_nodes = [
                n for n in child_nodes if n.is_seed_node and not n.is_seed_agg_node
            ]
            agg_node = None
            for n in child_nodes:
                if n.is_seed_node and n.is_seed_agg_node:
                    agg_node = n
                    break
            if agg_node is None:
                # skip agg node
                return {
                    "best node": get_node_log(best_node),
                    "best node with different seeds": [
                        get_node_log(n) for n in multi_seed_nodes
                    ],
                }
            else:
                return {
                    "best node": get_node_log(best_node),
                    "best node with different seeds": [
                        get_node_log(n) for n in multi_seed_nodes
                    ],
                    "aggregated results of nodes with different seeds": get_node_log(
                        agg_node
                    ),
                }
        elif idx == 3:
            good_leaf_nodes = [
                n for n in journal.good_nodes if n.is_leaf and n.ablation_name
            ]
            return [get_node_log(n) for n in good_leaf_nodes]
        elif idx == 0:
            if cfg.agent.get("summary", None) is not None:
                model = cfg.agent.summary.get("model", "")
            else:
                model = "gpt-4o-2024-08-06"
            client = get_ai_client(model)
            summary_json = get_stage_summary(journal, stage_name, model, client)
            return summary_json

    from tqdm import tqdm

    with ThreadPoolExecutor() as executor:
        results = list(
            tqdm(
                executor.map(process_stage, range(len(list(journals))), journals),
                desc="Processing stages",
                total=len(list(journals)),
            )
        )
        draft_summary, baseline_summary, research_summary, ablation_summary = results

    return draft_summary, baseline_summary, research_summary, ablation_summary


if __name__ == "__main__":
    # Test
    example_path = "logs/247-run"

    def load_stage_folders(base_path):
        """
        加载以 'stage_' 开头并后跟数字的文件夹。

        Args:
            base_path (str): 阶段文件夹所在的基础目录路径。

        Returns:
            list: 排序后的阶段文件夹路径列表。
        """
        stage_folders = []
        for folder_name in os.listdir(base_path):
            if folder_name.startswith("stage_"):
                stage_folders.append(os.path.join(base_path, folder_name))
        return sorted(stage_folders, key=lambda x: int(x.split("_")[1]))

    def reconstruct_journal(journal_data):
        """
        根据 JSON 数据重构 Journal 对象。

        恢复节点之间的父子关系。

        Args:
            journal_data (dict): 包含 'nodes' 和 'node2parent' 的字典数据。

        Returns:
            Journal: 重构后的 Journal 对象。
        """
        # Create a mapping of node IDs to Node instances
        id_to_node = {}
        for node_data in journal_data["nodes"]:
            # Remove unused or invalid keys if needed
            if "actionable_insights_from_plots" in node_data:
                del node_data["actionable_insights_from_plots"]
            node = Node.from_dict(node_data)
            id_to_node[node.id] = node

        # Set up parent-child relationships using node2parent
        for node_id, parent_id in journal_data["node2parent"].items():
            child_node = id_to_node[node_id]
            parent_node = id_to_node[parent_id]
            child_node.parent = parent_node
            parent_node.children.add(child_node)

        # Create a Journal and add all nodes
        journal = Journal()
        journal.nodes.extend(id_to_node.values())

        return journal

    # Example usage
    stage_folders = load_stage_folders(example_path)
    journals = []
    for index, folder in enumerate(stage_folders, start=1):
        print(f"Stage {index}: {folder}")
        stage_name = os.path.basename(folder)
        journal_path = os.path.join(folder, "journal.json")
        if os.path.exists(journal_path):
            with open(journal_path, "r") as file:
                journal_data = json.load(file)
                print(f"Loaded journal.json for Stage {index}")
        else:
            print(f"No journal.json found for Stage {index}")
        journal = reconstruct_journal(journal_data)
        journals.append((stage_name, journal))

    # Convert manager journals to list of (stage_name, journal) tuples
    (
        draft_summary,
        baseline_summary,
        research_summary,
        ablation_summary,
    ) = overall_summarize(journals)
    log_dir = "logs/247-run"
    draft_summary_path = log_dir + "/draft_summary.json"
    baseline_summary_path = log_dir + "/baseline_summary.json"
    research_summary_path = log_dir + "/research_summary.json"
    ablation_summary_path = log_dir + "/ablation_summary.json"

    with open(draft_summary_path, "w") as draft_file:
        json.dump(draft_summary, draft_file, indent=2)

    with open(baseline_summary_path, "w") as baseline_file:
        json.dump(baseline_summary, baseline_file, indent=2)

    with open(research_summary_path, "w") as research_file:
        json.dump(research_summary, research_file, indent=2)

    with open(ablation_summary_path, "w") as ablation_file:
        json.dump(ablation_summary, ablation_file, indent=2)

    print(f"Summary reports written to files:")
    print(f"- Draft summary: {draft_summary_path}")
    print(f"- Baseline summary: {baseline_summary_path}")
    print(f"- Research summary: {research_summary_path}")
    print(f"- Ablation summary: {ablation_summary_path}")
