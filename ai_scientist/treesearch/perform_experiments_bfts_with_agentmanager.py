"""
BFTS 实验执行模块
=================

本模块负责使用 Best-First Tree Search (BFTS) 策略和 AgentManager 执行 AI 科学家实验。
它管理实验的生命周期，包括工作区准备、代理初始化、实时进度可视化、结果记录和报告生成。

主要功能：
1. perform_experiments_bfts: 执行实验的主入口函数。
2. journal_to_rich_tree: 将实验日志 (Journal) 转换为 Rich 树状结构用于可视化。
3. 实时监控: 使用 Rich 库展示实验进度、当前阶段和搜索树状态。
4. 结果汇总: 在实验结束后生成各阶段的总结报告。

作者: AI Scientist Team
日期: 2025-01-22
"""

import atexit
import logging
import shutil
import json
import pickle
from . import backend
from .journal import Journal, Node
from .journal2report import journal2report
from .interpreter import Interpreter
from rich.columns import Columns
from rich.console import Group
from rich.live import Live
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)
from rich.text import Text
from rich.status import Status
from rich.tree import Tree
from .utils.config import load_task_desc, prep_agent_workspace, save_run, load_cfg
from .agent_manager import AgentManager
from pathlib import Path
from .agent_manager import Stage
from .log_summarization import overall_summarize


logger = logging.getLogger("ai-scientist")


def journal_to_rich_tree(journal: Journal, cfg):
    """
    将 Journal 对象转换为 Rich Tree 对象，用于终端可视化。

    展示搜索树的结构，标记最佳节点和错误节点。

    Args:
        journal (Journal): 包含实验记录的 Journal 对象。
        cfg: 配置对象。

    Returns:
        Tree: Rich Tree 对象。
    """
    best_node = journal.get_best_node(cfg=cfg)

    def append_rec(node: Node, tree):
        if node.is_buggy:
            s = "[red]◍ bug"
        else:
            style = "bold " if node is best_node else ""

            if node is best_node:
                s = f"[{style}green]● {node.metric.value:.3f} (best)"
            else:
                s = f"[{style}green]● {node.metric.value:.3f}"

        subtree = tree.add(s)
        for child in node.children:
            append_rec(child, subtree)

    tree = Tree("[bold blue]Solution tree")
    for n in journal.draft_nodes:
        append_rec(n, tree)
    return tree


def perform_experiments_bfts(config_path: str):
    """
    执行 BFTS (Best-First Tree Search) 实验。

    流程：
    1. 加载配置和任务描述。
    2. 准备代理工作区（复制必要文件）。
    3. 初始化 AgentManager。
    4. 设置 Rich 实时监控界面。
    5. 运行实验循环 (manager.run)。
    6. 保存实验状态和结果。
    7. 生成最终总结报告。

    Args:
        config_path (str): 配置文件路径。
    """
    # 将配置路径字符串转换为 Path 对象
    config_path = Path(config_path)
    cfg = load_cfg(config_path)
    logger.info(f'Starting run "{cfg.exp_name}"')

    task_desc = load_task_desc(cfg)
    print(task_desc)
    task_desc_str = backend.compile_prompt_to_md(task_desc)

    global_step = 0

    # 准备代理工作区（复制模板代码和数据）
    with Status("Preparing agent workspace (copying and extracting files) ..."):
        prep_agent_workspace(cfg)

    def cleanup():
        # 如果第一步都未完成，则清理工作区
        if global_step == 0:
            shutil.rmtree(cfg.workspace_dir)

    atexit.register(cleanup)

    # Initialize Interpreter
    interpreter = Interpreter(
        working_dir=cfg.workspace_dir,
    )

    manager = AgentManager(
        task_desc=task_desc,
        cfg=cfg,
        workspace_dir=Path(cfg.workspace_dir),
    )

    # 设置进度条和状态显示
    prog = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=20),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
    )
    status = Status("[green]Running experiments...")
    prog.add_task("Progress:", total=cfg.agent.steps, completed=global_step)

    def create_exec_callback(status_obj):
        """创建代码执行回调函数，用于更新状态显示。"""
        def exec_callback(*args, **kwargs):
            status_obj.update("[magenta]Executing code...")
            res = interpreter.run(*args, **kwargs)
            status_obj.update("[green]Generating code...")
            return res

        return exec_callback

    def step_callback(stage, journal):
        """步骤完成回调函数，用于保存进度和生成中间摘要。"""
        print("Step complete")
        try:
            # 生成并保存当前步骤的笔记
            notes_dir = cfg.log_dir / f"stage_{stage.name}" / "notes"
            notes_dir.mkdir(parents=True, exist_ok=True)

            # 保存最新的节点摘要
            if journal.nodes:
                latest_node = journal.nodes[-1]
                if hasattr(latest_node, "_agent"):
                    summary = latest_node._agent._generate_node_summary(latest_node)
                    with open(
                        notes_dir / f"node_{latest_node.id}_summary.json", "w"
                    ) as f:
                        json.dump(summary, f, indent=2)


            # 生成当前阶段的发现总结
            if cfg.agent.get("summary", None) is not None:
                current_findings = journal.generate_summary(
                    include_code=False, 
                    **{
                        "model": cfg.agent.summary.model, 
                        "temp": cfg.agent.summary.temp
                    }
                )
            else:
                current_findings = journal.generate_summary(include_code=False)

            best_metric = journal.get_best_node(cfg=cfg)

            # 生成并保存阶段进度摘要
            stage_summary = {
                "stage": stage.name,
                "total_nodes": len(journal.nodes),
                "buggy_nodes": len(journal.buggy_nodes),
                "good_nodes": len(journal.good_nodes),
                "best_metric": (
                    str(best_metric.metric)
                    if best_metric
                    else "None"
                ),
                "current_findings": current_findings,
            }

            with open(notes_dir / "stage_progress.json", "w") as f:
                json.dump(stage_summary, f, indent=2)

            # 保存运行状态（Journal）
            save_run(cfg, journal, stage_name=f"stage_{stage.name}")

        except Exception as e:
            print(f"Error in step callback: {e}")

        print(f"Run saved at {cfg.log_dir / f'stage_{stage.name}'}")
        print(f"Step {len(journal)}/{stage.max_iterations} at stage_{stage.name}")
        print(f"Run saved at {cfg.log_dir / f'stage_{stage.name}'}")

    def generate_live(manager):
        """生成实时监控界面的布局。"""
        current_stage = manager.current_stage
        current_journal = manager.journals.get(
            current_stage.name if current_stage else None, None
        )

        if current_journal:
            tree = journal_to_rich_tree(current_journal, cfg)
        else:
            tree = Tree("[bold blue]No results yet")

        file_paths = [
            f"Result visualization:\n[yellow]▶ {str((cfg.log_dir / 'tree_plot.html'))}",
            f"Agent workspace directory:\n[yellow]▶ {str(cfg.workspace_dir)}",
            f"Experiment log directory:\n[yellow]▶ {str(cfg.log_dir)}",
        ]

        stage_info = [
            "[bold]Experiment Progress:",
            f"Current Stage: [cyan]{current_stage.name if current_stage else 'None'}[/cyan]",
            f"Completed Stages: [green]{', '.join(manager.completed_stages)}[/green]",
        ]

        left = Group(
            Panel(Text(task_desc_str.strip()), title="Task description"),
            Panel(Text("\n".join(stage_info)), title="Stage Progress"),
            prog,
            status,
        )
        right = tree
        wide = Group(*file_paths)

        return Panel(
            Group(
                Padding(wide, (1, 1, 1, 1)),
                Columns(
                    [Padding(left, (1, 2, 1, 1)), Padding(right, (1, 1, 1, 2))],
                    equal=True,
                ),
            ),
            title=f'[b]AIDE is working on experiment: [bold green]"{cfg.exp_name}[/b]"',
            subtitle="Press [b]Ctrl+C[/b] to stop the run",
        )

    # 启动 Rich 实时监控
    live = Live(
        generate_live(manager),
        refresh_per_second=16,
        screen=True,
    )

    # 启动代理管理器
    with live:
        manager.run(exec_callback=create_exec_callback(status), step_callback=step_callback)

    # 保存管理器状态
    manager_pickle_path = cfg.log_dir / "manager.pkl"
    try:
        with open(manager_pickle_path, "wb") as f:
            pickle.dump(manager, f)
        logger.info(f"Saved manager state to: {manager_pickle_path}")
    except Exception as e:
        logger.warning(f"Failed to save full manager state: {e}")
        try:
            with open(manager_pickle_path, "wb") as f:
                pickle.dump(manager.journals.items(), f)
            logger.info(f"Saved manager journals to: {manager_pickle_path}")
        except Exception as e:
            logger.error(f"Failed to save manager journals: {e}")

    # 生成最终报告
    if cfg.generate_report:
        print("Generating final report from all stages...")
        (
            draft_summary,
            baseline_summary,
            research_summary,
            ablation_summary,
        ) = overall_summarize(manager.journals.items(), cfg)
        draft_summary_path = cfg.log_dir / "draft_summary.json"
        baseline_summary_path = cfg.log_dir / "baseline_summary.json"
        research_summary_path = cfg.log_dir / "research_summary.json"
        ablation_summary_path = cfg.log_dir / "ablation_summary.json"

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


if __name__ == "__main__":
    cfg_path = "treesearch/utils/config.yaml"
    cfg = load_cfg(cfg_path)
    perform_experiments_bfts(cfg_path)