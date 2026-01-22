"""
配置与设置工具模块
==================

本模块定义了实验的配置结构（使用 Dataclasses）和相关工具函数。
主要功能包括：
1. 定义实验各阶段、Agent、搜索策略等的配置类。
2. 加载和合并配置文件（YAML）与命令行参数。
3. 准备实验目录（日志、工作区）。
4. 加载任务描述。
5. 准备 Agent 工作环境（复制数据、预处理）。
6. 保存实验运行状态（日志、配置、可视化树、最佳代码）。

作者: AI Scientist Team
日期: 2025-01-22
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Hashable, cast, Literal, Optional

import coolname
import rich
from omegaconf import OmegaConf
from rich.syntax import Syntax
import shutup
from rich.logging import RichHandler
import logging

from . import tree_export
from . import copytree, preproc_data, serialize
from ai_scientist.config_loader import load_config as loader_load_config

shutup.mute_warnings()
logging.basicConfig(
    level="WARNING", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)
logger = logging.getLogger("ai-scientist")
logger.setLevel(logging.WARNING)


""" these dataclasses are just for type hinting, the actual config is in config.yaml """


@dataclass
class ThinkingConfig:
    """
    思考过程配置。

    Args:
        type (str): 思考类型。
        budget_tokens (int, optional): 思考过程的 token 预算。
    """
    type: str
    budget_tokens: Optional[int] = None


@dataclass
class StageConfig:
    """
    实验阶段配置。

    Args:
        model (str): 使用的模型名称。
        temp (float): 温度参数。
        thinking (ThinkingConfig): 思考过程配置。
        betas (str): Beta 参数（用于特定采样策略）。
        max_tokens (int, optional): 最大 token 数。
    """
    model: str
    temp: float
    thinking: ThinkingConfig
    betas: str
    max_tokens: Optional[int] = None


@dataclass
class SearchConfig:
    """
    搜索策略配置。

    Args:
        max_debug_depth (int): 最大调试深度。
        debug_prob (float): 触发调试的概率。
        num_drafts (int): 生成的草稿数量。
    """
    max_debug_depth: int
    debug_prob: float
    num_drafts: int


@dataclass
class DebugConfig:
    """
    调试配置。

    Args:
        stage4 (bool): 是否在阶段 4 启用调试。
    """
    stage4: bool


@dataclass
class AgentConfig:
    """
    Agent 配置。

    包含实验步骤、阶段定义、验证设置、反馈机制、搜索参数等。

    Args:
        steps (int): 总步数。
        stages (dict[str, int]): 各阶段及其持续时间。
        k_fold_validation (int): K 折验证的折数。
        expose_prediction (bool): 是否暴露预测结果。
        data_preview (bool): 是否预览数据。
        code (StageConfig): 代码生成阶段配置。
        feedback (StageConfig): 反馈阶段配置。
        vlm_feedback (StageConfig): VLM 反馈阶段配置。
        search (SearchConfig): 搜索配置。
        num_workers (int): 并行工作进程数。
        type (str): Agent 类型 ('parallel' 或 'sequential')。
        multi_seed_eval (dict[str, int]): 多种子评估配置。
        summary (Optional[StageConfig]): 摘要生成阶段配置。
        select_node (Optional[StageConfig]): 节点选择阶段配置。
    """
    steps: int
    stages: dict[str, int]
    k_fold_validation: int
    expose_prediction: bool
    data_preview: bool

    code: StageConfig
    feedback: StageConfig
    vlm_feedback: StageConfig

    search: SearchConfig
    num_workers: int
    type: str
    multi_seed_eval: dict[str, int]

    summary: Optional[StageConfig] = None
    select_node: Optional[StageConfig] = None

@dataclass
class ExecConfig:
    """
    执行配置。

    Args:
        timeout (int): 执行超时时间（秒）。
        agent_file_name (str): Agent 代码文件名。
        format_tb_ipython (bool): 是否使用 IPython 格式化 Traceback。
    """
    timeout: int
    agent_file_name: str
    format_tb_ipython: bool


@dataclass
class ExperimentConfig:
    """
    实验配置。

    Args:
        num_syn_datasets (int): 合成数据集数量。
    """
    num_syn_datasets: int


@dataclass
class Config(Hashable):
    """
    全局配置类。

    聚合所有子配置，并包含全局路径和任务设置。

    Args:
        data_dir (Path): 数据目录。
        desc_file (Path | None): 任务描述文件路径。
        goal (str | None): 任务目标（如果没有描述文件）。
        eval (str | None): 评估方法（如果没有描述文件）。
        log_dir (Path): 日志目录。
        workspace_dir (Path): 工作区目录。
        preprocess_data (bool): 是否预处理数据。
        copy_data (bool): 是否复制数据。
        exp_name (str): 实验名称。
        exec (ExecConfig): 执行配置。
        generate_report (bool): 是否生成报告。
        report (StageConfig): 报告生成阶段配置。
        agent (AgentConfig): Agent 配置。
        experiment (ExperimentConfig): 实验配置。
        debug (DebugConfig): 调试配置。
        llm_config (Optional[dict]): LLM 配置。
    """
    data_dir: Path
    desc_file: Path | None

    goal: str | None
    eval: str | None

    log_dir: Path
    workspace_dir: Path

    preprocess_data: bool
    copy_data: bool

    exp_name: str

    exec: ExecConfig
    generate_report: bool
    report: StageConfig
    agent: AgentConfig
    experiment: ExperimentConfig
    debug: DebugConfig
    llm_config: Optional[dict] = None


def _get_next_logindex(dir: Path) -> int:
    """
    获取日志目录的下一个可用索引。

    扫描目录下以数字开头的文件夹，返回最大索引加 1。

    Args:
        dir (Path): 目录路径。

    Returns:
        int: 下一个可用索引。
    """
    max_index = -1
    for p in dir.iterdir():
        try:
            if (current_index := int(p.name.split("-")[0])) > max_index:
                max_index = current_index
        except ValueError:
            pass
    print("max_index: ", max_index)
    return max_index + 1


def _load_cfg(
    path: Path = Path(__file__).parent / "config.yaml", use_cli_args=False
) -> Config:
    """
    加载配置。

    从 YAML 文件加载配置，并可选择合并命令行参数。

    Args:
        path (Path, optional): 配置文件路径。默认为当前目录下的 "config.yaml"。
        use_cli_args (bool, optional): 是否合并命令行参数。默认为 False。

    Returns:
        Config: 加载的配置对象。
    """
    cfg = OmegaConf.load(path)
    if use_cli_args:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())
    return cfg


def load_cfg(path: Path = Path(__file__).parent / "config.yaml") -> Config:
    """
    加载配置并进行预处理。

    加载 YAML 配置，合并命令行参数，并设置日志和工作区目录。

    Args:
        path (Path, optional): 配置文件路径。

    Returns:
        Config: 准备好的配置对象。
    """
    return prep_cfg(_load_cfg(path))


def prep_cfg(cfg: Config):
    """
    预处理配置对象。

    验证配置完整性，设置绝对路径，生成实验名称和目录结构。

    Args:
        cfg (Config): 原始配置对象。

    Returns:
        Config: 处理后的配置对象。

    Raises:
        ValueError: 如果缺少必要的配置项（如 data_dir, goal/desc_file）。
    """
    if cfg.data_dir is None:
        raise ValueError("`data_dir` must be provided.")

    if cfg.desc_file is None and cfg.goal is None:
        raise ValueError(
            "You must provide either a description of the task goal (`goal=...`) or a path to a plaintext file containing the description (`desc_file=...`)."
        )

    if cfg.data_dir.startswith("example_tasks/"):
        cfg.data_dir = Path(__file__).parent.parent / cfg.data_dir
    cfg.data_dir = Path(cfg.data_dir).resolve()

    if cfg.desc_file is not None:
        cfg.desc_file = Path(cfg.desc_file).resolve()

    top_log_dir = Path(cfg.log_dir).resolve()
    top_log_dir.mkdir(parents=True, exist_ok=True)

    top_workspace_dir = Path(cfg.workspace_dir).resolve()
    top_workspace_dir.mkdir(parents=True, exist_ok=True)

    # generate experiment name and prefix with consecutive index
    ind = max(_get_next_logindex(top_log_dir), _get_next_logindex(top_workspace_dir))
    cfg.exp_name = cfg.exp_name or coolname.generate_slug(3)
    cfg.exp_name = f"{ind}-{cfg.exp_name}"

    cfg.log_dir = (top_log_dir / cfg.exp_name).resolve()
    cfg.workspace_dir = (top_workspace_dir / cfg.exp_name).resolve()

    # validate the config
    cfg_schema: Config = OmegaConf.structured(Config)
    cfg = OmegaConf.merge(cfg_schema, cfg)

    if cfg.agent.type not in ["parallel", "sequential"]:
        raise ValueError("agent.type must be either 'parallel' or 'sequential'")

    return cast(Config, cfg)


def print_cfg(cfg: Config) -> None:
    """
    打印配置内容。

    使用 Rich 库以 YAML 格式高亮显示配置。

    Args:
        cfg (Config): 配置对象。
    """
    rich.print(Syntax(OmegaConf.to_yaml(cfg), "yaml", theme="paraiso-dark"))


def load_task_desc(cfg: Config):
    """
    加载任务描述。

    从文件读取或根据 goal 和 eval 参数生成任务描述。

    Args:
        cfg (Config): 配置对象。

    Returns:
        str | dict: 任务描述内容。
    """

    # either load the task description from a file
    if cfg.desc_file is not None:
        if not (cfg.goal is None and cfg.eval is None):
            logger.warning(
                "Ignoring goal and eval args because task description file is provided."
            )

        with open(cfg.desc_file) as f:
            return f.read()

    # or generate it from the goal and eval args
    if cfg.goal is None:
        raise ValueError(
            "`goal` (and optionally `eval`) must be provided if a task description file is not provided."
        )

    task_desc = {"Task goal": cfg.goal}
    if cfg.eval is not None:
        task_desc["Task evaluation"] = cfg.eval
    print(task_desc)
    return task_desc


def prep_agent_workspace(cfg: Config):
    """
    准备 Agent 工作区。

    创建输入和工作目录，复制数据，并可选地进行数据预处理。

    Args:
        cfg (Config): 配置对象。
    """
    (cfg.workspace_dir / "input").mkdir(parents=True, exist_ok=True)
    (cfg.workspace_dir / "working").mkdir(parents=True, exist_ok=True)

    copytree(cfg.data_dir, cfg.workspace_dir / "input", use_symlinks=not cfg.copy_data)
    if cfg.preprocess_data:
        preproc_data(cfg.workspace_dir / "input")


def save_run(cfg: Config, journal, stage_name: str = None):
    """
    保存实验运行状态。

    保存 Journal（日志）、Config（配置）、可视化树（Tree Plot）和最佳解决方案代码。

    Args:
        cfg (Config): 配置对象。
        journal (Journal): 实验日志对象。
        stage_name (str, optional): 阶段名称。如果未提供，默认为 "NoStageRun"。

    Raises:
        Exception: 保存过程中发生的任何错误。
    """
    if stage_name is None:
        stage_name = "NoStageRun"
    save_dir = cfg.log_dir / stage_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # save journal
    try:
        serialize.dump_json(journal, save_dir / "journal.json")
    except Exception as e:
        print(f"Error saving journal: {e}")
        raise
    # save config
    try:
        OmegaConf.save(config=cfg, f=save_dir / "config.yaml")
    except Exception as e:
        print(f"Error saving config: {e}")
        raise
    # create the tree + code visualization
    try:
        tree_export.generate(cfg, journal, save_dir / "tree_plot.html")
    except Exception as e:
        print(f"Error generating tree: {e}")
        raise
    # save the best found solution
    try:
        best_node = journal.get_best_node(only_good=False, cfg=cfg)
        if best_node is not None:
            for existing_file in save_dir.glob("best_solution_*.py"):
                existing_file.unlink()
            # Create new best solution file
            filename = f"best_solution_{best_node.id}.py"
            with open(save_dir / filename, "w") as f:
                f.write(best_node.code)
            # save best_node.id to a text file
            with open(save_dir / "best_node_id.txt", "w") as f:
                f.write(str(best_node.id))
        else:
            print("No best node found yet")
    except Exception as e:
        print(f"Error saving best solution: {e}")
