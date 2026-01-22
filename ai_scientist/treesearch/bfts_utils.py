"""
BFTS 实用工具模块
=================

本模块提供了支持 BFTS (Best-First Tree Search) 实验流程的辅助函数。
主要功能包括将想法数据转换为 Markdown 格式，以及根据实验想法动态生成配置文件。

功能包括：
1. `idea_to_markdown`: 将结构化的想法数据（字典）渲染为 Markdown 文件，便于阅读和 LLM 处理。
2. `edit_bfts_config_file`: 根据当前实验的想法路径，创建并修改专用的实验配置文件。

作者: AI Scientist Team
日期: 2025-01-22
"""

import os
import os.path as osp
import shutil
import yaml


def idea_to_markdown(data: dict, output_path: str, load_code: str) -> None:
    """
    将字典格式的想法数据转换为 Markdown 文件。

    该函数遍历字典中的键值对，将其格式化为 Markdown 的标题和列表/段落。
    如果提供了 `load_code` 路径，还会将代码内容作为代码块追加到文件末尾。

    Args:
        data (dict): 包含想法数据的字典。
        output_path (str): 输出 Markdown 文件的保存路径。
        load_code (str): (可选) 需要包含在 Markdown 中的代码文件路径。
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for key, value in data.items():
            # 将键名转换为标题格式 (如 "experiment_plan" -> "Experiment Plan") 并作为二级标题
            header = key.replace("_", " ").title()
            f.write(f"## {header}\n\n")

            # 根据值类型进行不同的格式化处理
            if isinstance(value, (list, tuple)):
                # 列表类型：转换为 Markdown 列表
                for item in value:
                    f.write(f"- {item}\n")
                f.write("\n")
            elif isinstance(value, dict):
                # 字典类型：子键作为三级标题
                for sub_key, sub_value in value.items():
                    f.write(f"### {sub_key}\n")
                    f.write(f"{sub_value}\n\n")
            else:
                # 其他类型（如字符串）：直接写入
                f.write(f"{value}\n\n")

        # 如果指定了代码文件，将其内容追加到 Markdown 中
        if load_code:
            # 确保代码文件存在
            assert os.path.exists(load_code), f"Code path at {load_code} must exist if using the 'load_code' flag. This is an optional code prompt that you may choose to include; if not, please do not set 'load_code'."
            f.write(f"## Code To Potentially Use\n\n")
            f.write(f"Use the following code as context for your experiments:\n\n")
            with open(load_code, "r") as code_file:
                code = code_file.read()
                f.write(f"```python\n{code}\n```\n\n")


def edit_bfts_config_file(config_path: str, idea_dir: str, idea_path: str) -> str:
    """
    编辑 BFTS 配置文件以指向特定的想法文件和工作目录。

    该函数会复制原始配置文件到实验目录下，并修改其中的关键路径配置，
    以便为当前实验运行创建一个独立的配置环境。

    Args:
        config_path (str): 原始 bfts_config.yaml 文件的路径。
        idea_dir (str): 想法所在的目录（也是实验的工作目录）。
        idea_path (str): 想法描述文件 (JSON/Markdown) 的路径。

    Returns:
        str: 新生成的配置文件的路径。
    """
    # 确定新配置文件的路径
    run_config_path = osp.join(idea_dir, "bfts_config.yaml")
    # 复制原始配置
    shutil.copy(config_path, run_config_path)
    
    # 加载配置内容
    with open(run_config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # 修改配置项
    config["desc_file"] = idea_path
    config["workspace_dir"] = idea_dir

    # 创建并设置数据目录
    data_dir = osp.join(idea_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    config["data_dir"] = data_dir

    # 创建并设置日志目录
    log_dir = osp.join(idea_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    config["log_dir"] = log_dir

    # 保存修改后的配置
    with open(run_config_path, "w") as f:
        yaml.dump(config, f)
    return run_config_path
