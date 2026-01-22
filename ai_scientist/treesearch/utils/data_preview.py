"""
数据预览工具模块
================

本模块提供了用于生成文件和目录预览的功能。
支持计算文件大小、生成目录树结构以及预览 CSV 文件内容。
帮助 Agent 快速了解文件系统结构和数据内容。

主要功能：
1. get_file_len_size: 计算文件行数或字节大小。
2. file_tree: 生成指定目录的树状结构字符串。
3. preview_csv: 读取并预览 CSV 文件的前几行。

作者: AI Scientist Team
日期: 2025-01-22
"""

import json
import os
from pathlib import Path

import humanize
import pandas as pd
from genson import SchemaBuilder
from pandas.api.types import is_numeric_dtype

# these files are treated as code (e.g. markdown wrapped)
code_files = {".py", ".sh", ".yaml", ".yml", ".md", ".html", ".xml", ".log", ".rst"}
# we treat these files as text (rather than binary) files
plaintext_files = {".txt", ".csv", ".json", ".tsv"} | code_files


def get_file_len_size(f: Path) -> tuple[int, str]:
    """
    计算文件大小。

    对于文本文件，计算行数；对于其他文件，计算字节数。
    同时返回人类可读的大小描述。

    Args:
        f (Path): 文件路径。

    Returns:
        tuple[int, str]: (大小数值, 人类可读的大小字符串)。
    """
    if f.suffix in plaintext_files:
        num_lines = sum(1 for _ in open(f))
        return num_lines, f"{num_lines} lines"
    else:
        s = f.stat().st_size
        return s, humanize.naturalsize(s)


def file_tree(path: Path, depth=0) -> str:
    """
    生成目录的文件树结构。

    递归地列出目录下的文件和子目录，限制显示的文件数量以保持简洁。

    Args:
        path (Path): 目录路径。
        depth (int, optional): 当前递归深度。默认为 0。

    Returns:
        str: 文件树结构的字符串表示。
    """
    result = []
    files = [p for p in Path(path).iterdir() if not p.is_dir()]
    dirs = [p for p in Path(path).iterdir() if p.is_dir()]
    max_n = 4 if len(files) > 30 else 8
    for p in sorted(files)[:max_n]:
        result.append(f"{' '*depth*4}{p.name} ({get_file_len_size(p)[1]})")
    if len(files) > max_n:
        result.append(f"{' '*depth*4}... and {len(files)-max_n} other files")

    for p in sorted(dirs):
        result.append(f"{' '*depth*4}{p.name}/")
        result.append(file_tree(p, depth + 1))

    return "\n".join(result)


def _walk(path: Path):
    """
    递归遍历目录。

    类似于 os.walk，但针对 pathlib.Path 对象。

    Args:
        path (Path): 要遍历的目录路径。

    Yields:
        Path: 遍历到的文件或目录路径。
    """
    for p in sorted(Path(path).iterdir()):
        if p.is_dir():
            yield from _walk(p)
            continue
        yield p


def preview_csv(p: Path, file_name: str, simple=True) -> str:
    """
    生成 CSV 文件的文本预览。

    读取 CSV 文件并生成关于其结构和内容的摘要信息。

    Args:
        p (Path): CSV 文件路径。
        file_name (str): 在预览中显示的文件名。
        simple (bool, optional): 是否使用简化版预览（仅列名）。默认为 True。

    Returns:
        str: 文本预览内容。
    """
    df = pd.read_csv(p)

    out = []

    out.append(f"-> {file_name} has {df.shape[0]} rows and {df.shape[1]} columns.")

    if simple:
        cols = df.columns.tolist()
        sel_cols = 15
        cols_str = ", ".join(cols[:sel_cols])
        res = f"The columns are: {cols_str}"
        if len(cols) > sel_cols:
            res += f"... and {len(cols)-sel_cols} more columns"
        out.append(res)
    else:
        out.append("Here is some information about the columns:")
        for col in sorted(df.columns):
            dtype = df[col].dtype
            name = f"{col} ({dtype})"

            nan_count = df[col].isnull().sum()

            if dtype == "bool":
                v = df[col][df[col].notnull()].mean()
                out.append(f"{name} is {v*100:.2f}% True, {100-v*100:.2f}% False")
            elif df[col].nunique() < 10:
                out.append(
                    f"{name} has {df[col].nunique()} unique values: {df[col].unique().tolist()}"
                )
            elif is_numeric_dtype(df[col]):
                out.append(
                    f"{name} has range: {df[col].min():.2f} - {df[col].max():.2f}, {nan_count} nan values"
                )
            elif dtype == "object":
                out.append(
                    f"{name} has {df[col].nunique()} unique values. Some example values: {df[col].value_counts().head(4).index.tolist()}"
                )

    return "\n".join(out)


def preview_json(p: Path, file_name: str):
    """Generate a textual preview of a json file using a generated json schema"""
    builder = SchemaBuilder()
    with open(p) as f:
        builder.add_object(json.load(f))
    return f"-> {file_name} has auto-generated json schema:\n" + builder.to_json(
        indent=2
    )


def generate(base_path, include_file_details=True, simple=False):
    """
    Generate a textual preview of a directory, including an overview of the directory
    structure and previews of individual files
    """
    tree = f"```\n{file_tree(base_path)}```"
    out = [tree]

    if include_file_details:
        for fn in _walk(base_path):
            file_name = str(fn.relative_to(base_path))

            if fn.suffix == ".csv":
                out.append(preview_csv(fn, file_name, simple=simple))
            elif fn.suffix == ".json":
                out.append(preview_json(fn, file_name))
            elif fn.suffix in plaintext_files:
                if get_file_len_size(fn)[0] < 30:
                    with open(fn) as f:
                        content = f.read()
                        if fn.suffix in code_files:
                            content = f"```\n{content}\n```"
                        out.append(f"-> {file_name} has content:\n\n{content}")

    result = "\n\n".join(out)

    # if the result is very long we generate a simpler version
    if len(result) > 6_000 and not simple:
        return generate(
            base_path, include_file_details=include_file_details, simple=True
        )

    return result
