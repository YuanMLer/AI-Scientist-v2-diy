"""
TreeSearch 工具函数模块
=======================

提供文件系统操作和数据集预处理的通用工具函数。
主要用于复制目录、清理数据集（移除 macOS 临时文件）、解压归档文件等。

主要功能：
1. copytree: 增强版的目录复制函数，支持合并目录和符号链接。
2. clean_up_dataset: 清理数据集中的无关文件（如 __MACOSX, .DS_Store）。
3. extract_archives: 递归解压目录中的所有 .zip 文件并清理。
4. preproc_data: 数据预处理流程（解压 + 清理）。

作者: AI Scientist Team
日期: 2025-01-22
"""
import logging
import shutil
import zipfile
from pathlib import Path

logger = logging.getLogger("ai-scientist")


def copytree(src: Path, dst: Path, use_symlinks=True):
    """
    Copy contents of `src` to `dst`. Unlike shutil.copytree, the dst dir can exist and will be merged.
    If src is a file, only that file will be copied. Optionally uses symlinks instead of copying.

    Args:
        src (Path): source directory
        dst (Path): destination directory
    """
    assert dst.is_dir()

    if src.is_file():
        dest_f = dst / src.name
        assert not dest_f.exists(), dest_f
        if use_symlinks:
            (dest_f).symlink_to(src)
        else:
            shutil.copyfile(src, dest_f)
        return

    for f in src.iterdir():
        dest_f = dst / f.name
        assert not dest_f.exists(), dest_f
        if use_symlinks:
            (dest_f).symlink_to(f)
        elif f.is_dir():
            shutil.copytree(f, dest_f)
        else:
            shutil.copyfile(f, dest_f)


def clean_up_dataset(path: Path):
    for item in path.rglob("__MACOSX"):
        if item.is_dir():
            shutil.rmtree(item)
    for item in path.rglob(".DS_Store"):
        if item.is_file():
            item.unlink()


def extract_archives(path: Path):
    """
    解压路径下所有的 .zip 文件并清理任务目录。

    遍历目录中的 zip 文件进行解压，处理嵌套目录结构，并删除原始 zip 文件。
    如果目标目录已存在，则跳过解压。

    [TODO] 处理嵌套的 zip 文件。

    Args:
        path (Path): 要处理的目录路径。
    """
    for zip_f in path.rglob("*.zip"):
        f_out_dir = zip_f.with_suffix("")

        # special case: the intended output path already exists (maybe data has already been extracted by user)
        if f_out_dir.exists():
            logger.debug(
                f"Skipping {zip_f} as an item with the same name already exists."
            )
            # if it's a file, it's probably exactly the same as in the zip -> remove the zip
            # [TODO] maybe add an extra check to see if zip file content matches the colliding file
            if f_out_dir.is_file() and f_out_dir.suffix != "":
                zip_f.unlink()
            continue

        logger.debug(f"Extracting: {zip_f}")
        f_out_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(zip_f, "r") as zip_ref:
            zip_ref.extractall(f_out_dir)

        # remove any unwanted files
        clean_up_dataset(f_out_dir)

        contents = list(f_out_dir.iterdir())

        # special case: the zip contains a single dir/file with the same name as the zip
        if len(contents) == 1 and contents[0].name == f_out_dir.name:
            sub_item = contents[0]
            # if it's a dir, move its contents to the parent and remove it
            if sub_item.is_dir():
                logger.debug(f"Special handling (child is dir) enabled for: {zip_f}")
                for f in sub_item.rglob("*"):
                    shutil.move(f, f_out_dir)
                sub_item.rmdir()
            # if it's a file, rename it to the parent and remove the parent
            elif sub_item.is_file():
                logger.debug(f"Special handling (child is file) enabled for: {zip_f}")
                sub_item_tmp = sub_item.rename(f_out_dir.with_suffix(".__tmp_rename"))
                f_out_dir.rmdir()
                sub_item_tmp.rename(f_out_dir)

        zip_f.unlink()


def preproc_data(path: Path):
    """
    数据预处理流程。

    依次执行解压归档文件和清理数据集操作。

    Args:
        path (Path): 要处理的数据目录路径。
    """
    extract_archives(path)
    clean_up_dataset(path)
