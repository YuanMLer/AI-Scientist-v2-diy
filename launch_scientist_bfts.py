"""
启动 AI 科学家 (BFTS 版本)
==========================

本脚本是 AI 科学家系统的主要入口点，专门用于使用最佳优先树搜索 (BFTS) 算法运行实验。
它协调整个科学发现过程，包括：
1. 加载和预处理想法 (Ideas)。
2. 配置实验环境。
3. 执行实验 (使用 AgentManager 和 BFTS)。
4. 汇总结果和图表。
5. 生成论文 (Writeup)。
6. 进行同行评审 (Review)。

主要功能：
- 解析命令行参数以控制实验行为。
- 管理 GPU 资源。
- 处理实验日志和结果存储。
- 调用各个子模块完成从想法到论文的全流程。

作者: AI Scientist Team
日期: 2025-01-22
"""

import os.path as osp
import json
import argparse
import shutil
import torch
import os
import re
import sys
from datetime import datetime
from ai_scientist.llm import create_client

from contextlib import contextmanager
from ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager import (
    perform_experiments_bfts,
)
from ai_scientist.treesearch.bfts_utils import (
    idea_to_markdown,
    edit_bfts_config_file,
)
from ai_scientist.perform_plotting import aggregate_plots
from ai_scientist.perform_writeup import perform_writeup
from ai_scientist.perform_icbinb_writeup import (
    perform_writeup as perform_icbinb_writeup,
    gather_citations,
)
from ai_scientist.perform_llm_review import perform_review, load_paper
from ai_scientist.perform_vlm_review import perform_imgs_cap_ref_review

from ai_scientist.utils.token_tracker import token_tracker


def print_time():
    """打印当前时间。"""
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def save_token_tracker(idea_dir):
    """
    保存 Token 使用情况统计。

    将 Token 摘要和交互详情保存到实验目录下的 JSON 文件中。

    Args:
        idea_dir (str): 实验结果保存目录。
    """
    # 保存 Token 使用摘要
    with open(osp.join(idea_dir, "token_tracker.json"), "w") as f:
        json.dump(token_tracker.get_summary(), f)
    # 保存详细的交互记录
    with open(osp.join(idea_dir, "token_tracker_interactions.json"), "w") as f:
        json.dump(token_tracker.get_interactions(), f)


def parse_arguments():
    """
    解析命令行参数。

    定义并解析用于控制实验运行的所有命令行参数。

    Returns:
        argparse.Namespace: 包含解析后参数的对象。
    """
    parser = argparse.ArgumentParser(description="Run AI scientist experiments")
    parser.add_argument(
        "--writeup-type",
        type=str,
        default="icbinb",
        choices=["normal", "icbinb"],
        help="生成的论文类型 (normal=8页, icbinb=4页)",
    )
    parser.add_argument(
        "--load_ideas",
        type=str,
        default="ideas/i_cant_believe_its_not_better.json",
        help="包含预生成想法的 JSON 文件路径",
    )
    parser.add_argument(
        "--idea_idx",
        type=int,
        default=0,
        help="要运行的想法索引",
    )
    parser.add_argument(
        "--code_ref_path",
        type=str,
        default=None,
        help="指定参考代码的存储路径",
    )
    parser.add_argument(
        "--dataset_ref_path",
        type=str,
        default=None,
        help="指定参考数据集的元数据文件路径",
    )
    parser.add_argument(
        "--writeup-retries",
        type=int,
        default=3,
        help="尝试生成论文的重试次数",
    )
    parser.add_argument(
        "--attempt_id",
        type=int,
        default=0,
        help="尝试 ID，用于区分并行运行中同一想法的不同尝试",
    )
    parser.add_argument(
        "--model_agg_plots",
        type=str,
        default="o3-mini-2025-01-31",
        help="用于图表聚合的模型",
    )
    parser.add_argument(
        "--model_writeup",
        type=str,
        default="o1-preview-2024-09-12",
        help="用于生成论文的大模型",
    )
    parser.add_argument(
        "--model_citation",
        type=str,
        default="gpt-4o-2024-11-20",
        help="用于收集引用的模型",
    )
    parser.add_argument(
        "--num_cite_rounds",
        type=int,
        default=20,
        help="执行引用搜索的轮数",
    )
    parser.add_argument(
        "--model_writeup_small",
        type=str,
        default="gpt-4o-2024-05-13",
        help="用于生成论文的小模型",
    )
    parser.add_argument(
        "--model_review",
        type=str,
        default="gpt-4o-2024-11-20",
        help="用于评审正文和图片的模型",
    )
    parser.add_argument(
        "--skip_writeup",
        action="store_true",
        help="如果设置，跳过论文生成过程",
    )
    parser.add_argument(
        "--skip_review",
        action="store_true",
        help="如果设置，跳过评审过程",
    )

    return parser.parse_args()


def get_available_gpus(gpu_ids=None):
    """
    获取可用的 GPU ID 列表。

    Args:
        gpu_ids (str, optional): 逗号分隔的 GPU ID 字符串。如果为 None，则检测所有可用 GPU。

    Returns:
        List[int]: 可用 GPU ID 的列表。
    """
    if gpu_ids is not None:
        return [int(gpu_id) for gpu_id in gpu_ids.split(",")]
    return list(range(torch.cuda.device_count()))


def find_pdf_path_for_review(idea_dir):
    """
    查找用于评审的 PDF 文件路径。

    优先查找包含 "final" 的 PDF，否则查找版本号最大的 "reflection" PDF。

    Args:
        idea_dir (str): 实验目录路径。

    Returns:
        str: 找到的 PDF 文件路径，如果未找到则可能返回非最佳匹配或报错。
    """
    pdf_files = [f for f in os.listdir(idea_dir) if f.endswith(".pdf")]
    # 筛选包含 "reflection" 的 PDF
    reflection_pdfs = [f for f in pdf_files if "reflection" in f]
    if reflection_pdfs:
        # 首先检查是否有最终版本
        final_pdfs = [f for f in reflection_pdfs if "final" in f.lower()]
        if final_pdfs:
            # 如果有最终版本，直接使用
            pdf_path = osp.join(idea_dir, final_pdfs[0])
        else:
            # 尝试查找带编号的 reflection PDF
            reflection_nums = []
            for f in reflection_pdfs:
                match = re.search(r"reflection[_.]?(\d+)", f)
                if match:
                    reflection_nums.append((int(match.group(1)), f))

            if reflection_nums:
                # 获取编号最大的文件
                highest_reflection = max(reflection_nums, key=lambda x: x[0])
                pdf_path = osp.join(idea_dir, highest_reflection[1])
            else:
                # 如果找不到编号，回退到第一个 reflection PDF
                pdf_path = osp.join(idea_dir, reflection_pdfs[0])
    return pdf_path


@contextmanager
def redirect_stdout_stderr_to_file(log_file_path):
    """
    上下文管理器：将标准输出和标准错误重定向到文件。

    用于捕获代码块的输出日志。

    Args:
        log_file_path (str): 日志文件路径。
    """
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log = open(log_file_path, "a")
    sys.stdout = log
    sys.stderr = log
    try:
        yield
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log.close()



if __name__ == "__main__":
    args = parse_arguments()
    
    # 设置 AI_SCIENTIST_ROOT 环境变量，指向当前脚本所在目录
    os.environ["AI_SCIENTIST_ROOT"] = os.path.dirname(os.path.abspath(__file__))
    print(f"Set AI_SCIENTIST_ROOT to {os.environ['AI_SCIENTIST_ROOT']}")

    # 检查可用 GPU 并根据需要调整并行进程
    available_gpus = get_available_gpus()
    print(f"Using GPUs: {available_gpus}")

    # 加载预生成的想法
    with open(args.load_ideas, "r") as f:
        ideas = json.load(f)
        print(f"Loaded {len(ideas)} pregenerated ideas from {args.load_ideas}")

    # 获取当前要运行的想法
    idea = ideas[args.idea_idx]

    # 创建带有时间戳和想法名称的实验目录
    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    idea_dir = f"experiments/{date}_{idea['Name']}_attempt_{args.attempt_id}"
    print(f"Results will be saved in {idea_dir}")
    os.makedirs(idea_dir, exist_ok=True)

    # 将想法 JSON 转换为 Markdown 文件，方便阅读和后续处理
    idea_path_md = osp.join(idea_dir, "idea.md")

    # 处理参考代码
    code = None
    code_path = None
    
    if args.code_ref_path:
        code_path = args.code_ref_path
        if os.path.exists(code_path):
            print(f"Loading reference code from {code_path}")
            with open(code_path, "r") as f:
                code = f.read()
        else:
            print(f"Warning: Code ref path {code_path} does not exist.")


    # 生成想法 Markdown 文件
    idea_to_markdown(ideas[args.idea_idx], idea_path_md, code_path)

    # 处理数据集引用/描述
    dataset_ref_content = None
    
    if args.dataset_ref_path:
        dataset_ref_path = args.dataset_ref_path
        if os.path.exists(dataset_ref_path):
            print(f"Loading dataset reference from {dataset_ref_path}")
            with open(dataset_ref_path, "r") as f:
                dataset_ref_content = f.read()
        else:
            print(f"Warning: Dataset ref path {dataset_ref_path} does not exist.")


    # 组合初始代码（数据集引用 + 加载的代码）
    added_code_parts = []
    if dataset_ref_content:
        added_code_parts.append(dataset_ref_content)
    if code:
        added_code_parts.append(code)
    
    added_code = "\n\n".join(added_code_parts) if added_code_parts else None

    print(f"Added code content length: {len(added_code) if added_code else 0}")

    # 如果有加载代码，将其添加到想法 JSON 中
    if added_code is not None:
        ideas[args.idea_idx]["Code"] = added_code

    # 存储原始想法 JSON 到实验目录
    idea_path_json = osp.join(idea_dir, "idea.json")
    with open(idea_path_json, "w") as f:
        json.dump(ideas[args.idea_idx], f, indent=4)

    # 编辑并保存实验配置文件
    config_path = "bfts_config.yaml"
    idea_config_path = edit_bfts_config_file(
        config_path,
        idea_dir,
        idea_path_json,
    )

    # 执行 BFTS 实验
    # 这是核心步骤，调用 AgentManager 进行多阶段实验
    perform_experiments_bfts(idea_config_path)
    
    # 复制实验结果
    experiment_results_dir = osp.join(idea_dir, "logs/0-run/experiment_results")
    if os.path.exists(experiment_results_dir):
        shutil.copytree(
            experiment_results_dir,
            osp.join(idea_dir, "experiment_results"),
            dirs_exist_ok=True,
        )

    # 聚合生成的图表
    aggregate_plots(base_folder=idea_dir, model=args.model_agg_plots)

    # 清理临时的实验结果目录
    shutil.rmtree(osp.join(idea_dir, "experiment_results"))

    # 保存 Token 使用情况
    save_token_tracker(idea_dir)

    # 如果不跳过论文生成，则执行论文写作
    if not args.skip_writeup:
        writeup_success = False
        # 收集引用
        citations_text = gather_citations(
            idea_dir,
            num_cite_rounds=args.num_cite_rounds,
            small_model=args.model_citation,
        )
        # 尝试生成论文，支持重试
        for attempt in range(args.writeup_retries):
            print(f"Writeup attempt {attempt+1} of {args.writeup_retries}")
            if args.writeup_type == "normal":
                writeup_success = perform_writeup(
                    base_folder=idea_dir,
                    small_model=args.model_writeup_small,
                    big_model=args.model_writeup,
                    page_limit=8,
                    citations_text=citations_text,
                )
            else:
                writeup_success = perform_icbinb_writeup(
                    base_folder=idea_dir,
                    small_model=args.model_writeup_small,
                    big_model=args.model_writeup,
                    page_limit=4,
                    citations_text=citations_text,
                )
            if writeup_success:
                break

        if not writeup_success:
            print("Writeup process did not complete successfully after all retries.")

    # 再次保存 Token 使用情况（包含写作阶段的消耗）
    save_token_tracker(idea_dir)

    # 如果不跳过评审且不跳过写作，执行论文评审
    if not args.skip_review and not args.skip_writeup:
        # 如果论文存在，执行评审
        pdf_path = find_pdf_path_for_review(idea_dir)
        if os.path.exists(pdf_path):
            print("Paper found at: ", pdf_path)
            paper_content = load_paper(pdf_path)
            client, client_model = create_client(args.model_review)
            # 评审正文
            review_text = perform_review(paper_content, client_model, client)
            # 评审图片和标题
            review_img_cap_ref = perform_imgs_cap_ref_review(
                client, client_model, pdf_path
            )
            # 保存评审结果
            with open(osp.join(idea_dir, "review_text.txt"), "w") as f:
                f.write(json.dumps(review_text, indent=4))
            with open(osp.join(idea_dir, "review_img_cap_ref.json"), "w") as f:
                json.dump(review_img_cap_ref, f, indent=4)
            print("Paper review completed.")

    print("Start cleaning up processes")
    # 终止与此实验相关的所有 mp 和 torch 进程
    import psutil
    import signal

    # 获取当前进程及其所有子进程
    current_process = psutil.Process()
    children = current_process.children(recursive=True)

    # 首先尝试优雅终止
    for child in children:
        try:
            child.send_signal(signal.SIGTERM)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    # 短暂等待进程终止
    gone, alive = psutil.wait_procs(children, timeout=3)

    # 如果仍有进程存活，强制终止
    for process in alive:
        try:
            process.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    # 额外清理：查找包含特定关键字的孤立进程
    keywords = ["python", "torch", "mp", "bfts", "experiment"]
    for proc in psutil.process_iter(["name", "cmdline"]):
        try:
            # 检查进程名称和命令行参数
            cmdline = " ".join(proc.cmdline()).lower()
            if any(keyword in cmdline for keyword in keywords):
                proc.send_signal(signal.SIGTERM)
                proc.wait(timeout=3)
                if proc.is_running():
                    proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
            continue

    # 最后，终止当前进程（注释掉以允许正常退出）
    # current_process.send_signal(signal.SIGTERM)
    # try:
    #     current_process.wait(timeout=3)
    # except psutil.TimeoutExpired:
    #     current_process.kill()

    # 退出程序
    sys.exit(0)

