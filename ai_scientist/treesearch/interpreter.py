"""
Python interpreter for executing code snippets and capturing their output.
Supports:
- captures stdout and stderr
- captures exceptions and stack traces
- limits execution time
"""

import logging
import os
import queue
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from multiprocessing import Process, Queue
from pathlib import Path

import humanize
from dataclasses_json import DataClassJsonMixin

logger = logging.getLogger("ai-scientist")


@dataclass
class ExecutionResult(DataClassJsonMixin):
    """
    Result of executing a code snippet in the interpreter.
    Contains the output, execution time, and exception information.
    """

    term_out: list[str]
    exec_time: float
    exc_type: str | None
    exc_info: dict | None = None
    exc_stack: list[tuple] | None = None


def exception_summary(e, working_dir, exec_file_name, format_tb_ipython):
    """
    生成异常及其堆栈跟踪的摘要字符串（支持标准 Python REPL 或 IPython 格式）。
    
    Args:
        e (Exception): 捕获的异常对象。
        working_dir (Path): 当前工作目录。
        exec_file_name (str): 执行的文件名。
        format_tb_ipython (bool): 是否使用 IPython 格式化堆栈跟踪。
        
    Returns:
        tuple: (堆栈跟踪字符串, 异常类名, 异常详细信息, 堆栈列表)
    """
    if format_tb_ipython:
        import IPython.core.ultratb

        tb = IPython.core.ultratb.VerboseTB(tb_offset=1, color_scheme="NoColor")
        tb_str = str(tb.text(*sys.exc_info()))
    else:
        tb_lines = traceback.format_exception(e)
        # 跳过 weflow 代码中的堆栈部分
        tb_str = "".join(
            [l for l in tb_lines if "treesearch/" not in l and "importlib" not in l]
        )

    # 将完整路径替换为仅文件名（移除代理工作空间目录路径，使输出更清晰）
    tb_str = tb_str.replace(str(working_dir / exec_file_name), exec_file_name)

    exc_info = {}
    if hasattr(e, "args"):
        exc_info["args"] = [str(i) for i in e.args]
    for att in ["name", "msg", "obj"]:
        if hasattr(e, att):
            exc_info[att] = str(getattr(e, att))

    tb = traceback.extract_tb(e.__traceback__)
    exc_stack = [(t.filename, t.lineno, t.name, t.line) for t in tb]

    return tb_str, e.__class__.__name__, exc_info, exc_stack


class RedirectQueue:
    """
    将标准输出/错误重定向到多进程队列的辅助类。
    """
    def __init__(self, queue):
        self.queue = queue

    def write(self, msg):
        self.queue.put(msg)

    def flush(self):
        pass


class Interpreter:
    def __init__(
        self,
        working_dir: Path | str,
        timeout: int = 3600,
        format_tb_ipython: bool = False,
        agent_file_name: str = "runfile.py",
        env_vars: dict[str, str] = {},
    ):
        """
        模拟具有执行时间限制的独立 Python REPL。

        Args:
            working_dir (Path | str): 代理的工作目录。
            timeout (int, optional): 每个代码执行步骤的超时时间（秒）。默认为 3600。
            format_tb_ipython (bool, optional): 异常格式化是否使用 IPython 格式。默认为 False。
            agent_file_name (str, optional): 代理代码文件的名称。默认为 "runfile.py"。
            env_vars (dict[str, str], optional): 设置在子进程中的环境变量。默认为 {}。
        """
        # 必须是绝对路径，否则可能会引发未捕获的问题
        self.working_dir = Path(working_dir).resolve()
        assert (
            self.working_dir.exists()
        ), f"Working directory {self.working_dir} does not exist"
        self.timeout = timeout
        self.format_tb_ipython = format_tb_ipython
        self.agent_file_name = agent_file_name
        self.process: Process = None  # type: ignore
        self.env_vars = env_vars

    def child_proc_setup(self, result_outq: Queue) -> None:
        # disable all warnings (before importing anything)
        import shutup

        shutup.mute_warnings()

        for key, value in self.env_vars.items():
            os.environ[key] = value

        os.chdir(str(self.working_dir))

        # this seems to only  benecessary because we're exec'ing code from a string,
        # a .py file should be able to import modules from the cwd anyway
        sys.path.append(str(self.working_dir))

        # capture stdout and stderr
        # trunk-ignore(mypy/assignment)
        sys.stdout = sys.stderr = RedirectQueue(result_outq)

    def _run_session(
        self, code_inq: Queue, result_outq: Queue, event_outq: Queue
    ) -> None:
        self.child_proc_setup(result_outq)

        global_scope: dict = {}
        while True:
            code = code_inq.get()
            os.chdir(str(self.working_dir))
            with open(self.agent_file_name, "w") as f:
                f.write(code)

            event_outq.put(("state:ready",))
            try:
                exec(compile(code, self.agent_file_name, "exec"), global_scope)
            except BaseException as e:
                tb_str, e_cls_name, exc_info, exc_stack = exception_summary(
                    e,
                    self.working_dir,
                    self.agent_file_name,
                    self.format_tb_ipython,
                )
                result_outq.put(tb_str)
                if e_cls_name == "KeyboardInterrupt":
                    e_cls_name = "TimeoutError"

                event_outq.put(("state:finished", e_cls_name, exc_info, exc_stack))
            else:
                event_outq.put(("state:finished", None, None, None))

            # put EOF marker to indicate that we're done
            result_outq.put("<|EOF|>")

    def create_process(self) -> None:
        # we use three queues to communicate with the child process:
        # - code_inq: send code to child to execute
        # - result_outq: receive stdout/stderr from child
        # - event_outq: receive events from child (e.g. state:ready, state:finished)
        # trunk-ignore(mypy/var-annotated)
        self.code_inq, self.result_outq, self.event_outq = Queue(), Queue(), Queue()
        self.process = Process(
            target=self._run_session,
            args=(self.code_inq, self.result_outq, self.event_outq),
        )
        self.process.start()

    def _drain_queues(self):
        """Quickly drain all in-flight messages to prevent blocking."""
        while not self.result_outq.empty():
            try:
                self.result_outq.get_nowait()
            except Exception:
                break

        while not self.event_outq.empty():
            try:
                self.event_outq.get_nowait()
            except Exception:
                break

        while not self.code_inq.empty():
            try:
                self.code_inq.get_nowait()
            except Exception:
                break

    def cleanup_session(self):
        if self.process is None:
            return
        # give the child process a chance to terminate gracefully
        self.process.terminate()
        self._drain_queues()
        self.process.join(timeout=2)
        # kill the child process if it's still alive
        if self.process.exitcode is None:
            logger.warning("Child process failed to terminate gracefully, killing it..")
            self.process.kill()
            self._drain_queues()
            self.process.join(timeout=2)
        # don't wait for gc, clean up immediately
        self.process.close()
        self.process = None  # type: ignore

    def run(self, code: str, reset_session=True) -> ExecutionResult:
        """
        在单独的进程中执行提供的 Python 命令并返回其输出。

        Args:
            code (str): 要执行的 Python 代码。
            reset_session (bool, optional): 是否在执行代码前重置解释器会话。默认为 True。

        Returns:
            ExecutionResult: 包含代码执行输出和元数据的对象。
        """

        logger.debug(f"REPL is executing code (reset_session={reset_session})")

        if reset_session:
            if self.process is not None:
                # 终止并清理之前的进程
                self.cleanup_session()
            self.create_process()
        else:
            # 第一次执行时 reset_session 必须为 True
            assert self.process is not None

        assert self.process.is_alive()

        self.code_inq.put(code)

        # 等待子进程实际开始执行（我们不想中断子进程的设置）
        try:
            state = self.event_outq.get(timeout=10)
        except queue.Empty:
            msg = "REPL child process failed to start execution"
            logger.critical(msg)
            while not self.result_outq.empty():
                logger.error(f"REPL output queue dump: {self.result_outq.get()}")
            raise RuntimeError(msg) from None
        assert state[0] == "state:ready", state
        start_time = time.time()

        # 此标志指示子进程已超时并发送了中断信号
        # 如果子进程在没有设置此标志的情况下死亡，则是意外终止
        child_in_overtime = False

        while True:
            try:
                # 检查子进程是否完成
                state = self.event_outq.get(timeout=1)  # wait for state:finished
                assert state[0] == "state:finished", state
                exec_time = time.time() - start_time
                break
            except queue.Empty:
                # 我们没有收到子进程的回复 -> 检查它是否仍然存活（假设尚未发送超时中断）
                if not child_in_overtime and not self.process.is_alive():
                    msg = "REPL child process died unexpectedly"
                    logger.critical(msg)
                    while not self.result_outq.empty():
                        logger.error(
                            f"REPL output queue dump: {self.result_outq.get()}"
                        )
                    raise RuntimeError(msg) from None

                # 子进程仍然存活且正在执行 -> 检查是否应该发送 sigint..
                if self.timeout is None:
                    continue
                running_time = time.time() - start_time
                if running_time > self.timeout:
                    # [TODO] handle this in a better way
                    assert reset_session, "Timeout ocurred in interactive session"

                    # 发送中断信号给子进程
                    os.kill(self.process.pid, signal.SIGINT)  # type: ignore
                    child_in_overtime = True
                    # 如果超时超过一分钟，则终止进程
                    if running_time > self.timeout + 60:
                        logger.warning("Child failed to terminate, killing it..")
                        self.cleanup_session()

                        state = (None, "TimeoutError", {}, [])
                        exec_time = self.timeout
                        break

        output: list[str] = []
        # 读取子进程的所有 stdout/stderr 直到 EOF 标记
        # 等待队列为空是不够的，因为子进程中的 feeder 线程可能仍在向队列添加内容
        while not self.result_outq.empty() or not output or output[-1] != "<|EOF|>":
            output.append(self.result_outq.get())
        output.pop()  # remove the EOF marker

        e_cls_name, exc_info, exc_stack = state[1:]

        if e_cls_name == "TimeoutError":
            output.append(
                f"TimeoutError: Execution exceeded the time limit of {humanize.naturaldelta(self.timeout)}"
            )
        else:
            output.append(
                f"Execution time: {humanize.naturaldelta(exec_time)} seconds (time limit is {humanize.naturaldelta(self.timeout)})."
            )
        return ExecutionResult(output, exec_time, e_cls_name, exc_info, exc_stack)
