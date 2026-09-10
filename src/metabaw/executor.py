from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime
import gzip
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import threading
import time
import traceback
from typing import Callable, Sequence

from .model import Task, topological_order
from .state import StateStore


@dataclass(frozen=True)
class RunResult:
    task_id: str
    return_code: int
    message: str
    memory_limit_exceeded: bool = False
    aborted_by_memory_limit: bool = False


MemorySampler = Callable[[Sequence[int]], int]


def _linux_process_tree_memory_bytes(
    root_pids: Sequence[int],
    proc_root: Path = Path("/proc"),
) -> int:
    """Return process-tree PSS, falling back to RSS when PSS is unavailable.

    PSS divides shared resident pages among the processes mapping them. Summing
    VmRSS would count a large shared mapping once per GTDB-Tk worker and can
    therefore report roughly twice the physical memory actually in use.
    """
    roots = {int(pid) for pid in root_pids if int(pid) > 0}
    if not roots or not proc_root.is_dir():
        return 0
    parents: dict[int, int] = {}
    rss_fallback: dict[int, int] = {}
    try:
        entries = tuple(proc_root.iterdir())
    except OSError:
        return 0
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="utf-8", errors="replace")
            close = stat.rfind(")")
            if close < 0:
                continue
            fields = stat[close + 2 :].split()
            parent_pid = int(fields[1])
            status = (entry / "status").read_text(
                encoding="utf-8",
                errors="replace",
            )
            match = re.search(r"^VmRSS:\s+(\d+)\s+kB$", status, re.MULTILINE)
            rss_bytes = int(match.group(1)) * 1024 if match else 0
            pid = int(entry.name)
            parents[pid] = parent_pid
            rss_fallback[pid] = rss_bytes
        except (OSError, ValueError, IndexError):
            continue
    selected = set(roots)
    while True:
        descendants = {
            pid
            for pid, parent_pid in parents.items()
            if parent_pid in selected
        }
        expanded = selected | descendants
        if expanded == selected:
            break
        selected = expanded
    total = 0
    for pid in selected:
        try:
            rollup = (proc_root / str(pid) / "smaps_rollup").read_text(
                encoding="utf-8",
                errors="replace",
            )
            match = re.search(r"^Pss:\s+(\d+)\s+kB$", rollup, re.MULTILINE)
        except OSError:
            match = None
        total += (
            int(match.group(1)) * 1024
            if match is not None
            else rss_fallback.get(pid, 0)
        )
    return total


class _WorkflowMemoryBudget:
    """Track accounted physical memory for every active task process tree."""

    def __init__(
        self,
        limit_gb: float | None,
        sampler: MemorySampler | None = None,
    ) -> None:
        self.limit_gb = limit_gb
        self.limit_bytes = (
            max(1, int(limit_gb * 1024**3))
            if limit_gb is not None
            else None
        )
        self.event = threading.Event()
        self._sampler = sampler or _linux_process_tree_memory_bytes
        self._lock = threading.Lock()
        self._roots: dict[int, str] = {}
        self.reason: str | None = None
        self.source_task: str | None = None
        self.observed_bytes: int | None = None
        self.peak_bytes = 0

    def register(self, task_id: str, process_id: int) -> None:
        with self._lock:
            self._roots[process_id] = task_id

    def unregister(self, process_id: int) -> None:
        with self._lock:
            self._roots.pop(process_id, None)

    def active_roots(self) -> tuple[int, ...]:
        with self._lock:
            return tuple(self._roots)

    def _trigger(
        self,
        reason: str,
        *,
        source_task: str | None = None,
        observed_bytes: int | None = None,
    ) -> bool:
        with self._lock:
            if self.event.is_set():
                return False
            self.reason = reason
            self.source_task = source_task
            self.observed_bytes = observed_bytes
            self.event.set()
            return True

    def sample(self) -> int:
        if self.limit_bytes is None:
            return 0
        roots = self.active_roots()
        usage = max(0, int(self._sampler(roots))) if roots else 0
        with self._lock:
            self.peak_bytes = max(self.peak_bytes, usage)
        if self.limit_bytes is not None and usage > self.limit_bytes:
            self._trigger("workflow_memory", observed_bytes=usage)
        return usage

    def report_task_allocation_failure(self, task_id: str) -> None:
        self._trigger("task_allocation_failure", source_task=task_id)

    def abort_detail(self) -> str:
        with self._lock:
            reason = self.reason
            source_task = self.source_task
            observed = self.observed_bytes
        limit = (
            f"{self.limit_gb:g} GiB"
            if self.limit_gb is not None
            else "available host memory"
        )
        if reason == "workflow_memory" and observed is not None:
            return (
                f"total workflow PSS={observed / 1024**3:.2f} GiB exceeded "
                f"--max-memory={limit}"
            )
        if reason == "task_allocation_failure":
            return (
                f"task={source_task or 'unknown'} reported a CPU-memory allocation "
                f"failure under total workflow --max-memory={limit}"
            )
        return f"the total workflow memory budget ({limit}) was exceeded"


def _safe_log_name(task_id: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in task_id) + ".log"


def _archive_retry_log(log_path: Path, attempt: int) -> Path | None:
    """Move a failed attempt log aside before the next attempt overwrites it."""
    if attempt < 1 or not log_path.is_file():
        return None
    candidate = log_path.with_name(
        f"{log_path.stem}.attempt-{attempt}{log_path.suffix}"
    )
    collision = 2
    while candidate.exists():
        candidate = log_path.with_name(
            f"{log_path.stem}.attempt-{attempt}.{collision}{log_path.suffix}"
        )
        collision += 1
    try:
        log_path.replace(candidate)
    except OSError:
        return None
    return candidate


def _local_time() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _elapsed_clock(started_at: float, now: float | None = None) -> str:
    total = max(0, int((time.monotonic() if now is None else now) - started_at))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _progress_group_label(task: Task, sample_names: Sequence[str]) -> str:
    """Remove sample components from a task ID for concise cohort progress."""
    if task.sample is None:
        return task.id
    label = task.id
    for sample in sorted(sample_names, key=len, reverse=True):
        if label == sample:
            label = ""
            continue
        if label.startswith(sample + "."):
            label = label[len(sample) + 1 :]
        label = label.replace(f".{sample}.", ".")
        if label.endswith("." + sample):
            label = label[: -(len(sample) + 1)]
    return label.strip(".") or task.stage


def _sample_summary(samples: Sequence[str], maximum: int = 8) -> str:
    unique = list(dict.fromkeys(samples))
    shown = ",".join(unique[:maximum])
    if len(unique) > maximum:
        shown += f",+{len(unique) - maximum}"
    return shown


def _is_internal_sample_name(sample: str) -> bool:
    """Return whether SAMPLE is a workflow input aggregate, not a real sample."""
    return sample.startswith("semibin2_multisample_input")


def _terminal_text(value: str, maximum_chars: int) -> str:
    compact = " ".join(value.split())
    if len(compact) <= maximum_chars:
        return compact
    return compact[: maximum_chars - 3] + "..."


def _unexpected_worker_result(
    task_id: str,
    log_path: Path,
    error: Exception,
) -> RunResult:
    """Convert an unexpected worker exception into a normal task failure."""
    error_text = str(error) or repr(error)
    diagnostic = f"{type(error).__name__}: {_terminal_text(error_text, 500)}"
    message = f"task execution worker raised an unexpected exception: {diagnostic}"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8", errors="replace") as log:
            log.write(
                "\n# executor_exception: "
                f"{diagnostic}\n"
                + "".join(
                    traceback.format_exception(
                        type(error),
                        error,
                        error.__traceback__,
                    )
                )
                + f"# finished: {_local_time()}\n# exit_code: 70\n"
            )
    except Exception as log_error:
        log_diagnostic = str(log_error) or repr(log_error)
        message += (
            f"; additionally could not append diagnostic information to {log_path}: "
            f"{type(log_error).__name__}: "
            f"{_terminal_text(log_diagnostic, 300)}"
        )
    return RunResult(task_id, 70, message)


def _failure_tail(log_path: Path, maximum_lines: int = 3, maximum_chars: int = 500) -> str:
    try:
        lines = [
            line.strip()
            for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            if (
                line.strip()
                and not line.lstrip().startswith("#")
                and not re.match(
                    r"^(?:real|user|sys)\s+\d",
                    line.strip(),
                    flags=re.IGNORECASE,
                )
            )
        ]
    except OSError:
        return ""
    error_lines = [
        line
        for line in lines
        if re.search(
            r"(?:traceback|(?:runtime|import|module|value|type|memory)error:|"
            r"cuda out of memory|resource temporarily unavailable|cannot allocate memory|"
            r"insufficient memory|std::bad_alloc)",
            line,
            flags=re.IGNORECASE,
        )
    ]
    selected = [*error_lines[-2:], *lines[-maximum_lines:]]
    summary = " | ".join(dict.fromkeys(selected))
    if len(summary) > maximum_chars:
        summary = "..." + summary[-(maximum_chars - 3) :]
    return summary


def _memory_limit_exceeded(log_path: Path, return_code: int | None = None) -> bool:
    """Detect a CPU-memory allocation failure that requires a workflow abort."""
    try:
        lines = log_path.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines()
    except OSError:
        return False
    # GPU exhaustion is handled separately through --max-gpu-memory. Do not
    # mistake it for a breach of the CPU --max-memory limit.
    cpu_output = "\n".join(
        line for line in lines if "cuda out of memory" not in line.lower()
    )
    if re.search(
        r"(?:\[morecore\]\s*insufficient memory|cannot allocate memory|"
        r"\bmemoryerror\b|std::bad_alloc|outofmemoryerror|"
        r"out of memory:\s*killed process|oom-kill|oom_kill)",
        cpu_output,
        flags=re.IGNORECASE,
    ):
        return True
    return return_code in {-9, 137} and bool(
        re.search(r"(?:out of memory|oom|\bkilled\b)", cpu_output, re.IGNORECASE)
    )


def _failure_diagnosis(log_path: Path) -> str:
    try:
        output = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    if "tf_data_private_threadpool creation via pthread_create() failed" in output:
        return (
            "TensorFlow could not create its prediction thread pool; the process reached "
            "a thread, process, or memory resource limit"
        )
    if "AF_UNIX path too long" in output:
        return (
            "Python multiprocessing could not create its Unix socket because the "
            "configured temporary path is too long"
        )
    if "EOFError" in output and "checkm2" in output.lower():
        return (
            "CheckM2's multiprocessing manager exited before the parent process could "
            "connect; an overlong TMPDIR Unix-socket path or a native worker crash is "
            "the likely cause"
        )
    if "libtorch_cpu.so" in output and "iJIT_NotifyEvent" in output:
        return (
            "LorBin's PyTorch runtime is incompatible with the installed Intel "
            "MKL version; run `metabaw check --scope lorbin` to repair the "
            "isolated environment"
        )
    if "Something went wrong with running training network" in output:
        if "CUDA out of memory" in output:
            return (
                "COMEBin exhausted GPU memory while training; lower "
                "--batch-size or increase --max-gpu-memory"
            )
        return (
            "COMEBin failed while training its representation model; the specific "
            "Python error is included in the task log"
        )
    if "No module named 'pkg_resources'" in output or (
        'No module named "pkg_resources"' in output
    ):
        return (
            "CheckM requires pkg_resources, which was removed from setuptools 82; "
            "run `metabaw check --scope checkm` and install the offered "
            "setuptools<82 compatibility version"
        )
    if (
        "OMP: Error #34" in output
        or (
            "OMP:" in output
            and "unable to allocate necessary resources for OMP thread" in output
        )
    ):
        return (
            "OpenMP could not create a worker thread because nested numerical-library "
            "threads exhausted the process or thread resource limit"
        )
    if (
        "Failed to load consolidated database" in output
        and "Cannot allocate memory" in output
    ):
        return (
            "GTDB-Tk could not map its consolidated database into the process address "
            "space; a virtual-memory limit or insufficient host memory is the likely cause"
        )
    if _memory_limit_exceeded(log_path):
        return (
            "the task reported CPU-memory exhaustion under the total workflow "
            "--max-memory budget or available host memory; reduce "
            "--task/--threads or increase --max-memory"
        )
    return ""


def _output_complete(path: Path) -> bool:
    if not path.exists():
        return False
    if not path.is_dir():
        return True
    try:
        return any(item.is_file() for item in path.rglob("*"))
    except OSError:
        return False


FASTA_SUFFIXES = (".fa", ".fna", ".fasta", ".fa.gz", ".fna.gz", ".fasta.gz")


def _fasta_validation_error(directory: Path) -> str | None:
    if not directory.is_dir():
        return f"{directory} (FASTA bin directory is missing)"
    fasta_files = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and any(path.name.lower().endswith(suffix) for suffix in FASTA_SUFFIXES)
    )
    if not fasta_files:
        return f"{directory} (no FASTA bins were generated)"
    for path in fasta_files:
        opener = gzip.open if path.name.lower().endswith(".gz") else open
        records = 0
        sequence_bases = 0
        current_has_sequence = False
        try:
            with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
                for raw in handle:
                    line = raw.strip()
                    if not line:
                        continue
                    if line.startswith(">"):
                        if records and not current_has_sequence:
                            return f"{path} (FASTA record has no sequence)"
                        if len(line) == 1:
                            return f"{path} (FASTA header is empty)"
                        records += 1
                        current_has_sequence = False
                    else:
                        if records == 0:
                            return f"{path} (sequence appears before the first FASTA header)"
                        sequence_bases += len(line)
                        current_has_sequence = True
        except OSError as exc:
            return f"{path} (cannot read FASTA: {exc})"
        if records == 0 or sequence_bases == 0 or not current_has_sequence:
            return f"{path} (FASTA contains no complete sequence record)"
    return None


def _validation_errors(task: Task) -> list[str]:
    return [
        error
        for directory in task.fasta_output_dirs
        if (error := _fasta_validation_error(directory)) is not None
    ]


def _outputs_exist(task: Task) -> bool:
    required = all(_output_complete(path) for path in task.outputs)
    alternatives = (
        not task.output_alternatives
        or any(
            all(_output_complete(path) for path in output_set)
            for output_set in task.output_alternatives
        )
    )
    return required and alternatives and not _validation_errors(task)


def _is_same_or_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _legacy_generated_inputs(
    ordered: Sequence[Task],
) -> dict[str, frozenset[Path]]:
    by_id = {task.id: task for task in ordered}
    upstream_outputs: dict[str, tuple[Path, ...]] = {}
    generated: dict[str, frozenset[Path]] = {}
    for task in ordered:
        candidates: list[Path] = []
        for dependency in task.deps:
            candidates.extend(upstream_outputs[dependency])
            candidates.extend(by_id[dependency].outputs)
        upstream_outputs[task.id] = tuple(dict.fromkeys(candidates))
        generated[task.id] = frozenset(
            path
            for path in task.inputs
            if any(_is_same_or_within(path, output) for output in candidates)
        )
    return generated


def _missing_outputs(task: Task) -> list[str]:
    missing = [
        (
            f"{path} (empty output directory)"
            if path.exists() and path.is_dir()
            else str(path)
        )
        for path in task.outputs
        if not _output_complete(path)
    ]
    if task.output_alternatives and not any(
        all(_output_complete(path) for path in output_set)
        for output_set in task.output_alternatives
    ):
        choices = " OR ".join(
            "[" + ", ".join(str(path) for path in output_set) + "]"
            for output_set in task.output_alternatives
        )
        missing.append(f"one complete alternative output set: {choices}")
    missing.extend(_validation_errors(task))
    return missing


def _remove_tolerated_outputs(task: Task) -> None:
    if not task.failure_tolerated:
        return
    paths = list(task.outputs)
    for output_set in task.output_alternatives:
        paths.extend(output_set)
    for path in paths:
        try:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
        except OSError:
            pass


def _terminate_process_tree(
    process: subprocess.Popen,
    grace_seconds: float = 3.0,
) -> None:
    """Terminate the complete process group created for one workflow task."""
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        elif os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            process.terminate()
    except (OSError, ProcessLookupError):
        pass
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except (OSError, ProcessLookupError):
        pass
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        pass


def _wait_for_process(
    process: subprocess.Popen,
    memory_budget: _WorkflowMemoryBudget | None,
) -> tuple[int, bool]:
    """Wait for a task, terminating its process tree after a memory abort."""
    while True:
        try:
            return process.wait(timeout=0.2), False
        except subprocess.TimeoutExpired:
            if memory_budget is not None and memory_budget.event.is_set():
                _terminate_process_tree(process)
                return process.wait(), True


def _execute_task(
    task: Task,
    log_path: Path,
    shell_executable: str,
    max_memory_gb: float | None,
    temp_dir: Path | None,
    memory_budget: _WorkflowMemoryBudget | None = None,
) -> RunResult:
    task.cwd.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(task.env)
    if temp_dir is not None:
        temp_dir.mkdir(parents=True, exist_ok=True)
        environment.update(
            {
                "TMPDIR": str(temp_dir),
                "TMP": str(temp_dir),
                "TEMP": str(temp_dir),
            }
        )
    if max_memory_gb is not None and task.enforce_memory_limit:
        environment["METABAW_MAX_MEMORY_GB"] = f"{max_memory_gb:g}"
    started_at = _local_time()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write(
            f"# task: {task.id}\n# stage: {task.stage}\n# description: {task.description}\n"
            f"# started: {started_at}\n# cwd: {task.cwd}\n"
        )
        log.write(f"# temp_dir: {temp_dir if temp_dir is not None else 'system default'}\n")
        if task.env:
            overrides = " ".join(
                f"{name}={value}" for name, value in sorted(task.env.items())
            )
            log.write(f"# environment_overrides: {overrides}\n")
        log.write(f"# command: {task.display_command()}\n")
        log.write(
            "# workflow_max_memory_gb: "
            f"{max_memory_gb if max_memory_gb is not None else 'unlimited'}\n"
        )
        log.write(
            "# workflow_memory_scope: combined PSS of all active task process trees "
            "(RSS fallback when PSS is unavailable)\n"
            "# per_process_address_limit: "
            f"{'enabled as a secondary guard' if task.enforce_memory_limit else 'disabled; task remains included in workflow memory accounting'}\n\n"
        )
        log.flush()
        try:
            process_options: dict[str, object] = {}
            if os.name == "posix":
                process_options["start_new_session"] = True
            elif os.name == "nt":
                process_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            if isinstance(task.command, str):
                command = task.command
                if (
                    max_memory_gb is not None
                    and task.enforce_memory_limit
                    and os.name == "posix"
                ):
                    limit_kib = max(1, int(max_memory_gb * 1024 * 1024))
                    command = f"ulimit -v {limit_kib}\n{command}"
                process = subprocess.Popen(
                    command,
                    cwd=task.cwd,
                    env=environment,
                    shell=True,
                    executable=shell_executable,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    **process_options,
                )
            else:
                process = subprocess.Popen(
                    task.command,
                    cwd=task.cwd,
                    env=environment,
                    shell=False,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    **process_options,
                )
        except OSError as exc:
            log.write(f"\nmetaBAW could not start the task: {exc}\n")
            log.write(f"# finished: {_local_time()}\n# exit_code: 127\n")
            return RunResult(task.id, 127, str(exc))
        if memory_budget is not None:
            memory_budget.register(task.id, process.pid)
        try:
            try:
                return_code, aborted = _wait_for_process(process, memory_budget)
            except Exception:
                _terminate_process_tree(process)
                raise
        finally:
            if memory_budget is not None:
                memory_budget.unregister(process.pid)
        log.flush()
        memory_exceeded = (
            not aborted
            and return_code != 0
            and _memory_limit_exceeded(log_path, return_code)
        )
        if memory_exceeded:
            if memory_budget is not None:
                memory_budget.report_task_allocation_failure(task.id)
            limit = (
                f"{max_memory_gb:g} GiB"
                if max_memory_gb is not None
                else "the available host memory"
            )
            log.write(
                f"\n# memory_limit_exceeded: task reported a CPU-memory "
                f"allocation failure under total workflow --max-memory={limit}\n"
                "# workflow_abort: all other running task process groups are "
                "being terminated; pending tasks will not start\n"
            )
        elif aborted:
            detail = (
                memory_budget.abort_detail()
                if memory_budget is not None
                else "the workflow memory budget was exceeded"
            )
            log.write(
                "\n# workflow_abort: task process group was terminated because "
                f"{detail}\n"
            )
        log.write(f"\n# finished: {_local_time()}\n# exit_code: {return_code}\n")
        log.flush()
    if aborted:
        return RunResult(
            task.id,
            130,
            "terminated because the total workflow memory limit was exceeded",
            aborted_by_memory_limit=True,
        )
    if return_code != 0:
        message = f"command exited with code {return_code}"
        diagnosis = _failure_diagnosis(log_path)
        if diagnosis:
            message += f"; diagnosis: {diagnosis}"
        tail = _failure_tail(log_path)
        if tail:
            message += f"; last output: {tail}"
        return RunResult(
            task.id,
            return_code,
            message,
            memory_limit_exceeded=memory_exceeded,
        )
    missing = _missing_outputs(task)
    if missing:
        message = f"expected outputs were not created: {missing}"
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"# validation_error: {message}\n")
        return RunResult(task.id, 98, message)
    return RunResult(task.id, 0, "ok")


class Executor:
    def __init__(
        self,
        state: StateStore,
        log_dir: Path,
        max_cpus: int,
        max_parallel: int,
        shell_executable: str = "/bin/bash",
        retries: int = 0,
        fail_fast: bool = True,
        max_memory_gb: float | None = None,
        show_progress: bool = False,
        temp_dir: Path | None = None,
        max_gpus: int = 0,
        started_at: float | None = None,
        memory_sampler: MemorySampler | None = None,
    ):
        self.state = state
        self.log_dir = log_dir
        self.max_cpus = max_cpus
        self.max_parallel = max_parallel
        self.shell_executable = shell_executable
        self.retries = retries
        self.fail_fast = fail_fast
        self.max_memory_gb = max_memory_gb
        self.show_progress = show_progress
        self.temp_dir = temp_dir
        self.max_gpus = max(0, max_gpus)
        self.started_at = time.monotonic() if started_at is None else started_at
        self.memory_sampler = memory_sampler
        self.peak_memory_gb = 0.0
        self.memory_abort_reason: str | None = None

    def _emit(
        self,
        event: str,
        label: str,
        position: int,
        total: int,
        detail: str,
    ) -> None:
        if not self.show_progress:
            return
        print(
            f"[{_elapsed_clock(self.started_at)}] "
            f"[{event} {position}/{total}] {label} - {detail}",
            flush=True,
        )

    def run(self, tasks: Sequence[Task], force: bool = False) -> dict[str, str]:
        ordered = topological_order(tasks)
        oversized_gpu_tasks = [
            task.id for task in ordered if task.gpus > self.max_gpus
        ]
        if oversized_gpu_tasks:
            raise ValueError(
                "GPU tasks exceed the detected GPU capacity: "
                + ", ".join(oversized_gpu_tasks)
            )
        by_id = {task.id: task for task in ordered}
        positions = {task.id: number for number, task in enumerate(ordered, start=1)}
        total = len(ordered)
        fingerprints: dict[str, str] = {}
        for task in ordered:
            fingerprints[task.id] = task.fingerprint([fingerprints[dep] for dep in task.deps])
        legacy_generated_inputs = _legacy_generated_inputs(ordered)
        legacy_fingerprints: dict[str, str] = {}
        for task in ordered:
            legacy_fingerprints[task.id] = task.fingerprint(
                [legacy_fingerprints[dependency] for dependency in task.deps],
                assume_missing_inputs=legacy_generated_inputs[task.id],
            )

        cache_hits: set[str] = set()
        adopted_hits: set[str] = set()
        for task in ordered:
            record = self.state.get(task.id)
            outputs_exist = _outputs_exist(task)
            fingerprint_matches = (
                record is not None
                and record.fingerprint
                in {fingerprints[task.id], legacy_fingerprints[task.id]}
            )
            if (
                not force
                and outputs_exist
                and bool(task.outputs or task.output_alternatives)
            ):
                if record is None:
                    cache_hits.add(task.id)
                    adopted_hits.add(task.id)
                    self.state.adopt(
                        task.id,
                        fingerprints[task.id],
                        "Adopted complete outputs during startup resume scan",
                    )
                elif record.status == "success" and fingerprint_matches:
                    cache_hits.add(task.id)

        if self.show_progress:
            adopted = len(adopted_hits)
            adopted_detail = f", adopted={adopted}" if adopted else ""
            print(
                f"[{_elapsed_clock(self.started_at)}] [RESUME] "
                f"Reusing {len(cache_hits)}/{total} "
                f"complete tasks; {total - len(cache_hits)} will run"
                f"{adopted_detail}.",
                flush=True,
            )

        statuses: dict[str, str] = {}
        pending: set[str] = set(by_id)
        attempts = {task_id: 0 for task_id in by_id}
        running: dict[Future[RunResult], tuple[str, int, int, float, Path]] = {}
        memory_budget = _WorkflowMemoryBudget(
            self.max_memory_gb,
            sampler=self.memory_sampler,
        )
        memory_notice_emitted = False
        active_samples: set[str] = set()
        used_cpus = 0
        used_gpus = 0
        failure_seen = False
        blocked_task_ids: list[str] = []
        sample_names = sorted(
            {
                task.sample
                for task in ordered
                if task.sample is not None
            },
            key=str,
        )
        progress_key_by_task: dict[str, str] = {}
        progress_members: dict[str, list[str]] = {}
        progress_labels: dict[str, str] = {}
        for task in ordered:
            if task.id in cache_hits:
                continue
            label = _progress_group_label(task, sample_names)
            key = f"{task.stage}\0{label}"
            progress_key_by_task[task.id] = key
            progress_members.setdefault(key, []).append(task.id)
            progress_labels[key] = label
        progress_positions = {
            key: index
            for index, key in enumerate(progress_members, start=1)
        }
        progress_order = list(progress_members)
        progress_total = len(progress_members)
        progress_started: set[str] = set()
        progress_finished: set[str] = set()
        progress_started_at: dict[str, float] = {}
        progress_failures: dict[
            str,
            list[tuple[str, str | None, Path]],
        ] = {}

        def announce_memory_abort() -> None:
            nonlocal memory_notice_emitted
            if memory_notice_emitted or not memory_budget.event.is_set():
                return
            memory_notice_emitted = True
            if self.show_progress:
                print(
                    f"[{_elapsed_clock(self.started_at)}] [MEMORY LIMIT] "
                    f"{memory_budget.abort_detail()}; terminating "
                    f"{len(running)} active task process group(s), disabling "
                    "retries, and blocking pending tasks.",
                    flush=True,
                )

        def start_progress_group(
            task: Task,
            requested_cpus: int,
            requested_gpus: int,
        ) -> None:
            key = progress_key_by_task[task.id]
            if key in progress_started:
                return
            progress_started.add(key)
            progress_started_at[key] = time.monotonic()
            members = [by_id[task_id] for task_id in progress_members[key]]
            samples = [
                member.sample
                for member in members
                if (
                    member.sample is not None
                    and not _is_internal_sample_name(member.sample)
                )
            ]
            resources = [
                (
                    f"cpu={requested_cpus}/sample"
                    if samples
                    else f"cpu={requested_cpus}"
                )
            ]
            if requested_gpus:
                resources.append(
                    (
                        f"gpu={requested_gpus}/sample"
                        if samples
                        else f"gpu={requested_gpus}"
                    )
                )
            if samples:
                unique_samples = list(dict.fromkeys(samples))
                detail = (
                    f"running {len(members)} task(s) across "
                    f"{len(unique_samples)} sample(s) "
                    f"[{_sample_summary(unique_samples)}] "
                    f"({', '.join(resources)})"
                )
            else:
                detail = (
                    f"{_terminal_text(task.description or 'Run task', 180)} "
                    f"({', '.join(resources)})"
                )
            self._emit(
                "STEP",
                progress_labels[key],
                progress_positions[key],
                progress_total,
                detail,
            )

        def finish_ready_progress_groups() -> None:
            terminal_states = {"success", "failed", "blocked"}
            for key in progress_order:
                if key in progress_finished:
                    continue
                members = progress_members[key]
                if not all(statuses.get(task_id) in terminal_states for task_id in members):
                    continue
                if key not in progress_started:
                    progress_finished.add(key)
                    continue
                progress_finished.add(key)
                elapsed = _duration(
                    time.monotonic() - progress_started_at[key]
                )
                failed = progress_failures.get(key, [])
                if not failed:
                    continue
                failed_samples = list(
                    dict.fromkeys(
                        sample or task_id
                        for task_id, sample, _ in failed
                    )
                )
                detail = (
                    f"finished in {elapsed}; failed={len(failed)}, "
                    f"affected=[{_sample_summary(failed_samples)}]"
                )
                self._emit(
                    "FAIL",
                    progress_labels[key],
                    progress_positions[key],
                    progress_total,
                    detail,
                )

        with ThreadPoolExecutor(max_workers=self.max_parallel) as pool:
            while pending or running:
                made_progress = False
                memory_budget.sample()
                announce_memory_abort()
                if not memory_budget.event.is_set() and not (
                    failure_seen and self.fail_fast
                ):
                    ready = [
                        by_id[task_id]
                        for task_id in pending
                        if (
                            all(
                                statuses.get(dep) in {"success", "skipped"}
                                for dep in by_id[task_id].scheduling_dependencies
                            )
                            or (
                                by_id[task_id].allow_failed_deps
                                and all(
                                    statuses.get(dep)
                                    in {"success", "skipped", "failed", "blocked"}
                                    for dep in by_id[task_id].scheduling_dependencies
                                )
                            )
                        )
                    ]
                    ready.sort(key=lambda task: (task.stage, task.priority, task.id))
                    active_progress_key = next(
                        (
                            key
                            for key in progress_order
                            if key not in progress_finished
                        ),
                        None,
                    )
                    for task in ready:
                        if task.id in cache_hits:
                            statuses[task.id] = "skipped"
                            pending.remove(task.id)
                            made_progress = True
                            continue
                        if progress_key_by_task[task.id] != active_progress_key:
                            continue
                        if task.sample is not None and task.sample in active_samples:
                            continue
                        requested = min(max(task.cpus, 1), self.max_cpus)
                        requested_gpus = max(0, task.gpus)
                        if (
                            len(running) >= self.max_parallel
                            or used_cpus + requested > self.max_cpus
                            or used_gpus + requested_gpus > self.max_gpus
                        ):
                            continue
                        fingerprints[task.id] = task.fingerprint(
                            [fingerprints[dependency] for dependency in task.deps]
                        )
                        attempts[task.id] += 1
                        log_path = self.log_dir / _safe_log_name(task.id)
                        self.state.start(task.id, fingerprints[task.id], log_path)
                        start_progress_group(task, requested, requested_gpus)
                        started = time.monotonic()
                        future = pool.submit(
                            _execute_task,
                            task,
                            log_path,
                            self.shell_executable,
                            self.max_memory_gb,
                            self.temp_dir,
                            memory_budget,
                        )
                        running[future] = (
                            task.id,
                            requested,
                            requested_gpus,
                            started,
                            log_path,
                        )
                        used_cpus += requested
                        used_gpus += requested_gpus
                        if task.sample is not None:
                            active_samples.add(task.sample)
                        pending.remove(task.id)
                        statuses[task.id] = "running"
                        made_progress = True

                if running:
                    done, _ = wait(running, timeout=0.2, return_when=FIRST_COMPLETED)
                    for future in done:
                        (
                            task_id,
                            requested,
                            requested_gpus,
                            started,
                            log_path,
                        ) = running.pop(future)
                        task = by_id[task_id]
                        elapsed = _duration(time.monotonic() - started)
                        used_cpus -= requested
                        used_gpus -= requested_gpus
                        if task.sample is not None:
                            active_samples.discard(task.sample)
                        try:
                            result = future.result()
                        except Exception as error:
                            result = _unexpected_worker_result(
                                task_id,
                                log_path,
                                error,
                            )
                        if result.memory_limit_exceeded:
                            memory_budget.report_task_allocation_failure(task_id)
                            announce_memory_abort()
                            statuses[task_id] = "failed"
                            failure_seen = True
                            _remove_tolerated_outputs(task)
                            self.state.finish(
                                task_id,
                                "failed",
                                result.return_code,
                                result.message,
                            )
                            key = progress_key_by_task[task_id]
                            progress_failures.setdefault(key, []).append(
                                (task_id, task.sample, log_path)
                            )
                        elif result.aborted_by_memory_limit:
                            statuses[task_id] = "blocked"
                            _remove_tolerated_outputs(task)
                            self.state.finish(
                                task_id,
                                "blocked",
                                result.return_code,
                                result.message,
                            )
                            blocked_task_ids.append(task_id)
                        elif result.return_code == 0:
                            statuses[task_id] = "success"
                            self.state.finish(task_id, "success", 0, result.message)
                        elif attempts[task_id] <= max(
                            self.retries,
                            task.automatic_retries,
                        ):
                            archived_log = _archive_retry_log(
                                log_path,
                                attempts[task_id],
                            )
                            statuses.pop(task_id, None)
                            pending.add(task_id)
                            self.state.finish(task_id, "retry", result.return_code, result.message)
                            key = progress_key_by_task[task_id]
                            self._emit(
                                "RETRY",
                                progress_labels[key],
                                progress_positions[key],
                                progress_total,
                                (
                                    f"sample={task.sample or task.id}; attempt "
                                    f"{attempts[task_id]} failed after {elapsed}: "
                                    f"{_terminal_text(result.message, 500)}; "
                                    f"log={archived_log or log_path}"
                                ),
                            )
                        else:
                            statuses[task_id] = "failed"
                            failure_seen = True
                            _remove_tolerated_outputs(task)
                            self.state.finish(task_id, "failed", result.return_code, result.message)
                            key = progress_key_by_task[task_id]
                            progress_failures.setdefault(key, []).append(
                                (task_id, task.sample, log_path)
                            )
                            self._emit(
                                "ERROR",
                                progress_labels[key],
                                progress_positions[key],
                                progress_total,
                                (
                                    f"sample={task.sample or task.id} failed after {elapsed}: "
                                    f"{_terminal_text(result.message, 500)}; "
                                    f"log={log_path}"
                                ),
                            )
                        made_progress = True
                elif pending:
                    blocked = [
                        task_id
                        for task_id in pending
                        if (
                            not by_id[task_id].allow_failed_deps
                            and any(
                                statuses.get(dep) in {"failed", "blocked"}
                                for dep in by_id[task_id].scheduling_dependencies
                            )
                        )
                    ]
                    if memory_budget.event.is_set() or (
                        failure_seen and self.fail_fast
                    ):
                        blocked = list(pending)
                    if blocked:
                        for task_id in sorted(blocked, key=positions.get):
                            statuses[task_id] = "blocked"
                            pending.remove(task_id)
                            task = by_id[task_id]
                            _remove_tolerated_outputs(task)
                            blocked_task_ids.append(task_id)
                        made_progress = True
                finish_ready_progress_groups()
                if not made_progress:
                    time.sleep(0.05)
        self.peak_memory_gb = memory_budget.peak_bytes / 1024**3
        self.memory_abort_reason = (
            memory_budget.abort_detail()
            if memory_budget.event.is_set()
            else None
        )
        if self.show_progress and blocked_task_ids:
            preview = ", ".join(blocked_task_ids[:5])
            remaining = len(blocked_task_ids) - 5
            if remaining > 0:
                preview += f", +{remaining} more"
            print(
                f"[{_elapsed_clock(self.started_at)}] [BLOCKED] "
                f"{len(blocked_task_ids)} downstream "
                f"task(s) were not run: {preview}.",
                flush=True,
            )
        return statuses

