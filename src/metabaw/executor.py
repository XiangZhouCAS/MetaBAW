from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime
import gzip
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Sequence

from .model import Task, topological_order
from .state import StateStore


@dataclass(frozen=True)
class RunResult:
    task_id: str
    return_code: int
    message: str


def _safe_log_name(task_id: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in task_id) + ".log"


def _local_time() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _terminal_text(value: str, maximum_chars: int) -> str:
    compact = " ".join(value.split())
    if len(compact) <= maximum_chars:
        return compact
    return compact[: maximum_chars - 3] + "..."


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
    summary = " | ".join(lines[-maximum_lines:])
    if len(summary) > maximum_chars:
        summary = "..." + summary[-(maximum_chars - 3) :]
    return summary


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


def _execute_task(
    task: Task,
    log_path: Path,
    shell_executable: str,
    max_memory_gb: float | None,
    temp_dir: Path | None,
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
    if max_memory_gb is not None:
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
            f"# max_memory_gb: {max_memory_gb if max_memory_gb is not None else 'unlimited'}\n\n"
        )
        log.flush()
        try:
            if isinstance(task.command, str):
                command = task.command
                if max_memory_gb is not None and os.name == "posix":
                    limit_kib = max(1, int(max_memory_gb * 1024 * 1024))
                    command = f"ulimit -v {limit_kib}\n{command}"
                process = subprocess.run(
                    command,
                    cwd=task.cwd,
                    env=environment,
                    shell=True,
                    executable=shell_executable,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            else:
                process = subprocess.run(
                    task.command,
                    cwd=task.cwd,
                    env=environment,
                    shell=False,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
        except OSError as exc:
            log.write(f"\nmetaBAW could not start the task: {exc}\n")
            log.write(f"# finished: {_local_time()}\n# exit_code: 127\n")
            return RunResult(task.id, 127, str(exc))
        log.write(f"\n# finished: {_local_time()}\n# exit_code: {process.returncode}\n")
        log.flush()
    if process.returncode != 0:
        message = f"command exited with code {process.returncode}"
        diagnosis = _failure_diagnosis(log_path)
        if diagnosis:
            message += f"; diagnosis: {diagnosis}"
        tail = _failure_tail(log_path)
        if tail:
            message += f"; last output: {tail}"
        return RunResult(task.id, process.returncode, message)
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

    def _emit(
        self,
        event: str,
        task: Task,
        position: int,
        total: int,
        detail: str,
    ) -> None:
        if not self.show_progress:
            return
        print(
            f"[{_local_time()}] [{event} {position}/{total}] {task.id} - {detail}",
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
                cache_hits.add(task.id)
                if (
                    record is None
                    or record.status != "success"
                    or not fingerprint_matches
                ):
                    adopted_hits.add(task.id)
                    self.state.adopt(
                        task.id,
                        fingerprints[task.id],
                        "Adopted complete outputs during startup resume scan",
                    )

        if self.show_progress:
            adopted = len(adopted_hits)
            adopted_detail = f", adopted={adopted}" if adopted else ""
            print(
                f"[{_local_time()}] [RESUME] Reusing {len(cache_hits)}/{total} "
                f"complete tasks; {total - len(cache_hits)} will run"
                f"{adopted_detail}.",
                flush=True,
            )

        statuses: dict[str, str] = {}
        pending: set[str] = set(by_id)
        attempts = {task_id: 0 for task_id in by_id}
        running: dict[Future[RunResult], tuple[str, int, int, float, Path]] = {}
        active_samples: set[str] = set()
        used_cpus = 0
        used_gpus = 0
        failure_seen = False
        blocked_task_ids: list[str] = []

        with ThreadPoolExecutor(max_workers=self.max_parallel) as pool:
            while pending or running:
                made_progress = False
                if not (failure_seen and self.fail_fast):
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
                    for task in ready:
                        if task.id in cache_hits:
                            statuses[task.id] = "skipped"
                            pending.remove(task.id)
                            made_progress = True
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
                        resources = [f"cpu={requested}"]
                        if requested_gpus:
                            resources.append(f"gpu={requested_gpus}")
                        self._emit(
                            "START",
                            task,
                            positions[task.id],
                            total,
                            (
                                f"{_terminal_text(task.description or 'Run task', 180)} "
                                f"({', '.join(resources)})"
                            ),
                        )
                        started = time.monotonic()
                        future = pool.submit(
                            _execute_task,
                            task,
                            log_path,
                            self.shell_executable,
                            self.max_memory_gb,
                            self.temp_dir,
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
                        result = future.result()
                        if result.return_code == 0:
                            statuses[task_id] = "success"
                            self.state.finish(task_id, "success", 0, result.message)
                            self._emit(
                                "DONE",
                                task,
                                positions[task_id],
                                total,
                                f"completed in {elapsed}",
                            )
                        elif attempts[task_id] <= max(
                            self.retries,
                            task.automatic_retries,
                        ):
                            statuses.pop(task_id, None)
                            pending.add(task_id)
                            self.state.finish(task_id, "retry", result.return_code, result.message)
                            self._emit(
                                "RETRY",
                                task,
                                positions[task_id],
                                total,
                                (
                                    f"attempt {attempts[task_id]} failed after {elapsed}: "
                                    f"{_terminal_text(result.message, 500)}; log={log_path}"
                                ),
                            )
                        else:
                            statuses[task_id] = "failed"
                            failure_seen = True
                            _remove_tolerated_outputs(task)
                            self.state.finish(task_id, "failed", result.return_code, result.message)
                            self._emit(
                                "FAIL",
                                task,
                                positions[task_id],
                                total,
                                (
                                    f"failed after {elapsed}: "
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
                    if failure_seen and self.fail_fast:
                        blocked = list(pending)
                    if blocked:
                        for task_id in sorted(blocked, key=positions.get):
                            statuses[task_id] = "blocked"
                            pending.remove(task_id)
                            task = by_id[task_id]
                            _remove_tolerated_outputs(task)
                            blocked_task_ids.append(task_id)
                        made_progress = True
                if not made_progress:
                    time.sleep(0.05)
        if self.show_progress and blocked_task_ids:
            preview = ", ".join(blocked_task_ids[:5])
            remaining = len(blocked_task_ids) - 5
            if remaining > 0:
                preview += f", +{remaining} more"
            print(
                f"[{_local_time()}] [BLOCKED] {len(blocked_task_ids)} downstream "
                f"task(s) were not run: {preview}.",
                flush=True,
            )
        return statuses

