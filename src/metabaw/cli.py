from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
from typing import Callable, Sequence

from . import __version__
from .direct import (
    BINNER_ORDER,
    AnnotationBuilder,
    AnnotationOptions,
    BinOptions,
    DirectBinBuilder,
    parse_choices,
)
from .dependencies import (
    CudaRuntimeStatus,
    HostCudaStatus,
    ISOLATED_TOOLS,
    IsolatedEnvironmentStatus,
    IsolatedToolSpec,
    SOFTWARE,
    SoftwareRequirement,
    SoftwareRuntimeIssue,
    all_requirements,
    annotation_requirements,
    bin_requirements,
    configured_binchicken_singlem_metapackage,
    configured_database_path,
    configured_isolated_environment,
    configured_magscot_directory,
    confirm_install,
    checkm_database_missing,
    checkm_database_valid,
    cuda_runtime_status,
    dbcan_database_missing,
    dbcan_database_valid,
    diamond_database_missing,
    diamond_database_valid,
    gtdbtk_database_missing,
    gtdbtk_database_valid,
    hydrogenase_database_missing,
    hydrogenase_database_valid,
    host_cuda_status,
    install_checkm2_database,
    install_dbcan_database,
    install_gtdbtk_database,
    install_gunc_database,
    install_kofam_database,
    install_isolated_cuda_runtime,
    install_isolated_environment,
    install_magscot,
    install_main_cuda_runtime,
    install_software,
    isolated_environment_status,
    isolated_run_prefix,
    kofam_database_missing,
    kofam_database_valid,
    magscot_files,
    missing_dastool_r_packages,
    missing_magscot_r_packages,
    missing_software,
    normalize_isolated_environment,
    print_cuda_runtime_status,
    print_host_cuda_status,
    print_isolated_status,
    print_software_status,
    reference_package_valid,
    save_database_path,
    save_binchicken_singlem_metapackage,
    save_isolated_environment,
    save_magscot_directory,
    isolated_environment_config_path,
    software_runtime_issues,
)
from .discovery import (
    CoassemblyPlan,
    ReadSample,
    attach_contigs,
    build_analyses,
    build_bin_chicken_analyses,
    build_cohort_analyses,
    discover_contigs,
    discover_contigs_from_file,
    discover_mag_files,
    discover_reads,
    discover_reads_from_file,
    public_sample_name,
    read_multi_files,
)
from .executor import Executor
from .internal import gtdbtk_result_missing
from .model import Task
from .state import StateStore
class MetaBAWHelpFormatter(argparse.HelpFormatter):
    """Show a requirement marker or an explicit default for every option."""

    def _get_help_string(self, action: argparse.Action) -> str:
        text = action.help or "No additional description."
        if text == argparse.SUPPRESS:
            return text
        if not action.option_strings:
            return text
        if action.required:
            return text if "required" in text.lower() else f"{text} (required)"
        if "default:" in text.lower() or action.default is argparse.SUPPRESS:
            return text
        if action.default is None:
            default = "not set"
        elif action.default is True:
            default = "enabled"
        elif action.default is False:
            default = "disabled"
        elif isinstance(action.default, list):
            default = ", ".join(map(str, action.default)) or "empty"
        else:
            default = str(action.default)
        return f"{text} (default: {default})"


class _ExplicitOptionParser(argparse.ArgumentParser):
    """Preserve which CLI options were explicitly supplied by the user."""

    def parse_args(
        self,
        args: Sequence[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ) -> argparse.Namespace:
        raw_args = list(sys.argv[1:] if args is None else args)
        parsed = super().parse_args(raw_args, namespace)
        parsed._provided_options = {
            token.split("=", 1)[0]
            for token in raw_args
            if token.startswith("-")
        }
        return parsed


CORE_HELP_DESTS = {
    "bin": (
        "path", "input_reads_files", "contig", "input_contig_files",
        "suffix", "separate_sample_name", "contig_suffix", "output",
        "threads", "task", "max_memory", "assembly_strategy",
    ),
    "annotation": (
        "path", "reads", "suffix", "place_species", "read_suffix",
        "separate_sample_name", "methods", "kegg", "cazy", "hydrogenase",
        "gtdbtk_res", "output", "threads", "task", "max_memory",
    ),
}


DEFAULT_COMEBIN_BATCH_SIZE = 1024


def _multi_file_group_name(path: str | Path) -> str:
    """Derive a short deterministic reads-group label from --multi-files."""
    name = Path(path).name
    for suffix in (".tsv", ".txt", ".csv"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    name = re.sub(r"[._-]+multi[._-]*files?$", "", name, flags=re.IGNORECASE)
    name = "".join(char if char.isalnum() or char in "._-" else "_" for char in name)
    return name.strip("._-") or "reads"

# Conservative per-process CUDA budgets used to prevent GPU-capable binners
# from starting configurations that are known to be too small for reliable
# training. These are MetaBAW scheduling minima rather than hardware claims
# made by the upstream projects.
GPU_MINIMUM_MEMORY_GIB = {
    "vamb": 4.0,
    "semibin2": 4.0,
    # COMEBin is calculated dynamically from --batch-size.
    "comebin": 8.0,
    "lorbin": 4.0,
}
GPU_BINNER_LABELS = {
    "vamb": "VAMB",
    "semibin2": "SemiBin2",
    "comebin": "COMEBin",
    "lorbin": "LorBin",
}


def _page_text(text: str) -> None:
    """Show TEXT in a terminal pager, falling back to plain output."""
    if not sys.stdout.isatty():
        print(text, end="")
        return
    command = shlex.split(os.environ.get("PAGER", "less"))
    if Path(command[0]).name == "less":
        command.append("-R")
    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, text=True)
        process.communicate(text)
    except (OSError, ValueError):
        print(text, end="")


class _ShortHelpAction(argparse.Action):
    """Print usage and core options only, without paging."""

    def __init__(self, option_strings, dest="help", default=argparse.SUPPRESS, help=None):
        super().__init__(option_strings, dest, default=default, nargs=0, help=help)

    def __call__(self, parser, namespace, values, option_string=None):
        command_name = parser.prog.split()[-1]
        core = set(CORE_HELP_DESTS.get(command_name, ())) | {"help"}
        core_actions = [
            action
            for action in parser._actions
            if action.option_strings
            and action.help is not argparse.SUPPRESS
            and action.dest in core
        ]
        formatter = parser._get_formatter()
        if parser.description:
            formatter.add_text(parser.description)
        formatter.start_section("core options")
        for action in core_actions:
            formatter.add_argument(action)
        formatter.end_section()
        formatter.add_text("Use --full-help to page through every option.")
        print(formatter.format_help(), end="")
        parser.exit()


class _FullHelpAction(argparse.Action):
    """Page through the complete help like coverm does."""

    def __init__(self, option_strings, dest="full_help", default=argparse.SUPPRESS, help=None):
        super().__init__(option_strings, dest, default=default, nargs=0, help=help)

    def __call__(self, parser, namespace, values, option_string=None):
        _page_text(parser.format_help())
        parser.exit()


_PROCESS_STARTED_AT = time.monotonic()


def _elapsed_clock(started_at: float | None = None) -> str:
    origin = _PROCESS_STARTED_AT if started_at is None else started_at
    total = max(0, int(time.monotonic() - origin))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _log_failure_excerpt(
    path: Path, maximum_lines: int = 8, maximum_chars: int = 1200
) -> str:
    """Return the useful tail of a subprocess log for a top-level error."""
    try:
        lines = [
            line.strip()
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
            if line.strip() and not line.lstrip().startswith("# command:")
        ]
    except OSError:
        return ""
    excerpt = " | ".join(lines[-maximum_lines:])
    if len(excerpt) > maximum_chars:
        excerpt = "..." + excerpt[-(maximum_chars - 3) :]
    return excerpt


def _task_id_summary(task_ids: Sequence[str], maximum: int = 5) -> str:
    preview = ", ".join(task_ids[:maximum])
    remaining = len(task_ids) - maximum
    return preview + (f", +{remaining} more" if remaining > 0 else "")


def _main_cuda_status() -> CudaRuntimeStatus:
    return cuda_runtime_status("MetaBAW environment", (sys.executable,))


def _isolated_cuda_status(
    key: str,
    environment: str,
    frontend: str | None,
) -> CudaRuntimeStatus:
    spec = ISOLATED_TOOLS[key]
    command = (*isolated_run_prefix(environment, frontend), "python")
    return cuda_runtime_status(
        f"{spec.display_name} environment {environment}",
        command,
    )


def _offer_cuda_repair(
    status: CudaRuntimeStatus,
    key: str | None = None,
    environment: str | None = None,
    frontend: str | None = None,
) -> CudaRuntimeStatus:
    if status.available:
        return status
    target = (
        ISOLATED_TOOLS[key].display_name
        if key is not None
        else "the MetaBAW environment"
    )
    if not confirm_install(f"Install a CUDA-enabled PyTorch build for {target}"):
        return status
    if key is None:
        install_main_cuda_runtime()
        repaired = _main_cuda_status()
    else:
        if environment is None:
            raise RuntimeError(f"No environment was supplied for CUDA repair: {key}")
        install_isolated_cuda_runtime(key, environment, frontend)
        repaired = _isolated_cuda_status(key, environment, frontend)
    print("[CUDA RETEST] Verifying the repaired runtime.", flush=True)
    print_cuda_runtime_status(repaired)
    return repaired


def _selected_gpu_memory_mib(status: HostCudaStatus) -> int | None:
    if not status.devices:
        return None
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        first = visible.split(",", 1)[0].strip()
        for device in status.devices:
            if device.index == first:
                return device.memory_total_mib
    return status.devices[0].memory_total_mib


def _gpu_task_slots(
    status: HostCudaStatus,
    visible_device_count: int,
    memory_budget: str,
    maximum_tasks: int,
) -> int:
    """Allow at most one external training process per physical GPU."""
    if visible_device_count < 1 or maximum_tasks < 1:
        return 0
    # These external binners do not expose a reliable hard VRAM cap. Treating
    # a memory estimate as several virtual slots lets processes overcommit the
    # same device. Until each child process has explicit device affinity,
    # serialize GPU training across the visible device set.
    return 1


def _gpu_memory_gib(value: str, total_memory_mib: int | None = None) -> float | None:
    match = re.fullmatch(
        r"(\d+(?:\.\d+)?)([KMGTP]?i?B?|%)?",
        value,
        re.IGNORECASE,
    )
    if match is None:
        return None
    amount = float(match.group(1))
    unit = (match.group(2) or "").upper()
    if unit == "%":
        if total_memory_mib is None:
            return None
        return total_memory_mib / 1024 * amount / 100
    if not unit:
        return amount / 1024**3
    prefix = unit[0]
    power = {"K": 1, "M": 2, "G": 3, "T": 4, "P": 5}[prefix]
    return amount * 1024**power / 1024**3


def _option_was_provided(args: argparse.Namespace, option: str) -> bool:
    return option in getattr(args, "_provided_options", set())


def _thread_budget(
    total_threads: int,
    maximum_parallel_samples: int,
    workload_count: int,
) -> tuple[int, int]:
    """Return per-sample threads and the effective concurrent sample count."""
    concurrent = max(
        1,
        min(maximum_parallel_samples, max(1, workload_count), total_threads),
    )
    return max(1, total_threads // concurrent), concurrent


def _configure_thread_budget(
    args: argparse.Namespace,
    workload_count: int,
) -> int:
    per_sample, concurrent = _thread_budget(
        args.threads,
        args.task,
        workload_count,
    )
    args.threads_per_task = per_sample
    args.concurrent_sample_limit = concurrent
    allocated = per_sample * concurrent
    remainder = max(0, args.threads - allocated)
    remainder_text = (
        f"; {remainder} thread(s) remain available for scheduler overhead"
        if remainder
        else ""
    )
    print(
        f"[RESOURCES] -t/--threads={args.threads} is the total CPU budget; "
        f"up to {concurrent} concurrent sample task(s) receive "
        f"{per_sample} thread(s) each{remainder_text}.",
        flush=True,
    )
    return per_sample


def _record_parameter_warning(args: argparse.Namespace, message: str) -> None:
    warnings = getattr(args, "parameter_warnings", None)
    if warnings is None:
        warnings = []
        args.parameter_warnings = warnings
    if message not in warnings:
        warnings.append(message)
        print(f"[WARNING] {message}", file=sys.stderr, flush=True)


def _resolve_bin_option_priorities(args: argparse.Namespace) -> None:
    """Apply documented soft-option priorities before workflow planning."""
    args.parameter_warnings = []
    if args.gpu is False and _option_was_provided(args, "--max-gpu-memory"):
        _record_parameter_warning(
            args,
            "--no-gpu takes precedence over --max-gpu-memory; the explicitly "
            f"supplied GPU memory budget ({args.max_gpu_memory}) is ignored.",
        )
    if args.environment and "semibin2" not in args.tools:
        _record_parameter_warning(
            args,
            f"--environment {args.environment!r} is effective only when "
            "SemiBin2 is selected with --tools; it is ignored for the current "
            "binner selection.",
        )
        args.environment = None
    elif args.environment:
        print(
            f"[CONFIG] --environment {args.environment!r} will be passed to "
            "SemiBin2 for compatible per-sample runs.",
            flush=True,
        )


def _comebin_minimum_memory_gib(batch_size: int) -> float:
    """Return MetaBAW's conservative COMEBin VRAM estimate for a batch."""
    return max(1.0, batch_size / 128.0)


def _gpu_minimum_memory_gib(tool: str, args: argparse.Namespace) -> float:
    if tool == "comebin":
        requested = getattr(args, "requested_batch_size", None)
        if requested is None:
            requested = args.batch_size
        return _comebin_minimum_memory_gib(requested)
    return GPU_MINIMUM_MEMORY_GIB[tool]


def _apply_comebin_gpu_budget(
    args: argparse.Namespace,
    total_memory_mib: int | None = None,
    emit: bool = True,
) -> None:
    requested = getattr(args, "requested_batch_size", None)
    if requested is None:
        requested = args.batch_size
    args.requested_batch_size = requested
    args.batch_size = requested
    gpu_binners = set(getattr(args, "gpu_binners", args.tools))
    if not args.gpu or "comebin" not in gpu_binners:
        return
    if emit:
        print(
            f"[GPU] COMEBin batch size {requested} meets the conservative "
            f"{_comebin_minimum_memory_gib(requested):g} GiB minimum for the "
            f"{args.max_gpu_memory} per-task memory budget.",
            flush=True,
        )


def _apply_gpu_memory_minima(
    args: argparse.Namespace,
    total_memory_mib: int | None,
) -> tuple[str, ...]:
    """Select GPU binners whose requested VRAM budget meets safe minima."""
    selected = [tool for tool in args.tools if tool in GPU_MINIMUM_MEMORY_GIB]
    memory_gib = _gpu_memory_gib(args.max_gpu_memory, total_memory_mib)
    if memory_gib is None:
        args.gpu_binners = tuple(selected)
        return tuple(selected)
    enabled: list[str] = []
    for tool in selected:
        minimum = _gpu_minimum_memory_gib(tool, args)
        if memory_gib + 1e-9 < minimum:
            requested_batch = getattr(args, "requested_batch_size", None)
            if requested_batch is None:
                requested_batch = args.batch_size
            batch_detail = (
                f" at --batch-size {requested_batch}" if tool == "comebin" else ""
            )
            _record_parameter_warning(
                args,
                f"--max-gpu-memory {args.max_gpu_memory} resolves to "
                f"{memory_gib:g} GiB, below the MetaBAW minimum of "
                f"{minimum:g} GiB for {GPU_BINNER_LABELS[tool]}"
                f"{batch_detail}; "
                f"{GPU_BINNER_LABELS[tool]} will use CPU for this run.",
            )
            continue
        enabled.append(tool)
    args.gpu_binners = tuple(enabled)
    return tuple(enabled)


def _enable_bin_cuda(
    args: argparse.Namespace,
    isolated_status: dict[str, IsolatedEnvironmentStatus],
) -> bool:
    """Enable CUDA independently for each eligible selected binner."""
    host = host_cuda_status()
    print_host_cuda_status(host)
    if not host.available:
        args.gpu = False
        args.cuda_device_count = 0
        args.cuda_task_slots = 0
        detail = host.error or "unknown error"
        print(
            f"[GPU AUTO] CUDA is unavailable ({detail}); using CPU execution.",
            flush=True,
        )
        return False
    selected_gpu_tools = [
        tool for tool in args.tools if tool in GPU_MINIMUM_MEMORY_GIB
    ]
    if not selected_gpu_tools:
        print(
            "[GPU] None of the selected binners has a GPU execution path; "
            "using CPU execution.",
            flush=True,
        )
        args.gpu = False
        args.cuda_device_count = 0
        args.cuda_task_slots = 0
        return False
    gpu_binners = _apply_gpu_memory_minima(
        args,
        _selected_gpu_memory_mib(host),
    )
    if not gpu_binners:
        args.gpu = False
        args.cuda_device_count = 0
        args.cuda_task_slots = 0
        print(
            "[GPU AUTO] The configured GPU memory budget is below the "
            "minimum for every selected GPU-capable binner; using CPU "
            "execution.",
            flush=True,
        )
        return False
    runtime_entries: list[
        tuple[CudaRuntimeStatus, tuple[str, ...]]
    ] = []
    main_tools = tuple(
        tool for tool in ("vamb", "semibin2") if tool in gpu_binners
    )
    if main_tools:
        runtime_entries.append((_main_cuda_status(), main_tools))
    for key in ("comebin", "lorbin"):
        if key not in gpu_binners:
            continue
        environment_status = isolated_status.get(key)
        if environment_status is None or not environment_status.available:
            continue
        runtime_entries.append(
            (
                _isolated_cuda_status(
                    key,
                    getattr(args, f"{key}_env"),
                    environment_status.frontend,
                ),
                (key,),
            )
        )
    if not runtime_entries:
        print(
            "[GPU] None of the selected binners has a GPU execution path; "
            "using CPU execution.",
            flush=True,
        )
        args.gpu = False
        args.cuda_device_count = 0
        args.cuda_task_slots = 0
        return False
    enabled = set(gpu_binners)
    runtime_statuses: list[CudaRuntimeStatus] = []
    for status, tools in runtime_entries:
        print_cuda_runtime_status(status)
        if status.available:
            runtime_statuses.append(status)
            continue
        enabled.difference_update(tools)
        labels = ", ".join(GPU_BINNER_LABELS[tool] for tool in tools)
        _record_parameter_warning(
            args,
            f"CUDA runtime is unavailable for {labels} "
            f"({status.error or 'CUDA allocation failed'}); "
            f"{labels} will use CPU for this run.",
        )
    gpu_binners = tuple(tool for tool in args.tools if tool in enabled)
    args.gpu_binners = gpu_binners
    if not gpu_binners:
        args.gpu = False
        args.cuda_device_count = 0
        args.cuda_task_slots = 0
        print(
            "[GPU AUTO] No selected binner has both sufficient GPU memory "
            "and a usable CUDA runtime; using CPU execution.",
            flush=True,
        )
        return False
    args.gpu = True
    args.cuda_device_count = min(
        len(host.devices),
        *(status.device_count for status in runtime_statuses),
    )
    args.cuda_task_slots = _gpu_task_slots(
        host,
        args.cuda_device_count,
        args.max_gpu_memory,
        args.task,
    )
    print(
        f"[GPU] {args.cuda_device_count} visible device(s) provide "
        f"{args.cuda_task_slots} safe concurrent GPU training task slot(s); "
        f"eligibility budget={args.max_gpu_memory}; GPU binners="
        f"{','.join(gpu_binners)}.",
        flush=True,
    )
    _apply_comebin_gpu_budget(args, _selected_gpu_memory_mib(host))
    return True


def _task_dict(task: Task) -> dict[str, object]:
    return {
        "id": task.id,
        "stage": task.stage,
        "description": task.description,
        "cpus": task.cpus,
        "gpus": task.gpus,
        "priority": task.priority,
        "allow_failed_dependencies": task.allow_failed_deps,
        "failure_tolerated": task.failure_tolerated,
        "memory_limit_enforced": task.enforce_memory_limit,
        "per_process_address_limit_enforced": task.enforce_memory_limit,
        "minimum_memory_gb": task.minimum_memory_gb,
        "memory_requirement_hint": task.memory_requirement_hint,
        "environment": dict(sorted(task.env.items())),
        "dependencies": list(task.deps),
        "wait_for": list(task.wait_for),
        "inputs": [str(path) for path in task.inputs],
        "outputs": [str(path) for path in task.outputs],
        "output_alternatives": [
            [str(path) for path in output_set]
            for output_set in task.output_alternatives
        ],
        "fasta_output_directories": [
            str(path) for path in task.fasta_output_dirs
        ],
        "sample": task.sample,
        "automatic_retries": task.automatic_retries,
        "command": task.display_command(),
    }


def _parse_advanced(values: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--advanced-arg must be TOOL=ARGUMENTS")
        tool, arguments = value.split("=", 1)
        tool = tool.strip().lower()
        if not tool or tool in parsed:
            raise ValueError(f"Empty or duplicate --advanced-arg tool: {tool!r}")
        parsed[tool] = arguments.strip()
    return parsed


def _temporary_path(value: str | None, output: Path) -> Path:
    if value is None:
        return output / "tmp"
    requested = Path(value).expanduser()
    return (requested if requested.is_absolute() else output / requested).resolve()


def _safe_to_delete_temp(temp_files: Path, output: Path) -> bool:
    try:
        output.relative_to(temp_files)
    except ValueError:
        return temp_files != output
    return False


def _option_was_provided(args: argparse.Namespace, *option_names: str) -> bool:
    provided = getattr(args, "_provided_options", None)
    return provided is not None and any(name in provided for name in option_names)


def _validate_metawrap_binner_limit(
    refinement: str,
    binners: Sequence[str],
) -> None:
    """Reject MetaWRAP input sets that exceed its three-binner limit."""
    if refinement != "metawrap" or len(binners) <= 3:
        return
    selected = ", ".join(binners)
    raise ValueError(
        "--refinement metawrap accepts results from at most 3 binning tools, "
        f"but {len(binners)} were selected: {selected}. Reduce --tools to 3 "
        "or fewer, or select --refinement magscot/das_tool."
    )


def _ensure_software(
    requirements: list[SoftwareRequirement],
    workflow: str,
) -> None:
    missing = missing_software(requirements)
    missing_r = _missing_r_dependency_groups(requirements)
    runtime_issues = software_runtime_issues(requirements)
    if not missing and not missing_r and not runtime_issues:
        return
    if missing:
        print(f"Missing software required for {workflow}:", file=sys.stderr, flush=True)
        print_software_status(missing)
    for profile, packages in missing_r.items():
        print(
            f"Missing {profile} R packages: " + ", ".join(packages),
            file=sys.stderr,
            flush=True,
        )
    for issue in runtime_issues:
        print(
            f"Broken {issue.executable} runtime: {issue.detail}",
            file=sys.stderr,
            flush=True,
        )
    if confirm_install("Install the missing software now"):
        remaining = install_software(requirements)
        remaining_r = _missing_r_dependency_groups(requirements)
        remaining_runtime = software_runtime_issues(requirements)
        if not remaining and not remaining_r and not remaining_runtime:
            return
        missing = remaining
        missing_r = remaining_r
        runtime_issues = remaining_runtime
    names = ", ".join(
        [
            *(requirement.executable for requirement in missing),
            *(
                f"{profile}:R:{name}"
                for profile, packages in missing_r.items()
                for name in packages
            ),
            *(f"{issue.executable}:runtime" for issue in runtime_issues),
        ]
    )
    raise RuntimeError(
        f"Missing required software for the selected {workflow} options: {names}. "
        "Install the listed software, run `metabaw check --all`, or rerun this "
        "command in an interactive Conda/Mamba environment and approve installation."
    )


def _ensure_isolated_tool(
    spec: IsolatedToolSpec,
    environment: str,
    install_immediately: bool = False,
) -> IsolatedEnvironmentStatus:
    status = isolated_environment_status(spec, environment)
    print_isolated_status(status)
    if status.available:
        return status
    action = "Repair" if status.python_version is not None else "Install"
    install = install_immediately or confirm_install(
        f"{action} {spec.display_name} in isolated Python {spec.python_version} "
        f"environment {environment!r}"
    )
    if install:
        return install_isolated_environment(spec, environment, status.frontend)
    raise RuntimeError(
        f"{spec.display_name} environment {environment!r} is unavailable: "
        f"{status.error or 'unknown error'}. Run `metabaw check --scope {spec.key} "
        f"{spec.option} {environment}` in an interactive terminal."
    )


def _check_isolated_tools_interactively(
    selected: Sequence[str],
    environments: dict[str, str],
) -> tuple[dict[str, IsolatedEnvironmentStatus], bool]:
    """Inspect and, when approved, install every selected isolated environment."""
    statuses: dict[str, IsolatedEnvironmentStatus] = {}
    for key in selected:
        status = isolated_environment_status(ISOLATED_TOOLS[key], environments[key])
        statuses[key] = status
        print_isolated_status(status)

    unavailable = [key for key in selected if not statuses[key].available]
    if not unavailable:
        return statuses, True

    names = ", ".join(
        f"{ISOLATED_TOOLS[key].display_name} ({environments[key]})"
        for key in unavailable
    )
    print(
        "Missing selected isolated software: " + names,
        file=sys.stderr,
        flush=True,
    )
    if not confirm_install(
        "Install or repair all missing selected isolated environments now"
    ):
        print(
            "Installation declined; the missing isolated software remains unavailable.",
            file=sys.stderr,
            flush=True,
        )
        return statuses, False

    total = len(unavailable)
    for number, key in enumerate(unavailable, start=1):
        spec = ISOLATED_TOOLS[key]
        environment = environments[key]
        print(
            f"[ENV INSTALL {number}/{total}] {spec.display_name}: {environment}",
            flush=True,
        )
        try:
            statuses[key] = install_isolated_environment(
                spec,
                environment,
                statuses[key].frontend,
            )
        except Exception as exc:
            print(
                f"[ENV INSTALL FAILED {number}/{total}] {spec.display_name}: {exc}. "
                "Continuing with the remaining isolated environments.",
                file=sys.stderr,
                flush=True,
            )
            statuses[key] = isolated_environment_status(spec, environment)

    remaining = [key for key in selected if not statuses[key].available]
    if remaining:
        print(
            "Isolated software remains missing after all installation attempts: "
            + ", ".join(ISOLATED_TOOLS[key].display_name for key in remaining),
            file=sys.stderr,
            flush=True,
        )
        return statuses, False
    return statuses, True


def _require_database(
    name: str,
    current: Path | None,
    installer: object,
    default_directory: Path,
    size_warning: str,
    validator: Callable[[Path], bool] | None = None,
) -> Path:
    valid = validator or (lambda path: path.exists())
    if current is not None and valid(current):
        return current
    detail = str(current) if current is not None else "not configured"
    print(f"[MISSING] {name} database: {detail}", file=sys.stderr, flush=True)
    if confirm_install(f"Install {name} database under {default_directory} ({size_warning})"):
        installed = installer(default_directory)
        if not valid(installed):
            raise RuntimeError(
                f"{name} database installation completed without all required files: "
                f"{installed}"
            )
        return installed
    raise RuntimeError(
        f"{name} database is missing. Configure its path or install it before running."
    )


def _prompt_database_path(
    name: str,
    current: Path | None,
    validator: Callable[[Path], bool],
    config_name: str,
    *,
    artifact_suffix: str | None = None,
    missing_reasons: Callable[[Path], Sequence[object]] | None = None,
) -> Path | None:
    """Prompt for a reusable database path when an interactive check finds none."""
    if current is not None and validator(current):
        return current
    if not sys.stdin.isatty():
        return current
    while True:
        try:
            value = input(
                f"{name} database path (press Enter to leave it unset): "
            ).strip()
        except EOFError:
            return current
        if not value:
            return current
        candidate = configured_database_path(
            value,
            "",
            artifact_suffix=artifact_suffix,
            validator=validator,
        )
        if candidate is not None and validator(candidate):
            save_database_path(config_name, candidate)
            print(f"[CONFIG] Saved {name} database: {candidate}", flush=True)
            return candidate
        checked_path = candidate or Path(value).expanduser()
        reasons = tuple(missing_reasons(checked_path)) if missing_reasons else ()
        detail = (
            f" Missing/invalid: {', '.join(map(str, reasons))}."
            if reasons
            else ""
        )
        print(
            f"[INVALID] {name} database path: {checked_path}.{detail} "
            "Enter the database root/file again, or press Enter to skip.",
            file=sys.stderr,
            flush=True,
        )


def _preflight_bin(args: argparse.Namespace) -> None:
    _ensure_software(bin_requirements(args), "bin")
    isolated_status: dict[str, IsolatedEnvironmentStatus] = {}
    if getattr(args, "assembly_strategy", "default") == "bin-chicken":
        status = _ensure_isolated_tool(
            ISOLATED_TOOLS["binchicken"],
            getattr(
                args,
                "binchicken_env",
                configured_isolated_environment("binchicken"),
            ),
        )
        args.binchicken_frontend = status.frontend
    if "comebin" in args.tools:
        status = _ensure_isolated_tool(ISOLATED_TOOLS["comebin"], args.comebin_env)
        isolated_status["comebin"] = status
        args.comebin_frontend = status.frontend
    if "lorbin" in args.tools:
        status = _ensure_isolated_tool(ISOLATED_TOOLS["lorbin"], args.lorbin_env)
        isolated_status["lorbin"] = status
        args.lorbin_frontend = status.frontend
    if args.refinement == "metawrap":
        status = _ensure_isolated_tool(ISOLATED_TOOLS["metawrap"], args.metawrap_env)
        args.metawrap_frontend = status.frontend
    if args.quality_control == "checkm2":
        status = _ensure_isolated_tool(ISOLATED_TOOLS["checkm2"], args.checkm2_env)
        args.checkm2_frontend = status.frontend
    if args.gpu is not False:
        _enable_bin_cuda(args, isolated_status)
    cache = Path.home() / ".cache" / "metabaw" / "databases"
    if args.refinement == "magscot":
        directory = Path(args.magscot_dir).expanduser().resolve()
        missing = [path for path in magscot_files(directory) if not path.is_file()]
        if missing:
            print(f"[MISSING] MAGScoT files: {', '.join(map(str, missing))}", file=sys.stderr)
            if confirm_install(f"Install MAGScoT under {directory}"):
                install_magscot(directory)
            else:
                raise RuntimeError(f"MAGScoT installation is incomplete: {directory}")
        args.magscot_dir = str(save_magscot_directory(directory))
    needs_checkm_database = (
        args.quality_control == "checkm"
        or args.refinement == "metawrap"
        or args.dereplication_tool == "drep"
    )
    if needs_checkm_database:
        checkm = configured_database_path(
            None,
            "CHECKM_DATA_PATH",
            "checkm",
            validator=checkm_database_valid,
        )
        if checkm is None or not checkm_database_valid(checkm):
            raise RuntimeError(
                "CheckM database is missing or incomplete. Configure it with "
                "`metabaw check --all --checkm-db PATH`."
            )
        os.environ["CHECKM_DATA_PATH"] = str(checkm)
    if args.quality_control == "checkm2":
        checkm2_runner = isolated_run_prefix(
            args.checkm2_env,
            args.checkm2_frontend,
        )
        configured = configured_database_path(
            args.checkm2_db,
            "CHECKM2DB",
            "checkm2",
            artifact_suffix=".dmnd",
        )
        database = _require_database(
            "CheckM2",
            configured,
            lambda directory: install_checkm2_database(directory, checkm2_runner),
            cache / "checkm2",
            "several GB",
            diamond_database_valid,
        )
        if args.checkm2_db or database != configured:
            save_database_path("checkm2", database)
        args.checkm2_db = str(database)
    if args.gunc:
        configured = configured_database_path(
            args.gunc_db,
            "GUNC_DB",
            "gunc",
            artifact_suffix=".dmnd",
        )
        database = _require_database(
            "GUNC",
            configured,
            install_gunc_database,
            cache / "gunc",
            "approximately 13 GB",
            diamond_database_valid,
        )
        if args.gunc_db or database != configured:
            save_database_path("gunc", database)
        args.gunc_db = str(database)
    if args.trna or args.rrna:
        configured = configured_database_path(
            args.gtdbtk_data,
            "GTDBTK_DATA_PATH",
            "gtdbtk",
            validator=gtdbtk_database_valid,
        )
        database = _require_database(
            "GTDB-Tk",
            configured,
            install_gtdbtk_database,
            cache / "gtdbtk",
            "approximately 100 GB",
            gtdbtk_database_valid,
        )
        if args.gtdbtk_data or database != configured:
            save_database_path("gtdbtk", database)
        args.gtdbtk_data = str(database)


def _bin_chicken_plan_path(output: Path) -> Path:
    return output / "coassembly" / "binchicken" / "coassemble" / "target" / "elusive_clusters.tsv"


def _parse_bin_chicken_plan(path: Path) -> list[CoassemblyPlan]:
    """Read the stable public Bin Chicken coassembly plan table."""
    if not path.is_file():
        raise RuntimeError(
            "Bin Chicken completed without its coassembly plan: " + str(path)
        )
    plans: list[CoassemblyPlan] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = set(reader.fieldnames or ())
        required = {"samples", "recover_samples", "coassembly"}
        if not required <= fields:
            raise ValueError(
                f"Unexpected Bin Chicken plan columns in {path}: {reader.fieldnames}; "
                f"required={sorted(required)}"
            )
        for line_number, row in enumerate(reader, start=2):
            identifier = (row.get("coassembly") or "").strip()
            assembly = tuple(
                item.strip()
                for item in (row.get("samples") or "").split(",")
                if item.strip()
            )
            recovery = tuple(
                item.strip()
                for item in (row.get("recover_samples") or "").split(",")
                if item.strip()
            )
            if not identifier or not assembly:
                raise ValueError(
                    f"Invalid Bin Chicken plan row {line_number} in {path}"
                )
            plans.append(CoassemblyPlan(identifier, assembly, recovery))
    return plans


def _stage_bin_chicken_reads(
    reads: list[ReadSample], input_root: Path
) -> tuple[list[Path], list[Path], dict[str, str]]:
    """Stage stable paired-read aliases so planner IDs map back to MetaBAW."""
    if input_root.exists():
        shutil.rmtree(input_root)
    input_root.mkdir(parents=True, exist_ok=True)
    forward: list[Path] = []
    reverse: list[Path] = []
    aliases: dict[str, str] = {}
    for index, sample in enumerate(reads, start=1):
        if sample.read2 is None:
            raise ValueError(
                "--assembly-strategy bin-chicken requires paired short reads; "
                "every sample must have read 1 and read 2"
            )
        alias = f"metabaw_{index:05d}"
        aliases[alias] = sample.name
        read1 = input_root / f"{alias}_R1.fastq.gz"
        read2 = input_root / f"{alias}_R2.fastq.gz"
        for source, destination in ((sample.read1, read1), (sample.read2, read2)):
            try:
                destination.symlink_to(source)
            except OSError:
                try:
                    os.link(source, destination)
                except OSError:
                    shutil.copy2(source, destination)
        forward.append(read1)
        reverse.append(read2)
    return forward, reverse, aliases


def _publish_bin_chicken_plan(
    source: Path, destination: Path, aliases: dict[str, str]
) -> None:
    """Write a user-facing plan containing original MetaBAW sample names."""
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise RuntimeError(f"Bin Chicken plan has no header: {source}")
        rows = list(reader)

    def translate(value: str | None) -> str:
        return ",".join(
            aliases.get(item.strip(), item.strip())
            for item in (value or "").split(",")
            if item.strip()
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            row["samples"] = translate(row.get("samples"))
            row["recover_samples"] = translate(row.get("recover_samples"))
            writer.writerow(row)


def _run_bin_chicken_planner(
    args: argparse.Namespace,
    reads: list[ReadSample],
    output: Path,
    temp_files: Path,
) -> tuple[list[CoassemblyPlan], Path]:
    """Run only Bin Chicken's selection stage; Aviary is expressly omitted."""
    if args.type != "short":
        raise ValueError(
            "--assembly-strategy bin-chicken currently requires --type short. "
            "Upstream Bin Chicken coassemble does not support long reads."
        )
    if len(reads) < 2:
        raise ValueError(
            "--assembly-strategy bin-chicken requires at least two paired-end samples"
        )
    metapackage = configured_binchicken_singlem_metapackage()
    if metapackage is None or not reference_package_valid(metapackage):
        raise RuntimeError(
            "Bin Chicken planning requires a valid SingleM metapackage. Configure "
            "it once with `metabaw check --scope binchicken "
            "--binchicken-singlem-metapackage PATH`."
        )
    plan_path = _bin_chicken_plan_path(output)
    if plan_path.is_file() and not args.force:
        plans = _parse_bin_chicken_plan(plan_path)
        print(
            f"[{_elapsed_clock()}] [RESUME] Reusing Bin Chicken coassembly plan: "
            f"{plan_path}",
            flush=True,
        )
        return plans, plan_path

    plan_root = temp_files / "work" / "binchicken"
    if plan_root.exists():
        shutil.rmtree(plan_root)
    temp_system = temp_files / "system_tmp"
    log_dir = temp_files / "runtime" / "logs"
    temp_system.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    forward, reverse, aliases = _stage_bin_chicken_reads(reads, plan_root / "inputs")
    forward_list = plan_root / "inputs" / "forward_reads.txt"
    reverse_list = plan_root / "inputs" / "reverse_reads.txt"
    forward_list.write_text(
        "".join(f"{path}\n" for path in forward), encoding="utf-8"
    )
    reverse_list.write_text(
        "".join(f"{path}\n" for path in reverse), encoding="utf-8"
    )
    binchicken_environment = getattr(
        args,
        "binchicken_env",
        configured_isolated_environment("binchicken"),
    )
    command = [
        *isolated_run_prefix(
            binchicken_environment,
            getattr(args, "binchicken_frontend", None),
            prefer_lock_free_runner=True,
        ),
        "binchicken",
        "coassemble",
        "--forward-list",
        str(forward_list),
        "--reverse-list",
        str(reverse_list),
        "--output",
        str(plan_root),
        "--singlem-metapackage",
        str(metapackage),
        "--cores",
        str(args.threads),
        "--local-cores",
        str(args.threads),
        "--tmp-dir",
        str(temp_system),
    ]
    log_path = log_dir / "00.binchicken.plan.log"
    print(
        f"[{_elapsed_clock()}] [PLAN] Running Bin Chicken to select coassembly "
        f"and recovery samples; Aviary is disabled; log={log_path}",
        flush=True,
    )
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"# command: {shlex.join(command)}\n")
        handle.flush()
        completed = subprocess.run(
            command,
            check=False,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if completed.returncode != 0:
        excerpt = _log_failure_excerpt(log_path)
        detail = f"; last output: {excerpt}" if excerpt else ""
        raise RuntimeError(
            "Bin Chicken planning failed with exit code "
            f"{completed.returncode}{detail}; log={log_path}"
        )
    source_plan = plan_root / "coassemble" / "target" / "elusive_clusters.tsv"
    if not source_plan.is_file():
        excerpt = _log_failure_excerpt(log_path)
        detail = f"; last output: {excerpt}" if excerpt else ""
        raise RuntimeError(
            "Bin Chicken exited successfully but did not create the expected plan "
            f"{source_plan}{detail}; log={log_path}"
        )
    _publish_bin_chicken_plan(source_plan, plan_path, aliases)
    plans = _parse_bin_chicken_plan(plan_path)
    if plans:
        print(
            f"[{_elapsed_clock()}] [PLAN] Bin Chicken selected {len(plans)} "
            "coassembly group(s); MetaBAW will perform assembly and binning.",
            flush=True,
        )
    else:
        print(
            f"[{_elapsed_clock()}] [PLAN] Bin Chicken selected no coassembly groups; "
            "MetaBAW will retain the default per-sample workflow.",
            flush=True,
        )
    return plans, plan_path


def _preflight_annotation(args: argparse.Namespace) -> None:
    _ensure_software(annotation_requirements(args), "annotation")
    cache = Path.home() / ".cache" / "metabaw" / "databases"
    if args.gtdbtk_res:
        args.gtdbtk_data = None
    else:
        configured = configured_database_path(
            args.gtdbtk_data,
            "GTDBTK_DATA_PATH",
            "gtdbtk",
            validator=gtdbtk_database_valid,
        )
        database = _require_database(
            "GTDB-Tk",
            configured,
            install_gtdbtk_database,
            cache / "gtdbtk",
            "approximately 100 GB",
            gtdbtk_database_valid,
        )
        if args.gtdbtk_data or database != configured:
            save_database_path("gtdbtk", database)
        args.gtdbtk_data = str(database)
    if args.kegg:
        configured = configured_database_path(
            getattr(args, "kegg_db", None),
            "KOFAM_DB",
            "kofam",
            validator=kofam_database_valid,
        )
        database = _require_database(
            "KOfam/KEGG",
            configured,
            install_kofam_database,
            cache / "kofam",
            "approximately 10 GB",
            kofam_database_valid,
        )
        if getattr(args, "kegg_db", None) or database != configured:
            save_database_path("kofam", database)
        args.kegg_db = str(database)
    if args.cazy:
        configured = configured_database_path(
            getattr(args, "dbcan_db", None),
            "DBCAN_DB",
            "dbcan",
            validator=dbcan_database_valid,
        )
        database = _require_database(
            "dbCAN/CAZy",
            configured,
            install_dbcan_database,
            cache / "dbcan",
            "several GB",
            dbcan_database_valid,
        )
        if getattr(args, "dbcan_db", None) or database != configured:
            save_database_path("dbcan", database)
        args.dbcan_db = str(database)
    if args.hydrogenase:
        hydrogenase_db = configured_database_path(
            getattr(args, "hydrogenase_db", None),
            "HYDROGENASE_DB",
            "hydrogenase",
            validator=hydrogenase_database_valid,
        )
        if hydrogenase_db is None or not hydrogenase_database_valid(hydrogenase_db):
            detail = (
                "; missing="
                + ", ".join(hydrogenase_database_missing(hydrogenase_db))
                if hydrogenase_db is not None
                else ""
            )
            raise RuntimeError(
                "Hydrogenase database is missing or invalid"
                f"{detail}. Configure it with `metabaw check --scope hydrogenase "
                "--hydrogenase-db PATH`."
            )
        args.hydrogenase_db = str(hydrogenase_db)


def _default_bin_dependency_profile() -> argparse.Namespace:
    return argparse.Namespace(
        align_tool="bowtie2",
        tools=["metabat2", "metadecoder", "vamb"],
        type="short",
        refinement="magscot",
        quality_control="checkm2",
        gunc=False,
        trna=False,
        rrna=False,
        dereplication_tool="galah",
    )


def _merge_requirements(
    *groups: list[SoftwareRequirement],
) -> list[SoftwareRequirement]:
    merged: list[SoftwareRequirement] = []
    seen: set[str] = set()
    for group in groups:
        for requirement in group:
            if requirement.executable not in seen:
                merged.append(requirement)
                seen.add(requirement.executable)
    return merged


def _missing_r_dependency_groups(
    requirements: list[SoftwareRequirement],
) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    if any("MAGScoT" in requirement.purpose for requirement in requirements):
        missing = missing_magscot_r_packages()
        if missing:
            groups["MAGScoT"] = missing
    if any(
        requirement.executable == "DAS_Tool"
        or "DAS Tool" in requirement.purpose
        for requirement in requirements
    ):
        missing = missing_dastool_r_packages()
        if missing:
            groups["DAS Tool"] = missing
    return groups


def _software_problem_names(
    missing: Sequence[SoftwareRequirement],
    missing_r: dict[str, list[str]],
    runtime_issues: Sequence[SoftwareRuntimeIssue],
) -> list[str]:
    return [
        *(requirement.executable for requirement in missing),
        *(
            f"{profile}:R:{package}"
            for profile, packages in missing_r.items()
            for package in packages
        ),
        *(f"{issue.executable}:runtime" for issue in runtime_issues),
    ]


def _check_software_interactively(
    requirements: list[SoftwareRequirement],
) -> bool:
    print_software_status(requirements)
    missing = missing_software(requirements)
    missing_r = _missing_r_dependency_groups(requirements)
    runtime_issues = software_runtime_issues(requirements)
    for profile, packages in missing_r.items():
        for package in packages:
            print(
                f"[MISSING] R package {package:<24} required by {profile}",
                flush=True,
            )
    for issue in runtime_issues:
        print(
            f"[BROKEN] {issue.executable:<34} {issue.detail}",
            flush=True,
        )
    if not missing and not missing_r and not runtime_issues:
        return True
    names = _software_problem_names(missing, missing_r, runtime_issues)
    print(
        "Missing selected software: " + ", ".join(names),
        file=sys.stderr,
        flush=True,
    )
    if not confirm_install("Install the missing selected software now"):
        print(
            "Installation declined; the missing software remains unavailable.",
            file=sys.stderr,
            flush=True,
        )
        return False
    previous = tuple(sorted(names))
    maximum_passes = 4
    for pass_number in range(1, maximum_passes + 1):
        if pass_number > 1:
            print(
                f"[INSTALL PASS {pass_number}/{maximum_passes}] Repairing "
                "dependencies detected after earlier installations.",
                flush=True,
            )
        install_software(requirements)
        remaining = missing_software(requirements)
        remaining_r = _missing_r_dependency_groups(requirements)
        remaining_runtime = software_runtime_issues(requirements)
        if not remaining and not remaining_r and not remaining_runtime:
            return True
        names = _software_problem_names(
            remaining,
            remaining_r,
            remaining_runtime,
        )
        current = tuple(sorted(names))
        if current == previous:
            break
        newly_detected = sorted(set(current) - set(previous))
        if newly_detected:
            print(
                "[RECHECK] Newly detectable dependencies: "
                + ", ".join(newly_detected),
                flush=True,
            )
        previous = current
    print(
        "Software remains missing after all installation passes: "
        + ", ".join(names),
        file=sys.stderr,
        flush=True,
    )
    return False


def command_check(args: argparse.Namespace) -> int:
    isolated_scopes = set(ISOLATED_TOOLS)
    explicitly_selected_isolated = [
        key
        for key in ISOLATED_TOOLS
        if _option_was_provided(
            args,
            f"--{key}-env",
            f"--{key}-environment",
        )
    ]
    selected_all = args.all
    explicit_essential = _option_was_provided(args, "-e", "--essential")
    explicit_binchicken_metapackage = _option_was_provided(
        args, "--binchicken-singlem-metapackage"
    )
    inferred_isolated_scope = (
        args.scope is None
        and not selected_all
        and bool(explicitly_selected_isolated)
        and not explicit_essential
    )
    inferred_binchicken_scope = (
        args.scope is None
        and not selected_all
        and explicit_binchicken_metapackage
        and not explicit_essential
    )
    selected_essential = (
        not selected_all
        and args.scope is None
        and args.essential
        and not inferred_isolated_scope
        and not inferred_binchicken_scope
    )
    if selected_all:
        requirements = all_requirements()
    elif args.scope == "annotation":
        requirements = annotation_requirements()
    elif args.scope == "kegg":
        requirements = [SOFTWARE[name] for name in ("bash", "prodigal", "exec_annotation")]
    elif args.scope == "cazy":
        requirements = [SOFTWARE[name] for name in ("bash", "prodigal", "run_dbcan")]
    elif args.scope == "hydrogenase":
        requirements = [
            SOFTWARE[name]
            for name in ("bash", "prodigal", "blastp", "makeblastdb", "diamond")
        ]
    elif args.scope == "binchicken" or inferred_binchicken_scope:
        requirements = [SOFTWARE["megahit"]]
    elif args.scope == "checkm":
        requirements = [SOFTWARE["checkm"]]
    elif args.scope == "minibwa":
        requirements = [SOFTWARE["minibwa"]]
    elif args.scope in isolated_scopes:
        requirements = []
    elif inferred_isolated_scope:
        requirements = []
    elif selected_essential:
        requirements = _merge_requirements(
            bin_requirements(_default_bin_dependency_profile()),
            annotation_requirements(),
        )
    else:
        requirements = bin_requirements(_default_bin_dependency_profile())
    if any(
        requirement.executable == "gtdbtk" for requirement in requirements
    ) and args.gtdbtk_data:
        requested_gtdbtk = Path(args.gtdbtk_data).expanduser().resolve()
        if gtdbtk_database_valid(requested_gtdbtk):
            # Ensure the runtime probe uses the path supplied to this check,
            # rather than a stale Conda environment variable.
            os.environ["GTDBTK_DATA_PATH"] = str(requested_gtdbtk)
    all_available = _check_software_interactively(requirements)

    selected_isolated: list[str] = []
    if selected_all:
        selected_isolated = list(ISOLATED_TOOLS)
    elif selected_essential or args.scope == "bin":
        selected_isolated = ["checkm2"]
    elif args.scope in isolated_scopes:
        selected_isolated = [args.scope]
    for key in explicitly_selected_isolated:
        if key not in selected_isolated:
            selected_isolated.append(key)
    isolated_environments: dict[str, str] = {}
    for key in selected_isolated:
        requested_environment = getattr(args, f"{key}_env")
        environment = normalize_isolated_environment(
            key,
            requested_environment,
        )
        if environment != requested_environment:
            print(
                f"[CONFIG] Replacing obsolete {ISOLATED_TOOLS[key].display_name} "
                f"environment {requested_environment!r} with {environment!r}.",
                flush=True,
            )
        setattr(args, f"{key}_env", environment)
        isolated_environments[key] = environment
    isolated_status, isolated_available = _check_isolated_tools_interactively(
        selected_isolated,
        isolated_environments,
    )
    all_available = all_available and isolated_available
    for key, status in isolated_status.items():
        if not status.available:
            continue
        environment = isolated_environments[key]
        saved = save_isolated_environment(key, environment)
        setattr(args, f"{key}_env", saved)
        print(
            f"[CONFIG] Saved {ISOLATED_TOOLS[key].display_name} environment: {saved}",
            flush=True,
        )
    if selected_isolated:
        print(
            f"[CONFIG] Reusable environment settings: "
            f"{isolated_environment_config_path()}",
            flush=True,
        )

    cuda_host = host_cuda_status()
    print_host_cuda_status(cuda_host)
    cuda_entries: list[
        tuple[CudaRuntimeStatus, str | None, str | None, str | None]
    ] = []
    if selected_all or selected_essential or args.scope == "bin":
        cuda_entries.append((_main_cuda_status(), None, None, None))
    for key in ("comebin", "lorbin"):
        status = isolated_status.get(key)
        if status is None or not status.available:
            continue
        cuda_entries.append(
            (
                _isolated_cuda_status(
                    key,
                    getattr(args, f"{key}_env"),
                    status.frontend,
                ),
                key,
                getattr(args, f"{key}_env"),
                status.frontend,
            )
        )
    cuda_runtimes: list[CudaRuntimeStatus] = []
    for status, key, environment, frontend in cuda_entries:
        print_cuda_runtime_status(status)
        if cuda_host.available and not status.available:
            status = _offer_cuda_repair(
                status,
                key,
                environment,
                frontend,
            )
        cuda_runtimes.append(status)
    cuda_available = cuda_host.available and all(
        status.available for status in cuda_runtimes
    )
    if args.require_cuda and not cuda_available:
        print(
            "CUDA was required, but the driver or a selected runtime environment "
            "failed its allocation test.",
            file=sys.stderr,
            flush=True,
        )
        all_available = False
    elif not cuda_available:
        print(
            "[CUDA INFO] CUDA is optional for this check; CPU execution remains available.",
            flush=True,
        )

    checks_default_bin = selected_all or selected_essential or args.scope == "bin"
    if checks_default_bin:
        magscot_dir = Path(args.magscot_dir).expanduser().resolve()
        missing_magscot = [
            path for path in magscot_files(magscot_dir) if not path.is_file()
        ]
        if missing_magscot:
            print(
                "[MISSING] MAGScoT files: "
                + ", ".join(map(str, missing_magscot)),
                file=sys.stderr,
                flush=True,
            )
            if confirm_install(f"Install MAGScoT under {magscot_dir}"):
                install_magscot(magscot_dir)
            else:
                all_available = False
        remaining_magscot = [
            path for path in magscot_files(magscot_dir) if not path.is_file()
        ]
        if not remaining_magscot:
            saved_magscot = save_magscot_directory(magscot_dir)
            args.magscot_dir = str(saved_magscot)
            print(
                f"[CONFIG] Saved MAGScoT directory: {saved_magscot}",
                flush=True,
            )

    checks_binchicken = (
        selected_all or args.scope == "binchicken" or inferred_binchicken_scope
    )
    if checks_binchicken:
        metapackage = (
            Path(args.binchicken_singlem_metapackage).expanduser().resolve()
            if args.binchicken_singlem_metapackage
            else configured_binchicken_singlem_metapackage()
        )
        while (
            (metapackage is None or not reference_package_valid(metapackage))
            and sys.stdin.isatty()
        ):
            try:
                value = input(
                    "Bin Chicken SingleM metapackage path "
                    "(press Enter to leave it unset): "
                ).strip()
            except EOFError:
                break
            if not value:
                break
            candidate = Path(value).expanduser().resolve()
            if reference_package_valid(candidate):
                metapackage = save_binchicken_singlem_metapackage(candidate)
                print(
                    "[CONFIG] Saved Bin Chicken SingleM metapackage: "
                    f"{metapackage}",
                    flush=True,
                )
                break
            print(
                f"[INVALID] Bin Chicken SingleM metapackage: {candidate}. "
                "Enter the path again, or press Enter to skip.",
                file=sys.stderr,
                flush=True,
            )
        valid_metapackage = (
            metapackage is not None and reference_package_valid(metapackage)
        )
        if valid_metapackage and args.binchicken_singlem_metapackage:
            metapackage = save_binchicken_singlem_metapackage(metapackage)
        print(
            f"[{'OK' if valid_metapackage else 'MISSING'}] Bin Chicken SingleM "
            f"metapackage: {metapackage or 'not configured'}",
            flush=True,
        )
        if not valid_metapackage:
            print(
                "Configure it with `metabaw check --scope binchicken "
                "--binchicken-singlem-metapackage PATH`. MetaBAW does not "
                "install or invoke Aviary.",
                file=sys.stderr,
                flush=True,
            )
            all_available = False

    database_root = Path(args.database_dir).expanduser().resolve()
    database_status: list[
        tuple[str, Path | None, Callable[[Path], bool]]
    ] = []
    checks_checkm = selected_all or args.scope == "checkm"
    checks_checkm2 = (
        selected_all
        or selected_essential
        or args.scope in {"bin", "checkm2"}
    )
    checks_gunc = selected_all
    checks_gtdbtk = selected_all or selected_essential or args.scope == "annotation"
    if checks_checkm:
        checkm = configured_database_path(
            args.checkm_db,
            "CHECKM_DATA_PATH",
            "checkm",
            validator=checkm_database_valid,
        )
        if selected_all:
            checkm = _prompt_database_path(
                "CheckM",
                checkm,
                checkm_database_valid,
                "checkm",
                missing_reasons=checkm_database_missing,
            )
        if args.checkm_db and checkm is not None and checkm_database_valid(checkm):
            save_database_path("checkm", checkm)
        database_status.append(("CheckM", checkm, checkm_database_valid))
    if checks_checkm2:
        checkm2 = configured_database_path(
            args.checkm2_db,
            "CHECKM2DB",
            "checkm2",
            artifact_suffix=".dmnd",
        )
        if selected_all:
            checkm2 = _prompt_database_path(
                "CheckM2",
                checkm2,
                diamond_database_valid,
                "checkm2",
                artifact_suffix=".dmnd",
            )
        installed = False
        if args.install_databases and (
            checkm2 is None or not diamond_database_valid(checkm2)
        ):
            status = isolated_status.get("checkm2")
            if status is None:
                raise RuntimeError(
                    "CheckM2 must be installed before its database can be downloaded."
                )
            checkm2 = install_checkm2_database(
                database_root / "checkm2",
                isolated_run_prefix(args.checkm2_env, status.frontend),
            )
            installed = True
        if (
            (args.checkm2_db or installed)
            and checkm2 is not None
            and diamond_database_valid(checkm2)
        ):
            save_database_path("checkm2", checkm2)
        database_status.append(("CheckM2", checkm2, diamond_database_valid))
    if checks_gunc:
        gunc = configured_database_path(
            args.gunc_db,
            "GUNC_DB",
            "gunc",
            artifact_suffix=".dmnd",
        )
        if selected_all:
            gunc = _prompt_database_path(
                "GUNC",
                gunc,
                diamond_database_valid,
                "gunc",
                artifact_suffix=".dmnd",
            )
        installed = False
        if args.install_databases and (
            gunc is None or not diamond_database_valid(gunc)
        ):
            gunc = install_gunc_database(database_root / "gunc")
            installed = True
        if (
            (args.gunc_db or installed)
            and gunc is not None
            and diamond_database_valid(gunc)
        ):
            save_database_path("gunc", gunc)
        database_status.append(("GUNC", gunc, diamond_database_valid))
    if checks_gtdbtk:
        gtdbtk = configured_database_path(
            args.gtdbtk_data,
            "GTDBTK_DATA_PATH",
            "gtdbtk",
            validator=gtdbtk_database_valid,
        )
        gtdbtk = _prompt_database_path(
            "GTDB-Tk",
            gtdbtk,
            gtdbtk_database_valid,
            "gtdbtk",
            missing_reasons=gtdbtk_database_missing,
        )
        installed = False
        if args.install_databases and (
            gtdbtk is None or not gtdbtk_database_valid(gtdbtk)
        ):
            gtdbtk = install_gtdbtk_database(database_root / "gtdbtk")
            installed = True
        if gtdbtk is not None and gtdbtk_database_valid(gtdbtk):
            save_database_path("gtdbtk", gtdbtk)
        database_status.append(("GTDB-Tk", gtdbtk, gtdbtk_database_valid))
    checks_kofam = (
        selected_all
        or selected_essential
        or args.scope in {"annotation", "kegg"}
    )
    if checks_kofam:
        kofam = configured_database_path(
            args.kegg_db,
            "KOFAM_DB",
            "kofam",
            validator=kofam_database_valid,
        )
        if selected_all:
            kofam = _prompt_database_path(
                "KOfam/KEGG",
                kofam,
                kofam_database_valid,
                "kofam",
            )
        installed = False
        if args.install_databases and (
            kofam is None or not kofam_database_valid(kofam)
        ):
            kofam = install_kofam_database(database_root / "kofam")
            installed = True
        if (
            (args.kegg_db or installed)
            and kofam is not None
            and kofam_database_valid(kofam)
        ):
            save_database_path("kofam", kofam)
        database_status.append(("KOfam/KEGG", kofam, kofam_database_valid))
    checks_dbcan = (
        selected_all
        or selected_essential
        or args.scope in {"annotation", "cazy"}
    )
    if checks_dbcan:
        dbcan = configured_database_path(
            args.dbcan_db,
            "DBCAN_DB",
            "dbcan",
            validator=dbcan_database_valid,
        )
        if selected_all:
            dbcan = _prompt_database_path(
                "dbCAN/CAZy",
                dbcan,
                dbcan_database_valid,
                "dbcan",
                missing_reasons=dbcan_database_missing,
            )
        installed = False
        if args.install_databases and (
            dbcan is None or not dbcan_database_valid(dbcan)
        ):
            dbcan = install_dbcan_database(database_root / "dbcan")
            installed = True
        if (
            (args.dbcan_db or installed)
            and dbcan is not None
            and dbcan_database_valid(dbcan)
        ):
            save_database_path("dbcan", dbcan)
        database_status.append(("dbCAN/CAZy", dbcan, dbcan_database_valid))
    checks_hydrogenase = (
        selected_all
        or selected_essential
        or args.scope in {"annotation", "hydrogenase"}
    )
    if checks_hydrogenase:
        hydrogenase = configured_database_path(
            args.hydrogenase_db,
            "HYDROGENASE_DB",
            "hydrogenase",
            validator=hydrogenase_database_valid,
        )
        if selected_all or args.scope == "hydrogenase":
            hydrogenase = _prompt_database_path(
                "Hydrogenase",
                hydrogenase,
                hydrogenase_database_valid,
                "hydrogenase",
                missing_reasons=hydrogenase_database_missing,
            )
        if (
            args.hydrogenase_db
            and hydrogenase is not None
            and hydrogenase_database_valid(hydrogenase)
        ):
            save_database_path("hydrogenase", hydrogenase)
        database_status.append(
            ("Hydrogenase", hydrogenase, hydrogenase_database_valid)
        )
    missing_databases = []
    for name, path, validator in database_status:
        valid = path is not None and validator(path)
        state = "OK" if valid else "MISSING"
        detail = ""
        if path is not None and not valid and name == "CheckM":
            detail = "; missing=" + ", ".join(checkm_database_missing(path))
        elif path is not None and not valid and name in {"CheckM2", "GUNC"}:
            detail = "; invalid=" + ", ".join(diamond_database_missing(path))
        elif path is not None and not valid and name == "GTDB-Tk":
            detail = "; missing=" + ", ".join(gtdbtk_database_missing(path))
        elif path is not None and not valid and name == "KOfam/KEGG":
            detail = "; missing=" + ", ".join(
                str(item) for item in kofam_database_missing(path)
            )
        elif path is not None and not valid and name == "dbCAN/CAZy":
            detail = "; missing=" + ", ".join(dbcan_database_missing(path))
        elif path is not None and not valid and name == "Hydrogenase":
            detail = "; missing=" + ", ".join(
                hydrogenase_database_missing(path)
            )
        print(f"[{state}] {name} database: {path or 'not configured'}{detail}")
        if not valid:
            missing_databases.append(name)
    if missing_databases:
        print(
            "Missing databases were not downloaded. Re-run with --install-databases "
            "or configure their paths.",
            file=sys.stderr,
        )
        return 1
    if not all_available:
        print(
            "One or more selected software dependencies are still missing.",
            file=sys.stderr,
            flush=True,
        )
        return 1
    print("All selected software and databases are available.", flush=True)
    return 0


def _run_direct(
    args: argparse.Namespace,
    tasks: list[Task],
    output: Path,
    temp_files: Path,
    payload: dict[str, object],
) -> int:
    known_memory_tasks = [task for task in tasks if task.minimum_memory_gb > 0]
    estimated_minimum_gb = max(
        (task.minimum_memory_gb for task in known_memory_tasks),
        default=0.0,
    )
    if known_memory_tasks:
        estimate_sources = sorted(
            {
                task.description or task.id
                for task in known_memory_tasks
                if task.minimum_memory_gb == estimated_minimum_gb
            }
        )
        difference = args.max_memory - estimated_minimum_gb
        balance = (
            f"headroom={difference:g} GiB"
            if difference >= 0
            else f"shortfall={-difference:g} GiB"
        )
        estimate_detail = (
            f"estimated minimum total workflow memory="
            f"{estimated_minimum_gb:g} GiB; configured "
            f"--max-memory={args.max_memory:g} GiB; {balance}; basis="
            f"{', '.join(estimate_sources)}. This is a conservative known-task "
            "floor; data-dependent additional usage remains monitored at runtime."
        )
    else:
        estimate_detail = (
            "estimated minimum total workflow memory has no fixed declared "
            f"high-memory floor for the selected tasks; configured --max-memory="
            f"{args.max_memory:g} GiB. Data-dependent usage remains monitored "
            "at runtime."
        )
    print(f"[00:00:00] [MEMORY ESTIMATE] {estimate_detail}", flush=True)
    memory_requirement = max(
        (task for task in tasks if task.minimum_memory_gb > args.max_memory),
        key=lambda task: task.minimum_memory_gb,
        default=None,
    )
    if memory_requirement is not None:
        hint = memory_requirement.memory_requirement_hint or (
            f"increase --max-memory to at least "
            f"{memory_requirement.minimum_memory_gb:g} GiB"
        )
        detail = (
            f"configured --max-memory={args.max_memory:g} GiB is below the known "
            f"minimum of {memory_requirement.minimum_memory_gb:g} GiB required by "
            f"task {memory_requirement.id} "
            f"({memory_requirement.description or memory_requirement.stage}); "
            f"{hint}. No workflow task process was started."
        )
        log_suffix = ""
        if not args.dry_run:
            preflight_log = temp_files / "runtime" / "logs" / "memory_preflight.log"
            preflight_log.parent.mkdir(parents=True, exist_ok=True)
            preflight_log.write_text(
                f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] "
                f"[MEMORY ESTIMATE] {estimate_detail}\n"
                f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] "
                f"[MEMORY CONFIG ERROR] {detail}\n",
                encoding="utf-8",
            )
            log_suffix = f" log={preflight_log}"
        print(
            f"[00:00:00] [MEMORY CONFIG ERROR] {detail}{log_suffix}",
            file=sys.stderr,
            flush=True,
        )
        return 1
    if args.dry_run:
        print(f"metaBAW plan: {len(tasks)} tasks")
        print(f"metaBAW temporary workspace: {temp_files}")
        current_stage = None
        for task in tasks:
            if task.stage != current_stage:
                current_stage = task.stage
                print(f"\n[{current_stage}]")
            print(f"  {task.id}  cpus={task.cpus}  {task.description}")
            print(f"    $ {task.display_command()}")
        return 0
    workflow_started_at = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    runtime_dir = temp_files / "runtime"
    log_dir = runtime_dir / "logs"
    system_temp_dir = temp_files / "system_tmp"
    log_dir.mkdir(parents=True, exist_ok=True)
    system_temp_dir.mkdir(parents=True, exist_ok=True)
    payload.update(
        {
            "software": "metaBAW",
            "version": __version__,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "resources": {
                "total_threads": args.threads,
                "threads_per_sample_task": getattr(
                    args, "threads_per_task", args.threads
                ),
                "threads_per_task": getattr(
                    args, "threads_per_task", args.threads
                ),
                "concurrent_samples": getattr(
                    args, "concurrent_sample_limit", args.task
                ),
                "maximum_cpu_threads": args.threads,
                "max_memory_gb": args.max_memory,
                "total_memory_gb": args.max_memory,
                "memory_scope": "combined_pss_of_workflow_task_process_trees",
                "memory_measurement": "linux_pss_with_rss_fallback",
                "estimated_minimum_total_memory_gb": estimated_minimum_gb,
                "memory_estimate_scope": "known_high_memory_task_floor",
                "memory_estimate_sources": sorted(
                    task.id
                    for task in known_memory_tasks
                    if task.minimum_memory_gb == estimated_minimum_gb
                ),
                "gpu_execution": bool(getattr(args, "gpu", False)),
                "visible_gpu_devices": int(
                    getattr(args, "cuda_device_count", 0)
                ),
                "gpu_task_slots": int(
                    getattr(args, "cuda_task_slots", 0)
                ),
            },
            "temporary_files": {
                "workspace": str(temp_files),
                "delete_after_success": args.delete_tmp_files,
                "system_temp": str(system_temp_dir),
                "state": str(runtime_dir / "state.sqlite3"),
                "logs": str(log_dir),
            },
            "tasks": [_task_dict(task) for task in tasks],
        }
    )
    manifest = output / "run_manifest.json"
    manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"[{_elapsed_clock(workflow_started_at)}] [RESOURCES] "
        f"--max-memory={args.max_memory:g} GiB is the total workflow PSS "
        "budget shared by all concurrent task process trees; RSS is used only "
        "when Linux PSS is unavailable.",
        flush=True,
    )
    print(
        f"[{_elapsed_clock(workflow_started_at)}] [PIPELINE] "
        f"Running {payload.get('module', 'workflow')} "
        f"with {len(tasks)} tasks (total_threads={args.threads}, "
        f"threads_per_sample={getattr(args, 'threads_per_task', args.threads)}, "
        f"concurrent_samples={getattr(args, 'concurrent_sample_limit', args.task)}, "
        f"total_memory_limit={args.max_memory:g} GiB); logs={log_dir}",
        flush=True,
    )
    state = StateStore(runtime_dir / "state.sqlite3")
    memory_abort_reason: str | None = None
    try:
        executor = Executor(
            state=state,
            log_dir=log_dir,
            max_cpus=args.threads,
            max_parallel=getattr(
                args, "concurrent_sample_limit", args.task
            ),
            shell_executable=os.environ.get("SHELL", "/bin/bash"),
            retries=args.retries,
            fail_fast=False,
            max_memory_gb=args.max_memory,
            show_progress=True,
            temp_dir=system_temp_dir,
            max_gpus=getattr(
                args,
                "cuda_task_slots",
                getattr(args, "cuda_device_count", 0),
            ),
            started_at=workflow_started_at,
        )
        statuses = executor.run(tasks, force=args.force)
        peak_memory = getattr(executor, "peak_memory_gb", None)
        reported_abort_reason = getattr(executor, "memory_abort_reason", None)
        if isinstance(reported_abort_reason, str) and reported_abort_reason:
            memory_abort_reason = reported_abort_reason
        resources = payload.get("resources")
        if isinstance(resources, dict):
            if isinstance(peak_memory, (int, float)):
                resources["observed_peak_workflow_pss_gb"] = round(peak_memory, 3)
            if memory_abort_reason is not None:
                resources["memory_abort_reason"] = memory_abort_reason
            manifest.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    finally:
        state.close()
    counts = Counter(statuses.values())
    summary = (
        ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))
        or "no tasks"
    )
    by_id = {task.id: task for task in tasks}
    tolerated_failures = [
        task_id
        for task_id, status in statuses.items()
        if status == "failed" and by_id[task_id].failure_tolerated
    ]
    if tolerated_failures:
        print(
            f"[{_elapsed_clock(workflow_started_at)}] [WARNING] "
            f"{len(tolerated_failures)} optional "
            f"task(s) failed; workflow continued: "
            f"{_task_id_summary(tolerated_failures)}",
            file=sys.stderr,
            flush=True,
        )
    failures = [
        task_id
        for task_id, status in statuses.items()
        if status == "failed" and not by_id[task_id].failure_tolerated
    ]
    required_blocked = [
        task_id
        for task_id, status in statuses.items()
        if status == "blocked" and not by_id[task_id].failure_tolerated
    ]
    if failures or required_blocked:
        details = []
        if failures:
            details.append(
                f"failed={len(failures)} [{_task_id_summary(failures)}]"
            )
        if required_blocked:
            details.append(
                "required_blocked="
                f"{len(required_blocked)} [{_task_id_summary(required_blocked)}]"
            )
        if memory_abort_reason is not None:
            print(
                f"[{_elapsed_clock(workflow_started_at)}] [ERROR] "
                f"Workflow stopped by --max-memory: {memory_abort_reason}; "
                f"{'; '.join(details)}; status={summary}; logs={log_dir}",
                file=sys.stderr,
                flush=True,
            )
        else:
            print(
                f"[{_elapsed_clock(workflow_started_at)}] [ERROR] "
                f"Workflow did not complete: {'; '.join(details)}; "
                f"status={summary}; logs={log_dir}",
                file=sys.stderr,
                flush=True,
            )
        return 1
    if args.delete_tmp_files:
        if not _safe_to_delete_temp(temp_files, output):
            raise ValueError(
                "--delete-tmp-files cannot remove a temporary directory that contains --output"
            )
        shutil.rmtree(temp_files)
        temp_detail = "temporary files deleted"
    else:
        temp_detail = f"temporary files={temp_files}"
    print(
        f"[{_elapsed_clock(workflow_started_at)}] [COMPLETE] "
        f"Workflow completed successfully; status={summary}; "
        f"results={output}; {temp_detail}",
        flush=True,
    )
    return 0


def command_bin(args: argparse.Namespace) -> int:
    # Tool environments, databases, and MAGScoT are configured once by
    # `metabaw check`; the bin command intentionally exposes no duplicate
    # path options.
    args.comebin_env = configured_isolated_environment("comebin")
    args.lorbin_env = configured_isolated_environment("lorbin")
    args.metawrap_env = configured_isolated_environment("metawrap")
    args.checkm2_env = configured_isolated_environment("checkm2")
    args.binchicken_env = configured_isolated_environment("binchicken")
    args.magscot_dir = str(configured_magscot_directory())
    args.checkm2_db = None
    args.gunc_db = None
    args.gtdbtk_data = None
    explicit_tools = parse_choices(args.tools) if args.tools is not None else None
    if explicit_tools is not None:
        _validate_metawrap_binner_limit(args.refinement, explicit_tools)
    if args.trna_pass is not None:
        args.trna = True
    if args.rrna_pass:
        args.rrna = True
    output = Path(args.output).expanduser().resolve()
    if args.multi_files:
        if args.mode == "single":
            raise ValueError("--multi-files cannot be combined with --single")
        # An explicit reads-to-contigs mapping is itself an unambiguous request
        # for multi-sample binning; requiring a second --multi flag only makes
        # otherwise valid commands fail for syntactic reasons.
        args.mode = "multi"
    if args.cohort_size is not None and args.cohort_size < 1:
        raise ValueError("--cohort-size must be positive")
    if args.cohort_size is not None and args.mode != "multi":
        raise ValueError("--cohort-size requires --multi")
    if args.assembly_strategy == "bin-chicken":
        if args.multi_files:
            raise ValueError(
                "--assembly-strategy bin-chicken cannot be combined with "
                "--multi-files because Bin Chicken selects its own sample groups"
            )
        if args.cohort_size is not None:
            raise ValueError(
                "--assembly-strategy bin-chicken cannot be combined with "
                "--cohort-size because Bin Chicken defines each coassembly cohort"
            )
    if args.align_tool is None:
        args.align_tool = "bowtie2" if args.type == "short" else "minimap2"
    if args.type == "long" and args.align_tool == "bowtie2":
        raise ValueError("Bowtie2 is for short reads; use minimap2 or minibwa with --type long")
    if args.type == "short" and args.align_tool == "minibwa":
        raise ValueError("minibwa is exposed here for accurate long reads; use bowtie2/minimap2 for short reads")

    if args.input_reads_files:
        reads = discover_reads_from_file(
            args.input_reads_files,
            args.type,
            args.separate_sample_name,
        )
    else:
        reads = discover_reads(
            args.path,
            args.suffix,
            args.type,
            args.separate_sample_name,
        )
    if args.input_contig_files:
        contigs = discover_contigs_from_file(args.input_contig_files)
        contig_suffix: str | None = None
    else:
        contigs = discover_contigs(args.contig, args.contig_suffix)
        contig_suffix = args.contig_suffix
    reads = attach_contigs(reads, contigs, contig_suffix)
    selected = None
    if args.multi_files:
        selected = read_multi_files(args.multi_files)
        selected_names = {sample.name for sample in selected}
        reads = selected + [
            sample for sample in reads if sample.name not in selected_names
        ]
        if args.cohort_size is not None:
            raise ValueError(
                "--cohort-size cannot be combined with the explicit "
                "--multi-files group"
            )

    args.mode = args.mode or "single"
    args.min_contig_length = (
        1500 if args.min_contig_length is None else args.min_contig_length
    )
    requested_tools = explicit_tools
    if args.threads < 1 or args.task < 1:
        raise ValueError("--threads and --task must be positive")
    if args.max_memory <= 0:
        raise ValueError("--max-memory must be positive")
    if (
        (args.min_contig_length is not None and args.min_contig_length < 1)
        or args.minfasta_kbs < 1
        or args.batch_size < 1
    ):
        raise ValueError("contig length, minimum bin size and batch size must be positive")
    if not 0 <= args.con <= 100 or not 0 <= args.com <= 100:
        raise ValueError("--con and --com must be in [0, 100]")
    if args.trna_pass is not None and args.trna_pass < 1:
        raise ValueError("--trna-pass must be positive")
    if not re.fullmatch(
        r"\d+(?:\.\d+)?(?:[KMGTP]i?B?|%)?",
        args.max_gpu_memory,
        re.IGNORECASE,
    ):
        raise ValueError(
            "--max-gpu-memory must look like 4G, 4096M, 50%, or a byte count"
        )
    if requested_tools is None:
        requested_tools = (
            ["metadecoder", "vamb", "lorbin"]
            if args.type == "long"
            else ["metabat2", "metadecoder", "vamb"]
        )
    binners = parse_choices(requested_tools)
    supported = {
        "metabat2",
        "vamb",
        "metadecoder",
        "comebin",
        "semibin2",
        "lorbin",
    }
    unknown = sorted(set(binners) - supported)
    if not binners or unknown:
        raise ValueError(f"--tools must select one or more of {sorted(supported)}; unknown={unknown}")
    binners = [name for name in BINNER_ORDER if name in binners]
    if args.type != "long" and "lorbin" in binners:
        raise ValueError("LorBin is supported only with --type long")
    _validate_metawrap_binner_limit(args.refinement, binners)
    args.tools = binners
    _resolve_bin_option_priorities(args)

    output = Path(args.output).expanduser().resolve()
    tmp = _temporary_path(args.tmp_files, output)
    args.requested_batch_size = args.batch_size
    args.cuda_device_count = 0
    args.cuda_task_slots = 0
    args.gpu_binners = ()
    bin_chicken_plans: list[CoassemblyPlan] = []
    bin_chicken_plan_path: Path | None = None

    if args.assembly_strategy == "bin-chicken":
        if args.dry_run:
            analyses = build_analyses(reads, args.mode)
            print(
                "[PLAN] --dry-run cannot execute Bin Chicken; displaying the "
                "default MetaBAW task graph. Run without --dry-run to generate "
                "the Bin Chicken coassembly plan.",
                flush=True,
            )
        else:
            # The planning software and native assembler must be verified before
            # launching the external planner, so a missing executable is reported
            # through MetaBAW's normal installation prompt.
            _preflight_bin(args)
            bin_chicken_plans, bin_chicken_plan_path = _run_bin_chicken_planner(
                args,
                reads,
                output,
                tmp,
            )
            if bin_chicken_plans:
                analyses = build_bin_chicken_analyses(
                    reads,
                    bin_chicken_plans,
                    tmp / "work" / "coassembly",
                    args.mode,
                )
            else:
                analyses = build_analyses(reads, args.mode)
    elif selected is not None:
        analyses = build_analyses(
            reads,
            args.mode,
            selected,
            explicit_group_name=_multi_file_group_name(args.multi_files),
        )
    elif args.cohort_size is not None:
        analyses = build_cohort_analyses(
            reads,
            args.mode,
            args.cohort_size,
        )
    else:
        analyses = build_analyses(reads, args.mode)

    threads_per_sample = _configure_thread_budget(args, len(analyses))

    mag_suffix = args.mag_suffix.lstrip(".")
    if not mag_suffix or any(char in mag_suffix for char in "/\\"):
        raise ValueError(f"Invalid MAG suffix: {args.mag_suffix!r}")
    selected_environments = []
    if "comebin" in binners:
        selected_environments.append(("comebin", args.comebin_env))
    if "lorbin" in binners:
        selected_environments.append(("lorbin", args.lorbin_env))
    if args.refinement == "metawrap":
        selected_environments.append(("metawrap", args.metawrap_env))
    if args.quality_control == "checkm2":
        selected_environments.append(("checkm2", args.checkm2_env))
    for key, environment in selected_environments:
        if not environment.strip():
            raise ValueError(
                f"{ISOLATED_TOOLS[key].option} cannot be empty when "
                f"{ISOLATED_TOOLS[key].display_name} is selected"
            )
    if args.dry_run:
        if args.gpu is None:
            args.gpu = False
            print(
                "[GPU AUTO] CUDA probing is skipped during --dry-run; "
                "generating a CPU plan. Run without --dry-run to probe CUDA "
                "automatically.",
                flush=True,
            )
        _apply_comebin_gpu_budget(args)
    elif args.assembly_strategy != "bin-chicken":
        _preflight_bin(args)
    if (
        "semibin2" in binners
        and args.environment
        and any(
            len(analysis.samples) > 1 and not analysis.combined
            for analysis in analyses
        )
    ):
        print(
            "[WARNING] SemiBin2 pretrained environment models cannot be used "
            "with multi-sample BAM sets; affected analyses will "
            "use self-supervised training. Per-sample assemblies use "
            "SemiBin2 multi_easy_bin.",
            file=sys.stderr,
            flush=True,
        )

    options = BinOptions(
        outdir=output,
        workdir=tmp / "work",
        threads=threads_per_sample,
        total_threads=args.threads,
        read_type=args.type,
        align_tool=args.align_tool,
        long_read_preset=args.long_read_preset,
        binners=tuple(binners),
        min_contig_length=args.min_contig_length,
        min_fasta_kbs=args.minfasta_kbs,
        batch_size=args.batch_size,
        refiner=args.refinement,
        quality_control=args.quality_control,
        min_completeness=args.com,
        max_contamination=args.con,
        min_quality_score=args.quality_score,
        run_gunc=args.gunc,
        run_trna=args.trna,
        trna_pass=args.trna_pass,
        run_rrna=args.rrna,
        rrna_pass=args.rrna_pass,
        dereplicator=args.dereplication_tool,
        ani=args.ani,
        min_aligned_fraction=args.min_aligned_fraction,
        mag_suffix=mag_suffix,
        environment=args.environment,
        tag_contigs=args.tag_contigs,
        gpu=args.gpu,
        gpu_binners=tuple(args.gpu_binners),
        max_gpu_memory=args.max_gpu_memory,
        assembly_strategy=args.assembly_strategy,
        magscot_dir=Path(args.magscot_dir).expanduser().resolve(),
        checkm2_db=Path(args.checkm2_db).expanduser().resolve() if args.checkm2_db else None,
        gunc_db=Path(args.gunc_db).expanduser().resolve() if args.gunc_db else None,
        gtdbtk_data=Path(args.gtdbtk_data).expanduser().resolve() if args.gtdbtk_data else None,
        extra_args=_parse_advanced(args.advanced_arg),
        comebin_run_prefix=isolated_run_prefix(
            args.comebin_env,
            getattr(args, "comebin_frontend", None),
            prefer_lock_free_runner=True,
            live_output=True,
        ),
        checkm2_run_prefix=isolated_run_prefix(
            args.checkm2_env,
            getattr(args, "checkm2_frontend", None),
            prefer_lock_free_runner=True,
        ),
        metawrap_run_prefix=isolated_run_prefix(
            args.metawrap_env,
            getattr(args, "metawrap_frontend", None),
            prefer_lock_free_runner=True,
        ),
        lorbin_run_prefix=isolated_run_prefix(
            args.lorbin_env,
            getattr(args, "lorbin_frontend", None),
            prefer_lock_free_runner=True,
        ),
    )
    tasks = DirectBinBuilder(analyses, options, options.workdir).build()
    payload = {
        "module": "bin",
        "mode": args.mode,
        "assembly_strategy": args.assembly_strategy,
        "bin_chicken": (
            {
                "planner": "binchicken coassemble",
                "aviary_installed_or_invoked": False,
                "plan": str(bin_chicken_plan_path),
                "coassembly_groups": [
                    {
                        "id": plan.identifier,
                        "assembly_samples": list(plan.assembly_sample_names),
                        "recovery_samples": list(plan.recovery_sample_names),
                    }
                    for plan in bin_chicken_plans
                ],
            }
            if args.assembly_strategy == "bin-chicken"
            else None
        ),
        "read_type": args.type,
        "input_sources": {
            "reads": (
                {"kind": "file_list", "path": str(Path(args.input_reads_files).expanduser().resolve())}
                if args.input_reads_files
                else {
                    "kind": "path",
                    "path": str(Path(args.path).expanduser().resolve()),
                    "suffix": args.suffix,
                }
            ),
            "contigs": (
                {"kind": "file_list", "path": str(Path(args.input_contig_files).expanduser().resolve())}
                if args.input_contig_files
                else {
                    "kind": "path",
                    "path": str(Path(args.contig).expanduser().resolve()),
                    "suffix": args.contig_suffix,
                }
            ),
            "multi_mapping": (
                str(Path(args.multi_files).expanduser().resolve())
                if args.multi_files
                else None
            ),
        },
        "align_tool": args.align_tool,
        "binners": binners,
        "refinement": args.refinement,
        "quality_control": args.quality_control,
        "dereplication_tool": args.dereplication_tool,
        "gpu": {
            "enabled": args.gpu,
            "enabled_binners": list(args.gpu_binners),
            "cpu_fallback_binners": [
                tool
                for tool in binners
                if tool in GPU_MINIMUM_MEMORY_GIB
                and tool not in args.gpu_binners
            ],
            "visible_device_count": args.cuda_device_count,
            "concurrent_task_slots": args.cuda_task_slots,
            "memory_budget_per_task": args.max_gpu_memory if args.gpu else None,
            "requested_memory_budget_per_task": args.max_gpu_memory,
            "comebin_batch_size_requested": args.requested_batch_size,
            "comebin_batch_size_effective": args.batch_size,
            "comebin_minimum_gpu_memory_gib": (
                _comebin_minimum_memory_gib(args.requested_batch_size)
                if "comebin" in binners
                else None
            ),
        },
        "parameter_warnings": list(args.parameter_warnings),
        "semibin2_environment": args.environment,
        "isolated_environments": {
            key: environment
            for key, environment in selected_environments
        },
        "analyses": [
            {
                "name": analysis.name,
                "public_name": analysis.public_name or public_sample_name(analysis.name),
                "explicit_mapping": analysis.explicit_mapping,
                "combined": analysis.combined,
                "plan_id": analysis.plan_id,
                "assembly_samples": [
                    sample.name for sample in analysis.assembly_samples
                ],
                "samples": [sample.name for sample in analysis.samples],
                "contigs": [str(path) for path in analysis.contigs],
            }
            for analysis in analyses
        ],
    }
    return _run_direct(args, tasks, output, tmp, payload)


def command_annotation(args: argparse.Namespace) -> int:
    if args.threads < 1 or args.task < 1:
        raise ValueError("--threads and --task must be positive")
    if args.max_memory <= 0:
        raise ValueError("--max-memory must be positive")
    mags = discover_mag_files(args.path, args.suffix)
    mag_dir = Path(args.path).expanduser().resolve()
    if mag_dir.is_file():
        raise ValueError("metabaw annotation currently requires -p/--path to be a MAG directory")
    gtdbtk_result = None
    if args.gtdbtk_res:
        gtdbtk_result = Path(args.gtdbtk_res).expanduser().resolve()
        missing = gtdbtk_result_missing(gtdbtk_result, mags)
        if missing:
            raise ValueError(
                "Provided GTDB-Tk result path contains incomplete species "
                "annotation information; GTDB-Tk must be rerun: "
                f"{gtdbtk_result}; missing={'; '.join(missing)}"
            )
        args.gtdbtk_res = str(gtdbtk_result)
        print(
            f"[REUSE] Validated GTDB-Tk species annotations for {len(mags)} "
            f"MAG(s) under {gtdbtk_result}; GTDB-Tk classification will be skipped.",
            flush=True,
        )
    reads = discover_reads(args.reads, args.read_suffix, args.type, args.separate_sample_name)
    methods = parse_choices(args.methods)
    allowed = {
        "relative_abundance",
        "rpkm",
        "tpm",
        "mean",
        "count",
        "trimmed_mean",
        "covered_fraction",
        "covered_bases",
        "reads_per_base",
        "variance",
        "length",
    }
    unknown = sorted(set(methods) - allowed)
    if not methods or unknown:
        raise ValueError(f"Unsupported CoverM methods: {unknown}")
    if not args.no_niche:
        if len(reads) < 2:
            raise ValueError(
                "Ecological niche classification requires at least two samples "
                f"for --niche-method {args.niche_method}; provide more samples "
                "or use --no-niche"
            )
        missing_niche_methods = sorted(
            {"relative_abundance", "count"} - set(methods)
        )
        if missing_niche_methods:
            raise ValueError(
                "Ecological niche classification requires CoverM methods "
                "relative_abundance and count; missing: "
                + ", ".join(missing_niche_methods)
            )
    args.kegg_db = None
    args.dbcan_db = None
    args.hydrogenase_db = None
    if args.dry_run:
        if args.kegg:
            configured = configured_database_path(
                None, "KOFAM_DB", "kofam", validator=kofam_database_valid
            )
            args.kegg_db = str(configured) if configured is not None else None
        if args.cazy:
            configured = configured_database_path(
                None, "DBCAN_DB", "dbcan", validator=dbcan_database_valid
            )
            args.dbcan_db = str(configured) if configured is not None else None
        if args.hydrogenase:
            configured = configured_database_path(
                None,
                "HYDROGENASE_DB",
                "hydrogenase",
                validator=hydrogenase_database_valid,
            )
            args.hydrogenase_db = (
                str(configured) if configured is not None else None
            )
    if not args.dry_run:
        _preflight_annotation(args)
    suffix = args.output_file_suffix
    if not suffix.startswith("."):
        suffix = "." + suffix
    output = Path(args.output).expanduser().resolve()
    tmp = _temporary_path(args.tmp_files, output)
    annotation_workloads = max(len(reads), len(mags))
    threads_per_sample = _configure_thread_budget(args, annotation_workloads)
    options = AnnotationOptions(
        mag_dir=mag_dir,
        mags=tuple(mags),
        mag_suffix=args.suffix,
        reads=tuple(reads),
        output=output,
        output_suffix=suffix,
        threads=threads_per_sample,
        read_type=args.type,
        methods=tuple(methods),
        place_species=args.place_species,
        niche_rank=args.niche_rank,
        niche_method=args.niche_method,
        no_niche=args.no_niche,
        gtdbtk_data=Path(args.gtdbtk_data).expanduser().resolve() if args.gtdbtk_data else None,
        run_kegg=args.kegg,
        run_cazy=args.cazy,
        run_hydrogenase=args.hydrogenase,
        kegg_db=Path(args.kegg_db).expanduser().resolve() if args.kegg_db else None,
        dbcan_db=Path(args.dbcan_db).expanduser().resolve() if args.dbcan_db else None,
        hydrogenase_db=(
            Path(args.hydrogenase_db).expanduser().resolve()
            if args.hydrogenase_db
            else None
        ),
        gtdbtk_result=gtdbtk_result,
        total_threads=args.threads,
    )
    tasks = AnnotationBuilder(options, tmp / "work").build()
    payload = {
        "module": "annotation",
        "mags": [str(path) for path in mags],
        "samples": [sample.name for sample in reads],
        "coverm_methods": methods,
        "niche_rank": args.niche_rank,
        "niche_method": args.niche_method,
        "kegg": args.kegg,
        "cazy": args.cazy,
        "hydrogenase": args.hydrogenase,
        "gtdbtk_result_source": str(gtdbtk_result) if gtdbtk_result else None,
        "gtdbtk_classification_skipped": gtdbtk_result is not None,
    }
    return _run_direct(args, tasks, output, tmp, payload)


def build_parser() -> argparse.ArgumentParser:
    parser = _ExplicitOptionParser(
        prog="metabaw",
        description="MetaBAW: metagenome Binning Automated Workflow",
        formatter_class=MetaBAWHelpFormatter,
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"metaBAW {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    command = subparsers.add_parser(
        "bin",
        help="run binning, refinement, quality control, and dereplication",
        description="Discover reads/contigs and run the complete metaBAW binning workflow.",
        formatter_class=MetaBAWHelpFormatter,
        add_help=False,
    )
    command.add_argument(
        "-h", "--help", action=_ShortHelpAction, help="show core options and exit"
    )
    command.add_argument(
        "--full-help", action=_FullHelpAction, help="page through every option and exit"
    )
    read_source = command.add_mutually_exclusive_group(required=True)
    read_source.add_argument("-p", "--path", help="read file or directory")
    read_source.add_argument(
        "--input_reads_files",
        "--input-reads-files",
        dest="input_reads_files",
        metavar="FILE",
        help=(
            "text file containing one read path per line; paired short reads "
            "are grouped from R1/R2 or 1/2 names and -p/-s are not required"
        ),
    )
    contig_source = command.add_mutually_exclusive_group(required=True)
    contig_source.add_argument("-c", "--contig", help="contig file or directory")
    contig_source.add_argument(
        "--input_contig_files",
        "--input-contig-files",
        dest="input_contig_files",
        metavar="FILE",
        help=(
            "text file containing one FASTA contig path per line; -c/-f are "
            "not required"
        ),
    )
    command.add_argument(
        "-s",
        "--suffix",
        default="fastq.gz",
        help="read suffix used with -p/--path (default: fastq.gz)",
    )
    command.add_argument(
        "--separate-sample-name",
        metavar="SEP",
        default=".",
        help=(
            "sample name separator; the part of the read filename before its "
            "first occurrence becomes the sample name, then read mates are "
            "detected from the remaining text (default: '.', so "
            "SRR1.clean.rehost.1.fastq.gz belongs to sample SRR1)"
        ),
    )
    command.add_argument(
        "--type",
        choices=("short", "long"),
        default="short",
        help="read technology type",
    )
    command.add_argument(
        "--align-tool",
        choices=("bowtie2", "minimap2", "minibwa"),
        help="default: bowtie2 for short reads, minimap2 for long reads",
    )
    command.add_argument(
        "--align_tool",
        dest="align_tool",
        choices=("bowtie2", "minimap2", "minibwa"),
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    command.add_argument(
        "--long-read-preset",
        choices=("map-ont", "map-pb", "map-hifi"),
        default="map-ont",
        help="Minimap2 preset for long reads",
    )
    command.add_argument(
        "-f", "--contig-suffix", default="fa", help="contig suffix used with -c/--contig when it is a directory"
    )
    command.add_argument("-o", "--output", default="metabaw_result", help="result directory")
    command.add_argument(
        "--tmp-files",
        help="temporary directory; relative names are created under --output (default: <output>/tmp)",
    )
    command.add_argument(
        "--delete-tmp-files",
        action="store_true",
        help="delete the temporary directory after a successful run",
    )
    command.add_argument(
        "-t",
        "--threads",
        type=int,
        default=1,
        help=(
            "total CPU thread budget shared by concurrently running sample "
            "tasks"
        ),
    )
    command.add_argument(
        "--max-memory",
        type=float,
        default=100,
        help=(
            "maximum combined resident memory in GiB for all workflow task "
            "process trees; exceeding it terminates the workflow"
        ),
    )
    command.add_argument(
        "--task",
        type=int,
        default=1,
        help="maximum number of samples processed concurrently",
    )
    command.add_argument("--retries", type=int, default=0, help="retries after a task failure")
    command.add_argument(
        "--no-gpu",
        dest="gpu",
        action="store_false",
        help="disable CUDA probing and GPU execution (default: automatic CUDA detection)",
    )
    command.set_defaults(gpu=None)
    command.add_argument(
        "--max-gpu-memory",
        default="4G",
        help=(
            "per-task GPU memory budget for automatic CUDA mode; ignored with "
            "--no-gpu; below a binner minimum it forces that binner to CPU "
            "(VAMB/SemiBin2/LorBin: 4G; COMEBin: depends on --batch-size); "
            "accepts bytes, "
            "K/M/G/T/P units, or a percentage"
        ),
    )
    mode = command.add_mutually_exclusive_group()
    mode.add_argument(
        "--single",
        dest="mode",
        action="store_const",
        const="single",
        help="process samples independently (default: selected)",
    )
    mode.add_argument(
        "--multi",
        dest="mode",
        action="store_const",
        const="multi",
        help="process the selected samples together (default: disabled)",
    )
    command.set_defaults(mode=None)
    command.add_argument(
        "--multi-files",
        help=(
            "explicit multi group: READ1[,READ2] TAB CONTIGS, optionally "
            "prefixed by SAMPLE TAB; setting it enables multi-sample binning "
            "without requiring --multi"
        ),
    )
    command.add_argument(
        "--cohort-size",
        type=int,
        help=(
            "maximum number of per-sample assemblies in one multi-sample "
            "coverage cohort; omit to use all selected samples together"
        ),
    )
    command.add_argument(
        "--assembly-strategy",
        choices=("default", "bin-chicken"),
        default="default",
        help=(
            "assembly selection strategy: default preserves the existing "
            "per-sample workflow; bin-chicken uses Bin Chicken only to choose "
            "paired short-read coassemblies, then MetaBAW assembles and bins "
            "them without Aviary"
        ),
    )
    command.add_argument(
        "--tools",
        nargs="+",
        default=None,
        metavar="BINNER",
        help=(
            "metabat2/metadecoder/vamb/comebin/semibin2/lorbin; comma or space "
            "separated (default: short=metabat2,metadecoder,vamb; "
            "long=metadecoder,vamb,lorbin)"
        ),
    )
    command.add_argument(
        "--min-contig-length",
        type=int,
        default=None,
        help="minimum contig length in base pairs (default: 1500)",
    )
    command.add_argument(
        "--minfasta-kbs",
        type=int,
        default=200,
        help="minimum output bin size in kilobases",
    )
    command.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_COMEBIN_BATCH_SIZE,
        help=(
            "COMEBin training batch size; used to calculate its conservative "
            "GPU-memory minimum (default: 1024)"
        ),
    )
    command.add_argument(
        "--refinement",
        type=str.lower,
        choices=("magscot", "das_tool", "metawrap"),
        default="magscot",
        help="bin refinement program",
    )
    command.add_argument(
        "--quality-control",
        type=str.lower,
        choices=("checkm2", "checkm"),
        default="checkm2",
        help="MAG quality estimation program",
    )
    command.add_argument("--con", type=float, default=10.0, help="maximum contamination (%%)")
    command.add_argument("--com", type=float, default=50.0, help="minimum completeness (%%)")
    command.add_argument(
        "--quality-score",
        type=float,
        help="optional minimum completeness - 5 * contamination score",
    )
    command.add_argument("--gunc", action="store_true", help="run GUNC contamination checks")
    command.add_argument(
        "--trna",
        action="store_true",
        help="predict and report tRNAs without filtering MAGs",
    )
    command.add_argument(
        "--trna-pass",
        type=int,
        help=(
            "require at least this many distinct tRNA types; "
            "also enables tRNA prediction"
        ),
    )
    command.add_argument(
        "--rrna",
        action="store_true",
        help="predict and report 5S, 16S, and 23S rRNAs without filtering MAGs",
    )
    command.add_argument(
        "--rrna-pass",
        action="store_true",
        help=(
            "require 5S, 16S, and 23S rRNAs; "
            "also enables rRNA prediction"
        ),
    )
    command.add_argument(
        "--dereplication-tool",
        type=str.lower,
        choices=("galah", "drep"),
        default="galah",
        help="MAG dereplication program",
    )
    command.add_argument(
        "--ani",
        type=float,
        default=99.0,
        help="dereplication ANI threshold in percent (default: 99)",
    )
    command.add_argument(
        "--min-aligned-fraction",
        type=float,
        default=30.0,
        help="minimum aligned fraction in percent for dereplication (default: 30)",
    )
    command.add_argument(
        "-x",
        "--mag-suffix",
        default="fa",
        help=(
            "MAG file extension; propagated to CheckM2/CheckM, GUNC, GTDB-Tk, "
            "and dereplication tools (default: fa)"
        ),
    )
    command.add_argument(
        "--dereplication_tool",
        dest="dereplication_tool",
        type=str.lower,
        choices=("galah", "drep"),
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    command.add_argument(
        "--environment",
        choices=(
            "human_gut",
            "dog_gut",
            "ocean",
            "soil",
            "cat_gut",
            "human_oral",
            "mouse_gut",
            "pig_gut",
            "built_environment",
            "wastewater",
            "chicken_caecum",
            "global",
        ),
        help=(
            "SemiBin2 pretrained environment model; effective only when "
            "SemiBin2 is selected with --tools, then passed to SemiBin2; "
            "otherwise ignored"
        ),
    )
    command.add_argument(
        "--tag-contigs",
        action="store_true",
        help="rename final contigs using the bin_name_number convention",
    )
    command.add_argument(
        "--advanced-arg",
        action="append",
        default=[],
        metavar="TOOL=ARGS",
        help="expert escape hatch; repeat for multiple tools",
    )
    command.add_argument(
        "--dry-run", action="store_true", help="print the task plan without executing it"
    )
    command.add_argument(
        "--force", action="store_true", help="rerun tasks even when cached outputs are valid"
    )
    command.set_defaults(func=command_bin)

    command = subparsers.add_parser(
        "annotation",
        help=(
            "run taxonomy, abundance, niche, KEGG, CAZy, hydrogenase, and "
            "terminal-enzyme annotation"
        ),
        description=(
            "Classify MAGs, quantify abundance, classify ecological niches, and "
            "annotate KEGG orthologs, CAZymes, hydrogenases, and hydrogen-"
            "metabolism terminal enzymes."
        ),
        formatter_class=MetaBAWHelpFormatter,
        add_help=False,
    )
    command.add_argument(
        "-h", "--help", action=_ShortHelpAction, help="show core options and exit"
    )
    command.add_argument(
        "--full-help", action=_FullHelpAction, help="page through every option and exit"
    )
    command.add_argument("-p", "--path", required=True, help="MAG directory")
    command.add_argument("-r", "--reads", required=True, help="read file or directory")
    command.add_argument("-s", "--suffix", default="fa", help="MAG suffix (default: fa)")
    species_placement = command.add_mutually_exclusive_group()
    species_placement.add_argument(
        "--place-species",
        dest="place_species",
        action="store_true",
        help="enable GTDB-Tk species placement (default: enabled)",
    )
    species_placement.add_argument(
        "--no-place-species",
        dest="place_species",
        action="store_false",
        help="disable GTDB-Tk species placement (default: not selected)",
    )
    command.set_defaults(place_species=True)
    command.add_argument(
        "-f", "--read-suffix", default="fastq.gz", help="read filename suffix"
    )
    command.add_argument(
        "--separate-sample-name",
        metavar="SEP",
        default=".",
        help=(
            "sample name separator; the part of the read filename before its "
            "first occurrence becomes the sample name, then read mates are "
            "detected from the remaining text (default: '.', so "
            "SRR1.clean.rehost.1.fastq.gz belongs to sample SRR1)"
        ),
    )
    command.add_argument(
        "--type",
        choices=("short", "long"),
        default="short",
        help="read technology type",
    )
    command.add_argument(
        "--methods",
        nargs="+",
        default=["relative_abundance", "rpkm", "tpm", "mean", "count"],
        metavar="METHOD",
        help="CoverM methods; comma or space separated",
    )
    command.add_argument(
        "-o", "--output", default="metabaw_annotation_result", help="result directory"
    )
    command.add_argument(
        "--tmp-files",
        help="temporary directory; relative names are created under --output (default: <output>/tmp)",
    )
    command.add_argument(
        "--delete-tmp-files",
        action="store_true",
        help="delete the temporary directory after a successful run",
    )
    command.add_argument(
        "--output-file-suffix", default=".tsv", help="suffix for generated abundance tables"
    )
    command.add_argument(
        "-t",
        "--threads",
        type=int,
        default=1,
        help=(
            "total CPU thread budget shared by concurrently running sample "
            "tasks"
        ),
    )
    command.add_argument(
        "--max-memory",
        type=float,
        default=160,
        help=(
            "maximum combined resident memory in GiB for all workflow task "
            "process trees; generated GTDB-Tk classification uses one pplacer "
            "CPU plus disk scratch and reserves 160 GiB, while a complete "
            "--gtdbtk_res skips that requirement; "
            "an undersized known configuration is rejected before tasks start"
        ),
    )
    command.add_argument(
        "--task",
        type=int,
        default=1,
        help=(
            "maximum number of read samples or MAGs processed concurrently; "
            "the total -t/--threads budget is divided evenly across these slots"
        ),
    )
    command.add_argument("--retries", type=int, default=0, help="retries after a task failure")
    command.add_argument(
        "--niche-rank",
        choices=("domain", "phylum", "class", "order", "family", "genus", "species", "strain"),
        default="family",
        help="ecological-niche level (default: family)",
    )
    command.add_argument(
        "--niche-method",
        choices=("cv", "prevalence"),
        default="cv",
        help=(
            "ecological-niche classification criterion: corrected coefficient "
            "of variation or prevalence (default: cv)"
        ),
    )
    command.add_argument(
        "--no-niche", action="store_true", help="disable ecological niche classification"
    )
    command.add_argument(
        "--no-kegg",
        dest="kegg",
        action="store_false",
        help="disable the default KEGG Orthology annotation (default: not selected)",
    )
    command.add_argument(
        "--no-cazy",
        dest="cazy",
        action="store_false",
        help="disable the default CAZy annotation (default: not selected)",
    )
    command.add_argument(
        "--no-hyd",
        dest="hydrogenase",
        action="store_false",
        help=(
            "disable the default hydrogenase and hydrogen-metabolism "
            "terminal-enzyme annotation "
            "(default: not selected)"
        ),
    )
    command.set_defaults(kegg=True, cazy=True, hydrogenase=True)
    gtdbtk_input = command.add_mutually_exclusive_group()
    gtdbtk_input.add_argument(
        "--gtdbtk-db",
        dest="gtdbtk_data",
        help="GTDB-Tk database path (default: GTDBTK_DATA_PATH environment variable)",
    )
    gtdbtk_input.add_argument(
        "--gtdbtk_res",
        help=(
            "existing GTDB-Tk output directory; validate complete taxonomy for "
            "every input MAG and skip GTDB-Tk classification"
        ),
    )
    command.add_argument(
        "--dry-run", action="store_true", help="print the task plan without executing it"
    )
    command.add_argument(
        "--force", action="store_true", help="rerun tasks even when cached outputs are valid"
    )
    command.set_defaults(func=command_annotation)

    command = subparsers.add_parser(
        "check",
        help="check software, isolated environments, databases, and reference assets",
        description=(
            "Check selected dependencies, validate database paths, and ask before "
            "installing missing software or saving replacement database locations."
        ),
        formatter_class=MetaBAWHelpFormatter,
    )
    selection = command.add_mutually_exclusive_group()
    selection.add_argument(
        "--all",
        action="store_true",
        help=(
            "check every supported program, environment, database, and reference "
            "asset; prompt for invalid or unset database paths"
        ),
    )
    selection.add_argument(
        "-e",
        "--essential",
        action="store_true",
        help="check only dependencies used by the default bin and annotation workflows",
    )
    selection.add_argument(
        "--scope",
        choices=(
            "bin",
            "binchicken",
            "annotation",
            "kegg",
            "cazy",
            "hydrogenase",
            "checkm",
            "minibwa",
            "comebin",
            "checkm2",
            "metawrap",
            "lorbin",
        ),
        help="check one dependency group or tool",
    )
    command.set_defaults(essential=True)
    command.add_argument(
        "--magscot-dir",
        default=str(configured_magscot_directory()),
        help="directory containing MAGScoT.R and hmm/ to validate and save",
    )
    command.add_argument(
        "--database-dir",
        default="~/.cache/metabaw/databases",
        help="parent directory used when downloading databases",
    )
    command.add_argument(
        "--binchicken-singlem-metapackage",
        help=(
            "SingleM metapackage file or unpacked directory for Bin Chicken planning; "
            "validated and saved for later --assembly-strategy bin-chicken runs"
        ),
    )
    command.add_argument(
        "--comebin-env",
        "--comebin-environment",
        dest="comebin_env",
        default=configured_isolated_environment("comebin"),
        help=(
            "COMEBin Python 3.7 Conda environment name or prefix to validate, "
            "install, and save for later commands"
        ),
    )
    command.add_argument(
        "--checkm2-env",
        "--checkm2-environment",
        dest="checkm2_env",
        default=configured_isolated_environment("checkm2"),
        help=(
            "CheckM2 Python 3.12 Conda environment name or prefix to validate, "
            "install, and save for later commands"
        ),
    )
    command.add_argument(
        "--metawrap-env",
        "--metawrap-environment",
        dest="metawrap_env",
        default=configured_isolated_environment("metawrap"),
        help=(
            "MetaWRAP Python 2.7 Conda environment name or prefix to validate, "
            "install, and save for later commands"
        ),
    )
    command.add_argument(
        "--lorbin-env",
        "--lorbin-environment",
        dest="lorbin_env",
        default=configured_isolated_environment("lorbin"),
        help=(
            "LorBin Python 3.10 Conda environment name or prefix to validate, "
            "install from pinned official source, and save for later commands; "
            "when explicitly supplied without --scope, select the LorBin check"
        ),
    )
    command.add_argument(
        "--binchicken-env",
        "--binchicken-environment",
        dest="binchicken_env",
        default=configured_isolated_environment("binchicken"),
        help=(
            "Bin Chicken 0.14.1 Python 3.11 Conda environment name or prefix "
            "to validate, install, and save for later coassembly planning; when "
            "explicitly supplied without --scope, select the Bin Chicken check"
        ),
    )
    command.add_argument(
        "--install-databases",
        action="store_true",
        help=(
            "download supported missing databases; legacy CheckM and the Bin Chicken "
            "SingleM metapackage must be supplied as existing paths"
        ),
    )
    command.add_argument(
        "--require-cuda",
        action="store_true",
        help=(
            "offer to repair CUDA PyTorch runtimes, then return an error unless "
            "the driver and each selected environment pass a tensor-allocation test"
        ),
    )
    command.add_argument(
        "--checkm-db",
        help=(
            "existing legacy CheckM data root containing .dmanifest, hmms/, "
            "genome_tree/, distributions/, pfam/, selected_marker_sets.tsv, and "
            "taxon_marker_sets.tsv (default: CHECKM_DATA_PATH or saved configuration)"
        ),
    )
    command.add_argument(
        "--checkm2-db",
        help=(
            "existing non-empty CheckM2 .dmnd database file or parent directory "
            "(default: CHECKM2DB or saved configuration)"
        ),
    )
    command.add_argument(
        "--gunc-db",
        help=(
            "existing non-empty GUNC .dmnd database file or parent directory "
            "(default: GUNC_DB or saved configuration)"
        ),
    )
    command.add_argument(
        "--gtdbtk-db",
        dest="gtdbtk_data",
        help=(
            "existing unpacked GTDB-Tk reference-data root (default: "
            "GTDBTK_DATA_PATH or saved configuration)"
        ),
    )
    command.add_argument(
        "--kegg-db",
        help=(
            "existing KOfam database root containing ko_list and profiles/ "
            "(default: KOFAM_DB or saved check configuration)"
        ),
    )
    command.add_argument(
        "--dbcan-db",
        help=(
            "existing run_dbCAN database root containing CAZy.dmnd, "
            "dbCAN.hmm or dbCAN.txt, dbCAN-sub.hmm or dbCAN_sub.hmm, and "
            "fam-substrate-mapping.tsv "
            "(default: DBCAN_DB or saved check configuration)"
        ),
    )
    command.add_argument(
        "--hydrogenase-db",
        help=(
            "existing hydrogenase database root containing hyddb.all.fa, "
            "FeFe.dmnd, Terminal.dmnd, and hyd_id-name.script.txt "
            "(default: HYDROGENASE_DB or saved check configuration)"
        ),
    )
    command.set_defaults(func=command_check)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_argv)
    print("MetaBAW is Running.", flush=True)
    print(f"Command: {shlex.join(['metabaw', *raw_argv])}", flush=True)
    try:
        code = args.func(args)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[{_elapsed_clock()}] [ERROR] {exc}", file=sys.stderr, flush=True)
        code = 2
    except KeyboardInterrupt:
        print(
            f"[{_elapsed_clock()}] [ABORTED] Interrupted by user",
            file=sys.stderr,
            flush=True,
        )
        code = 130
    if code == 0:
        print("ALL DONE.", flush=True)
    raise SystemExit(code)
