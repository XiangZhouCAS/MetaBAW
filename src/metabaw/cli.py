from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
from typing import Sequence

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
    SoftwareRequirement,
    all_requirements,
    annotation_requirements,
    bin_requirements,
    configured_database_path,
    configured_isolated_environment,
    configured_magscot_directory,
    confirm_install,
    cuda_runtime_status,
    host_cuda_status,
    install_checkm2_database,
    install_gtdbtk_database,
    install_gunc_database,
    install_isolated_cuda_runtime,
    install_isolated_environment,
    install_magscot,
    install_main_cuda_runtime,
    install_software,
    isolated_environment_status,
    isolated_run_prefix,
    magscot_files,
    missing_dastool_r_packages,
    missing_magscot_r_packages,
    missing_software,
    print_cuda_runtime_status,
    print_host_cuda_status,
    print_isolated_status,
    print_software_status,
    save_database_path,
    save_isolated_environment,
    save_magscot_directory,
    isolated_environment_config_path,
    software_runtime_issues,
)
from .discovery import (
    attach_contigs,
    build_analyses,
    discover_contigs,
    discover_mag_files,
    discover_reads,
    read_multi_files,
)
from .executor import Executor
from .model import Task
from .state import StateStore
from .strategy import (
    AutoResourcePlan,
    AutoStrategy,
    ServerResourceProfile,
    build_auto_analyses,
    choose_auto_resources,
    choose_auto_strategy,
    profile_server_resources,
)


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
        "path", "contig", "suffix", "separate_sample_name", "contig_suffix",
        "output", "threads", "task", "auto",
    ),
    "annotation": (
        "path", "reads", "suffix", "place_species", "read_suffix",
        "separate_sample_name", "methods", "output", "threads",
    ),
}


AUTO_OVERRIDE_OPTIONS: dict[str, tuple[str, ...]] = {
    "threads": ("-t", "--threads"),
    "task": ("--task", "--max-parallel"),
    "max_memory": ("--max-memory",),
    "gpu": ("--gpu", "--no-gpu"),
    "max_gpu_memory": ("--max-gpu-memory",),
    "batch_size": ("--batch-size",),
    "retries": ("--retries",),
    "mode": ("--single", "--multi", "--multi-files"),
    "cohort_size": ("--cohort-size",),
    "tools": ("--tools",),
    "min_contig_length": ("--min-contig-length",),
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


def _local_time() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


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
    """Return safe concurrent GPU process slots from the per-task memory budget."""
    if visible_device_count < 1 or maximum_tasks < 1:
        return 0
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    selected = list(status.devices)
    if visible:
        requested = [value.strip() for value in visible.split(",") if value.strip()]
        matched = [
            device
            for value in requested
            for device in status.devices
            if device.index == value
        ]
        if matched:
            selected = matched
    selected = selected[:visible_device_count]
    slots = 0
    for device in selected:
        budget_gib = _gpu_memory_gib(memory_budget, device.memory_total_mib)
        available_mib = (
            device.memory_free_mib
            if device.memory_free_mib is not None
            else device.memory_total_mib
        )
        available_gib = available_mib / 1024
        if budget_gib is None or budget_gib <= 0:
            slots += 1
            continue
        # Keep 15% of currently free VRAM for CUDA and framework overhead.
        slots += max(1, int((available_gib * 0.85) // budget_gib))
    return min(maximum_tasks, max(1, slots))


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


def _comebin_batch_limit(memory_gib: float) -> int:
    if memory_gib < 2:
        return 64
    if memory_gib < 4:
        return 128
    if memory_gib < 8:
        return 256
    if memory_gib < 16:
        return 512
    return 1024


def _apply_comebin_gpu_budget(
    args: argparse.Namespace,
    total_memory_mib: int | None = None,
    emit: bool = True,
) -> None:
    requested = getattr(args, "requested_batch_size", args.batch_size)
    args.requested_batch_size = requested
    args.batch_size = requested
    if not args.gpu or "comebin" not in args.tools:
        return
    memory_gib = _gpu_memory_gib(args.max_gpu_memory, total_memory_mib)
    if memory_gib is None:
        if emit:
            print(
                "[GPU] COMEBin batch size cannot be derived from a percentage until "
                "GPU memory is detected; the requested value is retained.",
                flush=True,
            )
        return
    limit = _comebin_batch_limit(memory_gib)
    args.batch_size = min(requested, limit)
    if emit and args.batch_size != requested:
        print(
            f"[GPU] COMEBin batch size capped from {requested} to {args.batch_size} "
            f"for the {args.max_gpu_memory} per-task GPU memory budget.",
            flush=True,
        )
    elif emit:
        print(
            f"[GPU] COMEBin batch size {args.batch_size} fits the conservative "
            f"{args.max_gpu_memory} memory-budget rule.",
            flush=True,
        )


def _require_bin_cuda(
    args: argparse.Namespace,
    isolated_status: dict[str, IsolatedEnvironmentStatus],
) -> None:
    host = host_cuda_status()
    print_host_cuda_status(host)
    if not host.available:
        raise RuntimeError(
            "--gpu was requested, but the NVIDIA driver layer is unavailable: "
            f"{host.error or 'unknown error'}"
        )
    runtime_entries: list[
        tuple[CudaRuntimeStatus, str | None, str | None, str | None]
    ] = []
    if {"vamb", "semibin2"} & set(args.tools):
        runtime_entries.append((_main_cuda_status(), None, None, None))
    for key in ("comebin", "lorbin"):
        if key not in args.tools:
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
                key,
                getattr(args, f"{key}_env"),
                environment_status.frontend,
            )
        )
    if not runtime_entries:
        print(
            "[GPU] None of the selected binners has a GPU execution path; "
            "--gpu has no effect.",
            flush=True,
        )
        args.cuda_device_count = 0
        args.cuda_task_slots = 0
        return
    unavailable: list[CudaRuntimeStatus] = []
    runtime_statuses: list[CudaRuntimeStatus] = []
    for status, key, environment, frontend in runtime_entries:
        print_cuda_runtime_status(status)
        if not status.available:
            status = _offer_cuda_repair(
                status,
                key,
                environment,
                frontend,
            )
        runtime_statuses.append(status)
        if not status.available:
            unavailable.append(status)
    if unavailable:
        details = "; ".join(
            f"{status.label}: {status.error or 'CUDA allocation failed'}"
            for status in unavailable
        )
        raise RuntimeError(
            "--gpu was requested, but CUDA is not usable in every selected "
            f"binner environment: {details}. Install a CUDA-enabled PyTorch "
            "build in the reported environment, then rerun `metabaw check`."
        )
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
        f"{args.cuda_task_slots} concurrent task slot(s) with the "
        f"{args.max_gpu_memory} per-task memory budget.",
        flush=True,
    )
    _apply_comebin_gpu_budget(args, _selected_gpu_memory_mib(host))


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


def _explicit_auto_value(
    args: argparse.Namespace,
    destination: str,
) -> object | None:
    """Return a value only when its auto-managed option was user supplied."""
    option_names = AUTO_OVERRIDE_OPTIONS[destination]
    if not _option_was_provided(args, *option_names):
        return None
    return getattr(args, destination)


def _require_explicit_auto_inputs(args: argparse.Namespace) -> None:
    provided = getattr(args, "_provided_options", None)
    if provided is None:
        return
    missing: list[str] = []
    if not _option_was_provided(args, "-s", "--suffix"):
        missing.append("-s/--suffix")
    if not _option_was_provided(args, "-f", "--contig-suffix"):
        missing.append("-f/--contig-suffix")
    if missing:
        raise ValueError(
            "--auto requires explicit read and contig suffixes: "
            + ", ".join(missing)
        )


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


def _resolved_auto_command(
    args: argparse.Namespace,
    output: Path,
    temp_files: Path,
) -> list[list[str]]:
    options: list[list[str]] = [
        ["metabaw", "bin"],
        ["-p", str(Path(args.path).expanduser().resolve())],
        ["-s", args.suffix],
        ["--separate-sample-name", args.separate_sample_name],
        ["--type", args.type],
        ["-c", str(Path(args.contig).expanduser().resolve())],
        ["-f", args.contig_suffix],
        ["-o", str(output)],
        ["--tmp-files", str(temp_files)],
        ["--align-tool", args.align_tool],
        ["--long-read-preset", args.long_read_preset],
        ["-t", str(args.threads)],
        ["--max-memory", f"{args.max_memory:g}"],
        ["--task", str(args.task)],
        ["--retries", str(args.retries)],
        ["--max-gpu-memory", args.max_gpu_memory],
        ["--single" if args.mode == "single" else "--multi"],
        ["--cohort-size", str(args.cohort_size)],
        ["--tools", *args.tools],
        ["--min-contig-length", str(args.min_contig_length)],
        ["--minfasta-kbs", str(args.minfasta_kbs)],
        ["--batch-size", str(args.batch_size)],
        ["--refinement", args.refinement],
        ["--quality-control", args.quality_control],
        ["--con", f"{args.con:g}"],
        ["--com", f"{args.com:g}"],
        ["--dereplication-tool", args.dereplication_tool],
        ["--ani", f"{args.ani:g}"],
        ["--min-aligned-fraction", f"{args.min_aligned_fraction:g}"],
        ["-x", args.mag_suffix],
    ]
    optional_values = (
        ("--multi-files", args.multi_files),
        ("--quality-score", args.quality_score),
        ("--trna-pass", args.trna_pass),
        ("--environment", args.environment),
    )
    for option, value in optional_values:
        if value is not None:
            options.append([option, str(value)])
    enabled_flags = (
        ("--delete-tmp-files", args.delete_tmp_files),
        ("--gpu", args.gpu),
        ("--gunc", args.gunc),
        ("--trna", args.trna),
        ("--rrna", args.rrna),
        ("--rrna-pass", args.rrna_pass),
        ("--tag-contigs", args.tag_contigs),
    )
    for option, enabled in enabled_flags:
        if enabled:
            options.append([option])
    if not args.gpu and _option_was_provided(args, "--no-gpu"):
        options.append(["--no-gpu"])
    for value in args.advanced_arg:
        options.append(["--advanced-arg", value])
    return options


def _script_parameter_value(value: object) -> str:
    if callable(value):
        return "<callable>"
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _write_auto_run_script(
    args: argparse.Namespace,
    output: Path,
    temp_files: Path,
    resources: ServerResourceProfile,
    plan: AutoResourcePlan,
    strategy: AutoStrategy,
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    script = output / "metabaw_auto_run.sh"
    resolved = {
        key: value
        for key, value in vars(args).items()
        if not key.startswith("_") and key != "func"
    }
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"# Generated by MetaBAW {__version__} at "
        f"{datetime.now(timezone.utc).isoformat()}.",
        "# This command is fully resolved and intentionally omits --auto, "
        "--dry-run, and --force so reruns use the recorded strategy safely.",
        "#",
        "# Server resource profile:",
    ]
    for key, value in resources.as_dict().items():
        lines.append(f"#   {key}={_script_parameter_value(value)}")
    lines.append("#")
    lines.append("# Automatic resource plan:")
    for key, value in plan.as_dict().items():
        lines.append(f"#   {key}={_script_parameter_value(value)}")
    lines.append("#")
    lines.append("# Automatically resolved effective parameters:")
    automatic_values = {
        "threads": args.threads,
        "concurrent_samples": args.task,
        "max_memory_gib_per_task": args.max_memory,
        "retries": args.retries,
        "gpu": args.gpu,
        "max_gpu_memory_per_task": args.max_gpu_memory,
        "comebin_batch_size_requested": args.requested_batch_size,
        "comebin_batch_size_effective": args.batch_size,
        "mode": args.mode,
        "cohort_size": args.cohort_size,
        "binners": args.tools,
        "min_contig_length": args.min_contig_length,
        "align_tool": args.align_tool,
        "temporary_directory": str(temp_files),
    }
    for key, value in automatic_values.items():
        lines.append(f"#   {key}={_script_parameter_value(value)}")
    manual_overrides = sorted(
        set(plan.manual_overrides) | set(strategy.manual_overrides)
    )
    lines.append(
        "#   manual_overrides="
        + _script_parameter_value(manual_overrides)
    )
    manual_override_values = {
        "threads": args.threads,
        "task": args.task,
        "max_memory": args.max_memory,
        "gpu": args.gpu,
        "max_gpu_memory": args.max_gpu_memory,
        "batch_size": args.requested_batch_size,
        "retries": args.retries,
        "mode": args.mode,
        "group_size": args.cohort_size,
        "tools": args.tools,
        "min_contig_length": args.min_contig_length,
    }
    lines.append("# User-provided values that overrode automatic decisions:")
    if manual_overrides:
        for key in manual_overrides:
            lines.append(
                f"#   {key}="
                f"{_script_parameter_value(manual_override_values[key])}"
            )
    else:
        lines.append("#   none")
    lines.append("#")
    lines.append(
        "# Fixed workflow policy values below are retained when the input data "
        "and server profile do not provide evidence for a safer alternative."
    )
    lines.append("#")
    lines.append("# Resolved MetaBAW parameters:")
    for key, value in sorted(resolved.items()):
        lines.append(f"#   {key}={_script_parameter_value(value)}")
    lines.extend(["", "# Reproducible resolved command:"])
    command_parts = _resolved_auto_command(args, output, temp_files)
    for index, part in enumerate(command_parts):
        prefix = "" if index == 0 else "  "
        continuation = " \\" if index < len(command_parts) - 1 else ""
        lines.append(f"{prefix}{shlex.join(part)}{continuation}")
    script.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if os.name != "nt":
        script.chmod(script.stat().st_mode | 0o111)
    return script


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
    install = install_immediately or confirm_install(
        f"Install {spec.display_name} in isolated Python {spec.python_version} "
        f"environment {environment!r}"
    )
    if install:
        return install_isolated_environment(spec, environment, status.frontend)
    raise RuntimeError(
        f"{spec.display_name} environment {environment!r} is unavailable: "
        f"{status.error or 'unknown error'}. Run `metabaw check --scope {spec.key} "
        f"{spec.option} {environment}` in an interactive terminal."
    )


def _require_database(
    name: str,
    current: Path | None,
    installer: object,
    default_directory: Path,
    size_warning: str,
) -> Path:
    if current is not None and current.exists():
        return current
    detail = str(current) if current is not None else "not configured"
    print(f"[MISSING] {name} database: {detail}", file=sys.stderr, flush=True)
    if confirm_install(f"Install {name} database under {default_directory} ({size_warning})"):
        return installer(default_directory)
    raise RuntimeError(
        f"{name} database is missing. Configure its path or install it before running."
    )


def _preflight_bin(args: argparse.Namespace) -> None:
    _ensure_software(bin_requirements(args), "bin")
    isolated_status: dict[str, IsolatedEnvironmentStatus] = {}
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
    if args.gpu:
        _require_bin_cuda(args, isolated_status)
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
        )
        if args.gunc_db or database != configured:
            save_database_path("gunc", database)
        args.gunc_db = str(database)
    if args.trna or args.rrna:
        configured = configured_database_path(
            args.gtdbtk_data, "GTDBTK_DATA_PATH", "gtdbtk"
        )
        database = _require_database(
            "GTDB-Tk",
            configured,
            install_gtdbtk_database,
            cache / "gtdbtk",
            "approximately 100 GB",
        )
        if args.gtdbtk_data or database != configured:
            save_database_path("gtdbtk", database)
        args.gtdbtk_data = str(database)


def _preflight_annotation(args: argparse.Namespace) -> None:
    _ensure_software(annotation_requirements(args), "annotation")
    cache = Path.home() / ".cache" / "metabaw" / "databases"
    configured = configured_database_path(args.gtdbtk_data, "GTDBTK_DATA_PATH", "gtdbtk")
    database = _require_database(
        "GTDB-Tk",
        configured,
        install_gtdbtk_database,
        cache / "gtdbtk",
        "approximately 100 GB",
    )
    if args.gtdbtk_data or database != configured:
        save_database_path("gtdbtk", database)
    args.gtdbtk_data = str(database)


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
    names = [
        *(requirement.executable for requirement in missing),
        *(
            f"{profile}:R:{package}"
            for profile, packages in missing_r.items()
            for package in packages
        ),
        *(f"{issue.executable}:runtime" for issue in runtime_issues),
    ]
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
    remaining = install_software(requirements)
    remaining_r = _missing_r_dependency_groups(requirements)
    remaining_runtime = software_runtime_issues(requirements)
    if remaining or remaining_r or remaining_runtime:
        names = [
            *(requirement.executable for requirement in remaining),
            *(
                f"{profile}:R:{package}"
                for profile, packages in remaining_r.items()
                for package in packages
            ),
            *(f"{issue.executable}:runtime" for issue in remaining_runtime),
        ]
        print(
            "Software remains missing after installation: " + ", ".join(names),
            file=sys.stderr,
            flush=True,
        )
        return False
    return True


def command_check(args: argparse.Namespace) -> int:
    isolated_scopes = set(ISOLATED_TOOLS)
    selected_all = args.all
    selected_essential = not selected_all and args.scope is None and args.essential
    if selected_all:
        requirements = all_requirements()
    elif args.scope == "annotation":
        requirements = annotation_requirements()
    elif args.scope in isolated_scopes:
        requirements = []
    elif selected_essential:
        requirements = _merge_requirements(
            bin_requirements(_default_bin_dependency_profile()),
            annotation_requirements(),
        )
    else:
        requirements = bin_requirements(_default_bin_dependency_profile())
    all_available = _check_software_interactively(requirements)

    selected_isolated: list[str] = []
    if selected_all:
        selected_isolated = list(ISOLATED_TOOLS)
    elif selected_essential or args.scope == "bin":
        selected_isolated = ["checkm2"]
    elif args.scope in isolated_scopes:
        selected_isolated = [args.scope]
    isolated_status: dict[str, IsolatedEnvironmentStatus] = {}
    for key in selected_isolated:
        environment = getattr(args, f"{key}_env")
        try:
            isolated_status[key] = _ensure_isolated_tool(
                ISOLATED_TOOLS[key],
                environment,
            )
        except RuntimeError as exc:
            print(f"[MISSING] {exc}", file=sys.stderr, flush=True)
            all_available = False
            continue
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
        if args.require_cuda and cuda_host.available and not status.available:
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

    database_root = Path(args.database_dir).expanduser().resolve()
    database_status: list[tuple[str, Path | None]] = []
    checks_checkm2 = (
        selected_all
        or selected_essential
        or args.scope in {"bin", "checkm2"}
    )
    checks_gunc = selected_all
    checks_gtdbtk = selected_all or selected_essential or args.scope == "annotation"
    if checks_checkm2:
        checkm2 = configured_database_path(
            args.checkm2_db,
            "CHECKM2DB",
            "checkm2",
            artifact_suffix=".dmnd",
        )
        installed = False
        if args.install_databases and (checkm2 is None or not checkm2.exists()):
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
            and checkm2.exists()
        ):
            save_database_path("checkm2", checkm2)
        database_status.append(("CheckM2", checkm2))
    if checks_gunc:
        gunc = configured_database_path(
            args.gunc_db,
            "GUNC_DB",
            "gunc",
            artifact_suffix=".dmnd",
        )
        installed = False
        if args.install_databases and (gunc is None or not gunc.exists()):
            gunc = install_gunc_database(database_root / "gunc")
            installed = True
        if (args.gunc_db or installed) and gunc is not None and gunc.exists():
            save_database_path("gunc", gunc)
        database_status.append(("GUNC", gunc))
    if checks_gtdbtk:
        gtdbtk = configured_database_path(
            args.gtdbtk_data, "GTDBTK_DATA_PATH", "gtdbtk"
        )
        installed = False
        if args.install_databases and (gtdbtk is None or not gtdbtk.exists()):
            gtdbtk = install_gtdbtk_database(database_root / "gtdbtk")
            installed = True
        if (args.gtdbtk_data or installed) and gtdbtk is not None and gtdbtk.exists():
            save_database_path("gtdbtk", gtdbtk)
        database_status.append(("GTDB-Tk", gtdbtk))
    missing_databases = []
    for name, path in database_status:
        valid = path is not None and path.exists()
        state = "OK" if valid else "MISSING"
        print(f"[{state}] {name} database: {path or 'not configured'}")
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
                "threads_per_task": args.threads,
                "concurrent_samples": args.task,
                "maximum_cpu_threads": args.threads * args.task,
                "max_memory_gb": args.max_memory,
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
        f"[{_local_time()}] [PIPELINE] Starting {payload.get('module', 'workflow')} "
        f"with {len(tasks)} tasks (threads={args.threads}, samples={args.task}, "
        f"memory={args.max_memory:g} GiB); logs={log_dir}",
        flush=True,
    )
    state = StateStore(runtime_dir / "state.sqlite3")
    try:
        executor = Executor(
            state=state,
            log_dir=log_dir,
            max_cpus=args.threads * args.task,
            max_parallel=args.task,
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
        )
        statuses = executor.run(tasks, force=args.force)
    finally:
        state.close()
    counts = Counter(statuses.values())
    summary = (
        ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))
        or "no tasks"
    )
    print(
        f"[{_local_time()}] [SUMMARY] {summary}",
        flush=True,
    )
    by_id = {task.id: task for task in tasks}
    tolerated_failures = [
        task_id
        for task_id, status in statuses.items()
        if status == "failed" and by_id[task_id].failure_tolerated
    ]
    if tolerated_failures:
        print(
            f"[{_local_time()}] [WARNING] {len(tolerated_failures)} optional "
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
    if failures:
        print(
            f"[{_local_time()}] [ERROR] Workflow failed in {len(failures)} "
            f"task(s): {_task_id_summary(failures)}; logs={log_dir}",
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
        f"[{_local_time()}] [COMPLETE] Results={output}; {temp_detail}",
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
    if args.auto:
        _require_explicit_auto_inputs(args)
    if args.cohort_size is not None and args.cohort_size < 1:
        raise ValueError("--cohort-size must be positive")
    if args.align_tool is None:
        args.align_tool = "bowtie2" if args.type == "short" else "minimap2"
    if args.type == "long" and args.align_tool == "bowtie2":
        raise ValueError("Bowtie2 is for short reads; use minimap2 or minibwa with --type long")
    if args.type == "short" and args.align_tool == "minibwa":
        raise ValueError("minibwa is exposed here for accurate long reads; use bowtie2/minimap2 for short reads")

    reads = discover_reads(args.path, args.suffix, args.type, args.separate_sample_name)
    contigs = discover_contigs(args.contig, args.contig_suffix)
    reads = attach_contigs(reads, contigs, args.contig_suffix)
    selected = None
    if args.multi_files:
        if args.mode not in {None, "multi"}:
            raise ValueError("--multi-files cannot be combined with --single")
        if args.mode is None and not args.auto:
            raise ValueError("--multi-files requires --multi unless --auto is enabled")
        args.mode = "multi"
        selected = read_multi_files(args.multi_files)
        selected_names = {sample.name for sample in selected}
        reads = selected + [
            sample for sample in reads if sample.name not in selected_names
        ]

    auto_strategy: AutoStrategy | None = None
    auto_resources: ServerResourceProfile | None = None
    auto_resource_plan: AutoResourcePlan | None = None
    if args.auto:
        auto_resources = profile_server_resources(output)
        auto_resource_plan = choose_auto_resources(
            auto_resources,
            len(reads),
            requested_threads=_explicit_auto_value(args, "threads"),
            requested_task=_explicit_auto_value(args, "task"),
            requested_max_memory_gib=_explicit_auto_value(args, "max_memory"),
            requested_gpu=_explicit_auto_value(args, "gpu"),
            requested_max_gpu_memory=_explicit_auto_value(
                args,
                "max_gpu_memory",
            ),
            requested_batch_size=_explicit_auto_value(args, "batch_size"),
            requested_retries=_explicit_auto_value(args, "retries"),
        )
        args.threads = auto_resource_plan.threads
        args.task = auto_resource_plan.task
        args.max_memory = auto_resource_plan.max_memory_gib
        args.gpu = auto_resource_plan.gpu
        args.max_gpu_memory = auto_resource_plan.max_gpu_memory
        args.batch_size = auto_resource_plan.batch_size
        args.retries = auto_resource_plan.retries
        memory_available = (
            f"{auto_resources.memory_available_gib:.1f} GiB"
            if auto_resources.memory_available_gib is not None
            else "unknown"
        )
        gpu_summary = (
            f"{auto_resources.gpu_count} device(s), "
            f"{auto_resources.gpu_free_memory_mib / 1024:.1f} GiB free"
            if auto_resources.gpu_available
            else f"unavailable ({auto_resources.cuda_error or 'not detected'})"
        )
        print(
            f"[AUTO RESOURCE] CPU total={auto_resources.cpu_total}, "
            f"affinity={auto_resources.cpu_affinity}, "
            f"currently_available={auto_resources.cpu_available}; "
            f"memory_available={memory_available}; "
            f"disk_free={auto_resources.disk_free_gib:.1f} GiB; "
            f"GPU={gpu_summary}.",
            flush=True,
        )
        print(
            f"[AUTO RESOURCE] Selected threads={args.threads}, task={args.task}, "
            f"max_memory={args.max_memory:g} GiB, gpu={args.gpu}, "
            f"max_gpu_memory={args.max_gpu_memory}, batch_size={args.batch_size}, "
            f"retries={args.retries}.",
            flush=True,
        )
        for reason in auto_resource_plan.reasons:
            print(f"[AUTO RESOURCE] {reason}", flush=True)
        auto_strategy = choose_auto_strategy(
            reads,
            read_type=args.type,
            gpu=args.gpu,
            requested_mode=_explicit_auto_value(args, "mode"),
            requested_tools=(
                explicit_tools
                if _explicit_auto_value(args, "tools") is not None
                else None
            ),
            requested_min_contig_length=_explicit_auto_value(
                args,
                "min_contig_length",
            ),
            requested_group_size=_explicit_auto_value(args, "cohort_size"),
        )
        args.mode = auto_strategy.mode
        args.cohort_size = auto_strategy.group_size
        args.min_contig_length = auto_strategy.min_contig_length
        requested_tools = list(auto_strategy.tools)
        evidence = auto_strategy.evidence
        print(
            f"[AUTO] Profiled {evidence['sample_count']} sample(s) and "
            f"{evidence['assembly_count']} unique assembly file(s): "
            f"median_N50={evidence['median_n50']}, "
            f"median_contigs={evidence['median_contig_count']}, "
            f"median_assembly_bp={evidence['median_assembly_bp']}.",
            flush=True,
        )
        print(
            f"[AUTO] Selected mode={args.mode}, group_size={auto_strategy.group_size}, "
            f"tools={','.join(requested_tools)}, "
            f"min_contig_length={args.min_contig_length}.",
            flush=True,
        )
        for reason in auto_strategy.reasons:
            print(f"[AUTO] {reason}", flush=True)
    else:
        args.mode = args.mode or "single"
        args.cohort_size = args.cohort_size or 1
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

    if selected is not None:
        analyses = build_analyses(reads, args.mode, selected)
    elif auto_strategy is not None or args.cohort_size > 1:
        analyses = build_auto_analyses(
            reads,
            args.mode,
            args.cohort_size,
        )
    else:
        analyses = build_analyses(reads, args.mode)

    args.requested_batch_size = args.batch_size
    args.cuda_device_count = 0
    args.cuda_task_slots = 0
    if auto_strategy is not None:
        _apply_comebin_gpu_budget(args, emit=False)
    mag_suffix = args.mag_suffix.lstrip(".")
    if not mag_suffix or any(char in mag_suffix for char in "/\\"):
        raise ValueError(f"Invalid MAG suffix: {args.mag_suffix!r}")
    tmp = _temporary_path(args.tmp_files, output)
    auto_script: Path | None = None
    if (
        auto_strategy is not None
        and auto_resources is not None
        and auto_resource_plan is not None
    ):
        auto_script = _write_auto_run_script(
            args,
            output,
            tmp,
            auto_resources,
            auto_resource_plan,
            auto_strategy,
        )
        print(
            f"[AUTO] Resolved run script: {auto_script}",
            flush=True,
        )
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
        _apply_comebin_gpu_budget(args)
    else:
        _preflight_bin(args)
    if (
        auto_script is not None
        and auto_resources is not None
        and auto_resource_plan is not None
    ):
        _write_auto_run_script(
            args,
            output,
            tmp,
            auto_resources,
            auto_resource_plan,
            auto_strategy,
        )
    if (
        "semibin2" in binners
        and args.environment
        and any(len(analysis.samples) > 1 for analysis in analyses)
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
        threads=args.threads,
        read_type=args.type,
        align_tool=args.align_tool,
        long_read_preset=args.long_read_preset,
        binners=tuple(binners),
        min_contig_length=args.min_contig_length,
        min_fasta_kbs=args.minfasta_kbs,
        batch_size=args.batch_size,
        refiner=args.refinement,
        quality_control=args.quality_control,
        min_completeness=args.con,
        max_contamination=args.com,
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
        max_gpu_memory=args.max_gpu_memory,
        magscot_dir=Path(args.magscot_dir).expanduser().resolve(),
        checkm2_db=Path(args.checkm2_db).expanduser().resolve() if args.checkm2_db else None,
        gunc_db=Path(args.gunc_db).expanduser().resolve() if args.gunc_db else None,
        gtdbtk_data=Path(args.gtdbtk_data).expanduser().resolve() if args.gtdbtk_data else None,
        extra_args=_parse_advanced(args.advanced_arg),
        comebin_run_prefix=isolated_run_prefix(
            args.comebin_env,
            getattr(args, "comebin_frontend", None),
        ),
        checkm2_run_prefix=isolated_run_prefix(
            args.checkm2_env,
            getattr(args, "checkm2_frontend", None),
        ),
        metawrap_run_prefix=isolated_run_prefix(
            args.metawrap_env,
            getattr(args, "metawrap_frontend", None),
        ),
        lorbin_run_prefix=isolated_run_prefix(
            args.lorbin_env,
            getattr(args, "lorbin_frontend", None),
        ),
    )
    tasks = DirectBinBuilder(analyses, options, options.workdir).build()
    if auto_strategy is not None:
        auto_payload = auto_strategy.as_dict()
        auto_payload.update(
            {
                "server_resources": (
                    auto_resources.as_dict()
                    if auto_resources is not None
                    else {}
                ),
                "resource_plan": (
                    auto_resource_plan.as_dict()
                    if auto_resource_plan is not None
                    else {}
                ),
                "resolved_run_script": (
                    str(auto_script) if auto_script is not None else None
                ),
            }
        )
    else:
        auto_payload = {"enabled": False}
    payload = {
        "module": "bin",
        "auto_strategy": auto_payload,
        "mode": args.mode,
        "read_type": args.type,
        "align_tool": args.align_tool,
        "binners": binners,
        "refinement": args.refinement,
        "quality_control": args.quality_control,
        "dereplication_tool": args.dereplication_tool,
        "gpu": {
            "enabled": args.gpu,
            "visible_device_count": args.cuda_device_count,
            "concurrent_task_slots": args.cuda_task_slots,
            "memory_budget_per_task": args.max_gpu_memory,
            "comebin_batch_size_requested": args.requested_batch_size,
            "comebin_batch_size_effective": args.batch_size,
        },
        "isolated_environments": {
            key: environment
            for key, environment in selected_environments
        },
        "analyses": [
            {
                "name": analysis.name,
                "combined": analysis.combined,
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
    if not args.dry_run:
        _preflight_annotation(args)
    mags = discover_mag_files(args.path, args.suffix)
    mag_dir = Path(args.path).expanduser().resolve()
    if mag_dir.is_file():
        raise ValueError("metabaw annotation currently requires -p/--path to be a MAG directory")
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
    suffix = args.output_file_suffix
    if not suffix.startswith("."):
        suffix = "." + suffix
    output = Path(args.output).expanduser().resolve()
    tmp = _temporary_path(args.tmp_files, output)
    options = AnnotationOptions(
        mag_dir=mag_dir,
        mag_suffix=args.suffix,
        reads=tuple(reads),
        output=output,
        output_suffix=suffix,
        threads=args.threads,
        read_type=args.type,
        methods=tuple(methods),
        place_species=args.place_species,
        niche_rank=args.niche_rank,
        no_niche=args.no_niche,
        gtdbtk_data=Path(args.gtdbtk_data).expanduser().resolve() if args.gtdbtk_data else None,
    )
    tasks = AnnotationBuilder(options, tmp / "work").build()
    payload = {
        "module": "annotation",
        "mags": [str(path) for path in mags],
        "samples": [sample.name for sample in reads],
        "coverm_methods": methods,
        "niche_rank": args.niche_rank,
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
    command.add_argument("-p", "--path", required=True, help="read file or directory")
    command.add_argument("-c", "--contig", required=True, help="contig file or directory")
    command.add_argument("-s", "--suffix", default="fastq.gz", help="read suffix (default: fastq.gz)")
    command.add_argument(
        "--separate-sample-name",
        metavar="SEP",
        default=".",
        help=(
            "sample name separator for names not matching the standard "
            "_R1/_R2/_1/_2/.1/.2 pair patterns; the part of the read filename "
            "before the first SEP becomes the sample name (default: '.', so "
            "B425.1.fastq.gz and B425.2.fastq.gz form sample B425)"
        ),
    )
    command.add_argument(
        "--type",
        choices=("short", "long"),
        default="short",
        help="read technology type",
    )
    command.add_argument(
        "--auto",
        action="store_true",
        help=(
            "after explicit -p/-s/-c/-f inputs, scan current CPU, RAM, disk, "
            "GPU and free resources; select all remaining runtime and binning "
            "parameters; write <output>/metabaw_auto_run.sh; explicit optional "
            "values take precedence"
        ),
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
        "-f", "--contig-suffix", default="fa", help="contig suffix when --contig is a directory"
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
        help="CPU threads assigned to each running sample task",
    )
    command.add_argument(
        "--max-memory",
        type=float,
        default=100,
        help="maximum virtual memory per task in GiB; 100 means 100 GiB",
    )
    command.add_argument(
        "--task",
        type=int,
        default=1,
        help="maximum number of samples processed concurrently",
    )
    command.add_argument(
        "--max-parallel",
        dest="task",
        type=int,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    command.add_argument("--retries", type=int, default=0, help="retries after a task failure")
    gpu = command.add_mutually_exclusive_group()
    gpu.add_argument(
        "--gpu",
        dest="gpu",
        action="store_true",
        help="enable supported GPU execution paths",
    )
    gpu.add_argument(
        "--no-gpu",
        dest="gpu",
        action="store_false",
        help="disable GPU execution, including when --auto detects a usable GPU",
    )
    command.set_defaults(gpu=False)
    command.add_argument(
        "--max-gpu-memory",
        default="4G",
        help=(
            "per-task GPU memory budget used to cap COMEBin batch size; "
            "accepts bytes, K/M/G/T/P units, or a percentage"
        ),
    )
    mode = command.add_mutually_exclusive_group()
    mode.add_argument(
        "--single",
        dest="mode",
        action="store_const",
        const="single",
        help="process samples independently (default: selected unless --auto is enabled)",
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
        help="explicit multi group: READ1[,READ2] TAB CONTIGS, optionally prefixed by SAMPLE TAB",
    )
    command.add_argument(
        "--cohort-size",
        type=int,
        help=(
            "maximum number of per-sample assemblies in one multi-sample "
            "coverage cohort; automatically resolved and recorded by --auto"
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
        help=(
            "minimum contig length in base pairs (default: 1500; --auto may "
            "select 2000 for high-continuity assemblies)"
        ),
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
        default=1024,
        help=(
            "requested COMEBin training batch size; --gpu may lower it to fit "
            "--max-gpu-memory"
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
    command.add_argument("--con", type=float, default=50.0, help="minimum completeness (%%)")
    command.add_argument("--com", type=float, default=10.0, help="maximum contamination (%%)")
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
        help="SemiBin2 pretrained environment model; omit to use automatic behavior",
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
        help="run GTDB-Tk taxonomy, CoverM profiling, and niche classification",
        description="Classify MAGs, quantify abundance, merge metrics and classify ecological niches.",
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
    command.add_argument(
        "--place-species",
        dest="place_species",
        action="store_true",
        default=True,
        help="enable GTDB-Tk species placement",
    )
    command.add_argument(
        "--no-place-species",
        dest="place_species",
        action="store_false",
        help="disable GTDB-Tk species placement (default: species placement enabled)",
    )
    command.add_argument(
        "-f", "--read-suffix", default="fastq.gz", help="read filename suffix"
    )
    command.add_argument(
        "--separate-sample-name",
        metavar="SEP",
        default=".",
        help=(
            "sample name separator for names not matching the standard "
            "_R1/_R2/_1/_2/.1/.2 pair patterns; the part of the read filename "
            "before the first SEP becomes the sample name (default: '.', so "
            "B425.1.fastq.gz and B425.2.fastq.gz form sample B425)"
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
        help="CPU threads assigned to each running sample task",
    )
    command.add_argument(
        "--max-memory",
        type=float,
        default=100,
        help="maximum virtual memory per task in GiB; 100 means 100 GiB",
    )
    command.add_argument(
        "--task",
        type=int,
        default=1,
        help="maximum number of samples processed concurrently",
    )
    command.add_argument(
        "--max-parallel",
        dest="task",
        type=int,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    command.add_argument("--retries", type=int, default=0, help="retries after a task failure")
    command.add_argument(
        "--niche-rank",
        choices=("domain", "phylum", "class", "order", "family", "genus", "species", "strain"),
        default="family",
        help="ecological-niche level (default: family)",
    )
    command.add_argument(
        "--no-niche", action="store_true", help="disable ecological niche classification"
    )
    command.add_argument(
        "--gtdbtk-db",
        dest="gtdbtk_data",
        help="GTDB-Tk database path (default: GTDBTK_DATA_PATH environment variable)",
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
        help="check and install external software and report database status",
        description=(
            "Check selected dependencies and ask before installing missing software "
            "with Mamba/Conda, CRAN, or Bioconductor as appropriate."
        ),
        formatter_class=MetaBAWHelpFormatter,
    )
    selection = command.add_mutually_exclusive_group()
    selection.add_argument(
        "--all",
        action="store_true",
        help="check all supported dependencies and offer to install every missing program",
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
            "annotation",
            "comebin",
            "checkm2",
            "metawrap",
            "lorbin",
        ),
        help="check one dependency group or isolated tool",
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
            "install from pinned official source, and save for later commands"
        ),
    )
    command.add_argument(
        "--install-databases",
        action="store_true",
        help="download missing selected databases; GTDB-Tk requires about 100 GB",
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
        "--checkm2-db",
        help="existing CheckM2 database path (default: CHECKM2DB environment variable)",
    )
    command.add_argument(
        "--gunc-db",
        help="existing GUNC database path (default: GUNC_DB environment variable)",
    )
    command.add_argument(
        "--gtdbtk-db",
        dest="gtdbtk_data",
        help="existing GTDB-Tk data path (default: GTDBTK_DATA_PATH environment variable)",
    )
    command.set_defaults(func=command_check)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_argv)
    try:
        code = args.func(args)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[{_local_time()}] [ERROR] {exc}", file=sys.stderr, flush=True)
        code = 2
    except KeyboardInterrupt:
        print(
            f"[{_local_time()}] [ABORTED] Interrupted by user",
            file=sys.stderr,
            flush=True,
        )
        code = 130
    raise SystemExit(code)
