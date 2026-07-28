from __future__ import annotations

import ctypes
from dataclasses import dataclass
import gzip
import math
import os
from pathlib import Path
import shutil
import statistics
from typing import Iterable

from .dependencies import HostCudaStatus, host_cuda_status
from .discovery import Analysis, ReadSample, build_analyses


@dataclass(frozen=True)
class AssemblyMetrics:
    path: Path
    contigs: int
    total_bp: int
    n50: int
    contigs_at_least_1500: int
    contigs_at_least_2000: int

    def as_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "contigs": self.contigs,
            "total_bp": self.total_bp,
            "n50": self.n50,
            "contigs_at_least_1500": self.contigs_at_least_1500,
            "contigs_at_least_2000": self.contigs_at_least_2000,
        }


@dataclass(frozen=True)
class AutoStrategy:
    mode: str
    tools: tuple[str, ...]
    min_contig_length: int
    group_size: int
    evidence: dict[str, object]
    reasons: tuple[str, ...]
    manual_overrides: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "enabled": True,
            "policy_version": 1,
            "mode": self.mode,
            "tools": list(self.tools),
            "min_contig_length": self.min_contig_length,
            "group_size": self.group_size,
            "evidence": self.evidence,
            "reasons": list(self.reasons),
            "manual_overrides": list(self.manual_overrides),
        }


@dataclass(frozen=True)
class ServerResourceProfile:
    cpu_total: int
    cpu_affinity: int
    cpu_load_1m: float | None
    cpu_available: int
    memory_total_gib: float | None
    memory_available_gib: float | None
    disk_free_gib: float
    disk_path: Path
    gpu_available: bool
    gpu_count: int
    gpu_total_memory_mib: int
    gpu_free_memory_mib: int
    gpu_names: tuple[str, ...]
    cuda_driver_version: str | None
    advertised_cuda_version: str | None
    cuda_error: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "cpu_total": self.cpu_total,
            "cpu_affinity": self.cpu_affinity,
            "cpu_load_1m": self.cpu_load_1m,
            "cpu_available": self.cpu_available,
            "memory_total_gib": self.memory_total_gib,
            "memory_available_gib": self.memory_available_gib,
            "disk_free_gib": self.disk_free_gib,
            "disk_path": str(self.disk_path),
            "gpu_available": self.gpu_available,
            "gpu_count": self.gpu_count,
            "gpu_total_memory_mib": self.gpu_total_memory_mib,
            "gpu_free_memory_mib": self.gpu_free_memory_mib,
            "gpu_names": list(self.gpu_names),
            "cuda_driver_version": self.cuda_driver_version,
            "advertised_cuda_version": self.advertised_cuda_version,
            "cuda_error": self.cuda_error,
        }


@dataclass(frozen=True)
class AutoResourcePlan:
    threads: int
    task: int
    max_memory_gib: float
    gpu: bool
    max_gpu_memory: str
    batch_size: int
    retries: int
    reasons: tuple[str, ...]
    manual_overrides: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "threads": self.threads,
            "task": self.task,
            "max_memory_gib": self.max_memory_gib,
            "gpu": self.gpu,
            "max_gpu_memory": self.max_gpu_memory,
            "batch_size": self.batch_size,
            "retries": self.retries,
            "reasons": list(self.reasons),
            "manual_overrides": list(self.manual_overrides),
        }


def _memory_from_proc() -> tuple[float | None, float | None]:
    path = Path("/proc/meminfo")
    if not path.is_file():
        return None, None
    values: dict[str, int] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            key, raw = line.split(":", 1)
            values[key] = int(raw.strip().split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        return None, None
    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    divisor = 1024**3
    return (
        total / divisor if total is not None else None,
        available / divisor if available is not None else None,
    )


def _memory_from_windows() -> tuple[float | None, float | None]:
    if os.name != "nt":
        return None, None

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ulong),
            ("memory_load", ctypes.c_ulong),
            ("total_physical", ctypes.c_ulonglong),
            ("available_physical", ctypes.c_ulonglong),
            ("total_page_file", ctypes.c_ulonglong),
            ("available_page_file", ctypes.c_ulonglong),
            ("total_virtual", ctypes.c_ulonglong),
            ("available_virtual", ctypes.c_ulonglong),
            ("available_extended_virtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.length = ctypes.sizeof(MemoryStatus)
    try:
        success = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    except (AttributeError, OSError):
        return None, None
    if not success:
        return None, None
    divisor = 1024**3
    return status.total_physical / divisor, status.available_physical / divisor


def _memory_from_sysconf() -> tuple[float | None, float | None]:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        total_pages = os.sysconf("SC_PHYS_PAGES")
        available_pages = os.sysconf("SC_AVPHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None, None
    divisor = 1024**3
    return (
        page_size * total_pages / divisor,
        page_size * available_pages / divisor,
    )


def _nearest_existing_path(path: Path) -> Path:
    current = path.expanduser().resolve()
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def profile_server_resources(
    output: Path,
    cuda_status: HostCudaStatus | None = None,
) -> ServerResourceProfile:
    cpu_total = max(1, os.cpu_count() or 1)
    try:
        cpu_affinity = max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        cpu_affinity = cpu_total
    try:
        cpu_load_1m = max(0.0, float(os.getloadavg()[0]))
    except (AttributeError, OSError):
        cpu_load_1m = None
    cpu_available = (
        max(1, cpu_affinity - math.ceil(cpu_load_1m))
        if cpu_load_1m is not None
        else cpu_affinity
    )

    memory_total, memory_available = _memory_from_proc()
    if memory_total is None:
        memory_total, memory_available = _memory_from_windows()
    if memory_total is None:
        memory_total, memory_available = _memory_from_sysconf()

    disk_path = _nearest_existing_path(output)
    disk_free_gib = shutil.disk_usage(disk_path).free / 1024**3
    cuda = cuda_status or host_cuda_status()
    devices = cuda.devices if cuda.available else ()
    gpu_total_memory_mib = sum(device.memory_total_mib for device in devices)
    gpu_free_memory_mib = sum(
        device.memory_free_mib
        if device.memory_free_mib is not None
        else device.memory_total_mib
        for device in devices
    )
    driver_versions = tuple(
        dict.fromkeys(device.driver_version for device in devices)
    )
    return ServerResourceProfile(
        cpu_total=cpu_total,
        cpu_affinity=cpu_affinity,
        cpu_load_1m=cpu_load_1m,
        cpu_available=cpu_available,
        memory_total_gib=memory_total,
        memory_available_gib=memory_available,
        disk_free_gib=disk_free_gib,
        disk_path=disk_path,
        gpu_available=bool(devices),
        gpu_count=len(devices),
        gpu_total_memory_mib=gpu_total_memory_mib,
        gpu_free_memory_mib=gpu_free_memory_mib,
        gpu_names=tuple(device.name for device in devices),
        cuda_driver_version=",".join(driver_versions) if driver_versions else None,
        advertised_cuda_version=cuda.advertised_cuda_version,
        cuda_error=cuda.error,
    )


def choose_auto_resources(
    profile: ServerResourceProfile,
    sample_count: int,
    *,
    requested_threads: int | None = None,
    requested_task: int | None = None,
    requested_max_memory_gib: float | None = None,
    requested_gpu: bool | None = None,
    requested_max_gpu_memory: str | None = None,
    requested_batch_size: int | None = None,
    requested_retries: int | None = None,
) -> AutoResourcePlan:
    if sample_count < 1:
        raise ValueError("Automatic resource selection requires at least one sample")
    reasons: list[str] = []
    overrides: list[str] = []
    cpu_available = max(1, profile.cpu_available)
    memory_available = profile.memory_available_gib

    if requested_task is not None:
        if requested_task < 1:
            raise ValueError("Explicit automatic task count must be positive")
        task = requested_task
        overrides.append("task")
        reasons.append("Kept the explicitly requested concurrent sample count.")
    else:
        cpu_per_task = (
            max(1, requested_threads)
            if requested_threads is not None
            else 8
        )
        task_by_cpu = max(1, cpu_available // cpu_per_task)
        task_by_memory = (
            max(1, int(memory_available * 0.80 // 24))
            if memory_available is not None
            else sample_count
        )
        task = min(sample_count, task_by_cpu, task_by_memory, 8)
        reasons.append(
            "Selected concurrent samples from currently available CPU and memory."
        )

    if requested_threads is not None:
        if requested_threads < 1:
            raise ValueError("Explicit automatic thread count must be positive")
        threads = requested_threads
        overrides.append("threads")
        reasons.append("Kept the explicitly requested threads per sample.")
    else:
        threads = max(1, cpu_available // task)
        reasons.append(
            "Allocated available CPU capacity across the selected concurrent samples."
        )

    if requested_max_memory_gib is not None:
        max_memory_gib = requested_max_memory_gib
        overrides.append("max_memory")
        reasons.append("Kept the explicitly requested per-task memory ceiling.")
    elif memory_available is not None:
        max_memory_gib = max(
            1.0,
            math.floor(memory_available * 0.85 / task),
        )
        reasons.append(
            "Reserved 15% of currently available RAM and divided the remainder "
            "between concurrent samples."
        )
    else:
        max_memory_gib = 100.0
        reasons.append(
            "Used the 100 GiB ceiling because available physical memory could not "
            "be detected."
        )

    if requested_gpu is not None:
        gpu = requested_gpu
        overrides.append("gpu")
        reasons.append("Kept the explicitly requested GPU setting.")
    else:
        gpu = profile.gpu_available and profile.gpu_free_memory_mib >= 2048
        reasons.append(
            "Enabled GPU execution because an NVIDIA device with at least 2 GiB "
            "free memory is currently available."
            if gpu
            else "Kept GPU execution disabled because usable free NVIDIA memory "
            "was not detected."
        )

    if requested_max_gpu_memory is not None:
        max_gpu_memory = requested_max_gpu_memory
        overrides.append("max_gpu_memory")
    elif gpu:
        per_task_free_mib = max(
            1024,
            int(profile.gpu_free_memory_mib * 0.85 / max(1, task)),
        )
        selected_mib = min(4096, per_task_free_mib)
        max_gpu_memory = (
            "4G" if selected_mib >= 4096 else f"{selected_mib}M"
        )
        reasons.append(
            "Capped per-task GPU memory at 4 GiB and kept 15% of currently free "
            "VRAM in reserve."
        )
    else:
        max_gpu_memory = "4G"

    if requested_batch_size is not None:
        batch_size = requested_batch_size
        overrides.append("batch_size")
    else:
        batch_size = 1024
    if requested_retries is not None:
        retries = requested_retries
        overrides.append("retries")
    else:
        retries = 1
        reasons.append("Enabled one automatic retry for transient task failures.")

    return AutoResourcePlan(
        threads=threads,
        task=task,
        max_memory_gib=max_memory_gib,
        gpu=gpu,
        max_gpu_memory=max_gpu_memory,
        batch_size=batch_size,
        retries=retries,
        reasons=tuple(reasons),
        manual_overrides=tuple(overrides),
    )


def _fasta_lengths(path: Path) -> list[int]:
    opener = gzip.open if path.name.lower().endswith(".gz") else open
    lengths: list[int] = []
    current: int | None = None
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if len(line) == 1:
                    raise ValueError(f"{path}:{line_number}: empty FASTA header")
                if current is not None:
                    if current == 0:
                        raise ValueError(
                            f"{path}:{line_number}: FASTA record has no sequence"
                        )
                    lengths.append(current)
                current = 0
            else:
                if current is None:
                    raise ValueError(
                        f"{path}:{line_number}: sequence appears before FASTA header"
                    )
                current += len(line)
    if current is not None:
        if current == 0:
            raise ValueError(f"{path}: final FASTA record has no sequence")
        lengths.append(current)
    if not lengths:
        raise ValueError(f"{path}: no FASTA records were found")
    return lengths


def _n50(lengths: Iterable[int]) -> int:
    ordered = sorted(lengths, reverse=True)
    threshold = sum(ordered) / 2
    cumulative = 0
    for length in ordered:
        cumulative += length
        if cumulative >= threshold:
            return length
    return 0


def profile_assemblies(samples: list[ReadSample]) -> tuple[AssemblyMetrics, ...]:
    paths = sorted(
        {
            sample.contigs.resolve()
            for sample in samples
            if sample.contigs is not None
        }
    )
    if not paths:
        raise ValueError("No sample assemblies are available for automatic strategy selection")
    profiles: list[AssemblyMetrics] = []
    for path in paths:
        lengths = _fasta_lengths(path)
        profiles.append(
            AssemblyMetrics(
                path=path,
                contigs=len(lengths),
                total_bp=sum(lengths),
                n50=_n50(lengths),
                contigs_at_least_1500=sum(length >= 1500 for length in lengths),
                contigs_at_least_2000=sum(length >= 2000 for length in lengths),
            )
        )
    return tuple(profiles)


def choose_auto_strategy(
    samples: list[ReadSample],
    read_type: str,
    gpu: bool,
    requested_mode: str | None,
    requested_tools: list[str] | None,
    requested_min_contig_length: int | None,
    requested_group_size: int | None = None,
    maximum_group_size: int = 20,
) -> AutoStrategy:
    if not samples:
        raise ValueError("Automatic strategy selection requires at least one sample")
    profiles = profile_assemblies(samples)
    sample_count = len(samples)
    shared_assembly = len(profiles) == 1 and sample_count > 1
    if shared_assembly:
        raise ValueError(
            "Automatic strategy selection currently supports one independently "
            "assembled contig file per sample. Shared co-assembly input is not "
            "enabled."
        )
    median_n50 = int(statistics.median(profile.n50 for profile in profiles))
    median_total_bp = int(
        statistics.median(profile.total_bp for profile in profiles)
    )
    median_contigs = int(
        statistics.median(profile.contigs for profile in profiles)
    )
    complex_assembly = median_total_bp >= 100_000_000 or median_contigs >= 50_000
    evidence: dict[str, object] = {
        "sample_count": sample_count,
        "assembly_count": len(profiles),
        "shared_assembly": shared_assembly,
        "paired_sample_fraction": (
            sum(sample.read2 is not None for sample in samples) / sample_count
        ),
        "median_assembly_bp": median_total_bp,
        "median_contig_count": median_contigs,
        "median_n50": median_n50,
        "complex_assembly": complex_assembly,
        "assemblies": [profile.as_dict() for profile in profiles],
    }
    reasons: list[str] = []
    overrides: list[str] = []

    if requested_mode is not None:
        mode = requested_mode
        overrides.append("mode")
        reasons.append(f"Kept the explicitly requested {mode}-sample mode.")
    elif sample_count >= 3:
        mode = "multi"
        reasons.append(
            "Selected differential-coverage multi-sample binning for a cohort of "
            f"{sample_count} samples."
        )
    else:
        mode = "single"
        reasons.append(
            "Selected independent binning because fewer than three samples were supplied."
        )

    if requested_group_size is not None:
        if requested_group_size < 1:
            raise ValueError("Explicit automatic cohort size must be positive")
        group_size = requested_group_size
        overrides.append("group_size")
        reasons.append(
            "Kept the explicitly requested maximum cohort group size."
        )
    elif mode == "multi":
        group_size = min(maximum_group_size, sample_count)
        if sample_count > maximum_group_size:
            reasons.append(
                f"Limited differential-coverage groups to {maximum_group_size} samples "
                "to avoid oversized cohorts."
            )
    else:
        group_size = 1

    if requested_tools is not None:
        tools = tuple(requested_tools)
        overrides.append("tools")
        reasons.append("Kept the explicitly requested binner set.")
    elif read_type == "long":
        selected = ["metadecoder", "vamb", "lorbin"]
        if sample_count >= 3 or complex_assembly:
            selected.append("semibin2")
        tools = tuple(selected)
        reasons.append(
            "Selected LorBin with composition/coverage binners for long-read data."
        )
    else:
        selected = ["metabat2", "metadecoder", "vamb"]
        if sample_count >= 3 or complex_assembly:
            selected.append("semibin2")
        if gpu and (sample_count >= 3 or complex_assembly):
            selected.append("comebin")
        tools = tuple(selected)
        reasons.append(
            "Selected complementary composition, coverage, and representation-learning "
            "binners for short-read data."
        )
        if gpu and "comebin" in selected:
            reasons.append(
                "Added COMEBin because CUDA was requested and the cohort or assembly "
                "complexity justifies the additional model."
            )

    if requested_min_contig_length is not None:
        min_contig_length = requested_min_contig_length
        overrides.append("min_contig_length")
        reasons.append("Kept the explicitly requested minimum contig length.")
    elif median_n50 >= 20_000 and all(
        profile.contigs_at_least_2000 >= 100 for profile in profiles
    ):
        min_contig_length = 2000
        reasons.append(
            "Selected a 2,000 bp contig threshold because assembly continuity is high."
        )
    else:
        min_contig_length = 1500
        reasons.append(
            "Selected the conservative 1,500 bp contig threshold for fragmented or "
            "moderate-continuity assemblies."
        )

    return AutoStrategy(
        mode=mode,
        tools=tools,
        min_contig_length=min_contig_length,
        group_size=group_size,
        evidence=evidence,
        reasons=tuple(reasons),
        manual_overrides=tuple(overrides),
    )


def build_auto_analyses(
    samples: list[ReadSample],
    mode: str,
    group_size: int,
) -> list[Analysis]:
    if mode != "multi" or len(samples) <= group_size:
        return build_analyses(samples, mode)
    contigs = {sample.contigs for sample in samples}
    if len(contigs) == 1:
        return build_analyses(samples, "multi")
    analyses: list[Analysis] = []
    for start in range(0, len(samples), group_size):
        group = samples[start : start + group_size]
        analyses.extend(
            build_analyses(group, "multi" if len(group) > 1 else "single")
        )
    return analyses
