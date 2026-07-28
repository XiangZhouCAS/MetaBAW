from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from typing import Iterable, Sequence


@dataclass(frozen=True)
class SoftwareRequirement:
    executable: str
    package: str
    purpose: str


@dataclass(frozen=True)
class SoftwareRuntimeIssue:
    executable: str
    detail: str
    repair_packages: tuple[str, ...]


@dataclass(frozen=True)
class RPackageRequirement:
    name: str
    source: str = "cran"


@dataclass(frozen=True)
class IsolatedToolSpec:
    key: str
    display_name: str
    executable: str
    package: str
    python_version: str
    default_environment: str
    option: str
    purpose: str
    conda_packages: tuple[str, ...] = ()
    channels: tuple[str, ...] = ("conda-forge", "bioconda")
    pip_source: str | None = None


ISOLATED_TOOLS: dict[str, IsolatedToolSpec] = {
    "comebin": IsolatedToolSpec(
        key="comebin",
        display_name="COMEBin",
        executable="run_comebin.sh",
        package="comebin",
        python_version="3.7",
        default_environment="metabaw-comebin-py37",
        option="--comebin-env",
        purpose="COMEBin binning",
    ),
    "checkm2": IsolatedToolSpec(
        key="checkm2",
        display_name="CheckM2",
        executable="checkm2",
        package="checkm2",
        python_version="3.12",
        default_environment="metabaw-checkm2-py312",
        option="--checkm2-env",
        purpose="CheckM2 quality estimation",
    ),
    "metawrap": IsolatedToolSpec(
        key="metawrap",
        display_name="MetaWRAP",
        executable="metawrap",
        package="metawrap-refinement",
        python_version="2.7",
        default_environment="metabaw-metawrap-py27",
        option="--metawrap-env",
        purpose="MetaWRAP bin refinement",
    ),
    "lorbin": IsolatedToolSpec(
        key="lorbin",
        display_name="LorBin",
        executable="LorBin",
        package="LorBin official source ee10232",
        python_version="3.10",
        default_environment="metabaw-lorbin-py310",
        option="--lorbin-env",
        purpose="long-read metagenomic binning",
        conda_packages=(
            # LorBin documents Biopython 1.78, but Conda does not provide a
            # Python 3.10 build of that release. Biopython 1.83 retains the
            # SeqIO interfaces used by LorBin and has Python 3.10 packages.
            "biopython=1.83",
            "hmmer",
            "prodigal",
            "bedtools",
            "samtools",
            "pytorch=1.11.0",
            "torchvision=0.12.0",
            "torchaudio=0.11.0",
            "numpy=1.23.3",
            "pip=22.2.2",
            "setuptools=65.5.0",
            "pandas=2.2.2",
            "scikit-learn=1.1.2",
            "scipy=1.13.1",
            "joblib=1.4.2",
        ),
        channels=("bioconda", "pytorch", "conda-forge"),
        pip_source=(
            "https://github.com/LorMeBioAI/LorBin/archive/"
            "ee10232282c2b71ed3ce2a34d5dbd78af3dd0b0a.tar.gz"
        ),
    ),
}

COMEBIN_ENV_DEFAULT = ISOLATED_TOOLS["comebin"].default_environment
COMEBIN_PYTHON_VERSION = ISOLATED_TOOLS["comebin"].python_version
CHECKM2_ENV_DEFAULT = ISOLATED_TOOLS["checkm2"].default_environment
METAWRAP_ENV_DEFAULT = ISOLATED_TOOLS["metawrap"].default_environment
LORBIN_ENV_DEFAULT = ISOLATED_TOOLS["lorbin"].default_environment
COMEBIN_REQUIREMENT = SoftwareRequirement(
    ISOLATED_TOOLS["comebin"].executable,
    ISOLATED_TOOLS["comebin"].package,
    ISOLATED_TOOLS["comebin"].purpose,
)

ISOLATED_CUDA_PACKAGES: dict[
    str,
    tuple[tuple[str, ...], tuple[str, ...]],
] = {
    "comebin": (
        ("pytorch", "conda-forge", "bioconda"),
        (
            "pytorch=1.10.2=py3.7_cuda11.1_cudnn8.0.5_0",
            "cudatoolkit=11.1.1",
            "pytorch-mutex=1.0=cuda",
        ),
    ),
    "lorbin": (
        ("pytorch", "conda-forge", "bioconda"),
        (
            "pytorch=1.11.0",
            "torchvision=0.12.0",
            "torchaudio=0.11.0",
            "cudatoolkit=11.3",
            "pytorch-mutex=1.0=cuda",
        ),
    ),
}


@dataclass(frozen=True)
class IsolatedEnvironmentStatus:
    spec: IsolatedToolSpec
    environment: str
    frontend: str | None
    python_version: str | None
    executable: str | None
    error: str | None = None

    @property
    def available(self) -> bool:
        return (
            self.frontend is not None
            and self.python_version == self.spec.python_version
            and self.executable is not None
        )


COMEBinEnvironmentStatus = IsolatedEnvironmentStatus


@dataclass(frozen=True)
class CudaDevice:
    index: str
    name: str
    memory_total_mib: int
    driver_version: str
    memory_free_mib: int | None = None


@dataclass(frozen=True)
class HostCudaStatus:
    executable: str | None
    devices: tuple[CudaDevice, ...]
    advertised_cuda_version: str | None
    error: str | None = None

    @property
    def available(self) -> bool:
        return self.executable is not None and bool(self.devices) and self.error is None


@dataclass(frozen=True)
class CudaRuntimeStatus:
    label: str
    python_version: str | None
    torch_version: str | None
    torch_cuda_version: str | None
    cuda_visible_devices: str | None
    device_count: int
    devices: tuple[str, ...]
    allocation_test: bool
    error: str | None = None

    @property
    def available(self) -> bool:
        return (
            self.error is None
            and self.device_count > 0
            and self.allocation_test
        )


SOFTWARE: dict[str, SoftwareRequirement] = {
    "bash": SoftwareRequirement("bash", "bash", "task shell"),
    "bowtie2": SoftwareRequirement("bowtie2", "bowtie2", "short-read mapping"),
    "bowtie2-build": SoftwareRequirement("bowtie2-build", "bowtie2", "Bowtie2 indexing"),
    "minimap2": SoftwareRequirement("minimap2", "minimap2", "long-read or CoverM mapping"),
    "minibwa": SoftwareRequirement("minibwa", "minibwa", "accurate long-read mapping"),
    "samtools": SoftwareRequirement("samtools", "samtools", "BAM filtering, sorting, and indexing"),
    "jgi_summarize_bam_contig_depths": SoftwareRequirement(
        "jgi_summarize_bam_contig_depths", "metabat2", "MetaBAT2 depth calculation"
    ),
    "metabat2": SoftwareRequirement("metabat2", "metabat2", "MetaBAT2 binning"),
    "strobealign": SoftwareRequirement("strobealign", "strobealign", "short-read VAMB abundance"),
    "vamb": SoftwareRequirement("vamb", "vamb", "VAMB binning"),
    "metadecoder": SoftwareRequirement("metadecoder", "metadecoder", "MetaDecoder binning"),
    "SemiBin2": SoftwareRequirement("SemiBin2", "semibin", "SemiBin2 binning"),
    "Rscript": SoftwareRequirement("Rscript", "r-base", "MAGScoT refinement"),
    "Rscript_das_tool": SoftwareRequirement(
        "Rscript", "r-base", "DAS Tool refinement"
    ),
    "prodigal": SoftwareRequirement("prodigal", "prodigal", "gene prediction"),
    "parallel": SoftwareRequirement("parallel", "parallel", "parallel single-threaded jobs"),
    "hmmsearch": SoftwareRequirement("hmmsearch", "hmmer", "marker HMM search"),
    "DAS_Tool": SoftwareRequirement("DAS_Tool", "das_tool", "DAS Tool refinement"),
    "diamond": SoftwareRequirement("diamond", "diamond", "DAS Tool search engine"),
    "pullseq": SoftwareRequirement("pullseq", "pullseq", "DAS Tool bin extraction"),
    "ruby": SoftwareRequirement("ruby", "ruby", "DAS Tool marker parsing"),
    "checkm": SoftwareRequirement("checkm", "checkm-genome", "CheckM quality estimation"),
    "gunc": SoftwareRequirement("gunc", "gunc", "GUNC contamination checking"),
    "gtdbtk": SoftwareRequirement("gtdbtk", "gtdbtk", "GTDB-Tk taxonomy or RNA domain detection"),
    "tRNAscan-SE": SoftwareRequirement("tRNAscan-SE", "trnascan-se", "tRNA quality control"),
    "barrnap": SoftwareRequirement("barrnap", "barrnap", "rRNA quality control"),
    "galah": SoftwareRequirement("galah", "galah", "MAG dereplication"),
    "dRep": SoftwareRequirement("dRep", "drep", "MAG dereplication"),
    "coverm": SoftwareRequirement("coverm", "coverm", "MAG abundance profiling"),
    "git": SoftwareRequirement("git", "git", "MAGScoT source installation"),
}

R_PACKAGE_PROFILES: dict[str, tuple[RPackageRequirement, ...]] = {
    "MAGScoT": (
        RPackageRequirement("optparse"),
        RPackageRequirement("dplyr"),
        RPackageRequirement("readr"),
        RPackageRequirement("funr"),
        RPackageRequirement("digest"),
    ),
    "DAS Tool": (
        RPackageRequirement("data.table"),
        RPackageRequirement("magrittr"),
        RPackageRequirement("docopt"),
    ),
}
MAGSCOT_R_NAMES = tuple(
    requirement.name for requirement in R_PACKAGE_PROFILES["MAGScoT"]
)
DAS_TOOL_R_NAMES = tuple(
    requirement.name for requirement in R_PACKAGE_PROFILES["DAS Tool"]
)
PIP_INSTALL_URLS = {
    "metadecoder": (
        "https://github.com/liu-congcong/MetaDecoder/releases/download/"
        "v1.2.2/metadecoder-1.2.2-py3-none-any.whl"
    ),
}


def unique_requirements(names: Iterable[str]) -> list[SoftwareRequirement]:
    seen: set[str] = set()
    requirements: list[SoftwareRequirement] = []
    for name in names:
        requirement = SOFTWARE[name]
        if requirement.executable not in seen:
            requirements.append(requirement)
            seen.add(requirement.executable)
    return requirements


def bin_requirements(args: object) -> list[SoftwareRequirement]:
    names = ["bash", str(getattr(args, "align_tool")), "samtools"]
    if getattr(args, "align_tool") == "bowtie2":
        names.append("bowtie2-build")
    for binner in getattr(args, "tools"):
        if binner == "metabat2":
            names.extend(("jgi_summarize_bam_contig_depths", "metabat2"))
        elif binner == "vamb":
            names.append("vamb")
            if getattr(args, "type") == "short":
                names.append("strobealign")
        elif binner == "metadecoder":
            names.append("metadecoder")
        elif binner == "comebin":
            # COMEBin is checked separately because its Bioconda package pins
            # Python 3.7 and must not be installed into MetaBAW's Python 3.11
            # environment.
            continue
        elif binner == "semibin2":
            names.append("SemiBin2")
        elif binner == "lorbin":
            # LorBin 0.1.0 uses a pinned Python 3.10 and PyTorch 1.11 stack.
            continue
    refiner = getattr(args, "refinement")
    if refiner == "magscot":
        names.extend(("Rscript", "prodigal", "parallel", "hmmsearch", "git"))
    elif refiner == "das_tool":
        names.extend(
            (
                "DAS_Tool",
                "Rscript_das_tool",
                "diamond",
                "prodigal",
                "pullseq",
                "ruby",
            )
        )
    else:
        # MetaWRAP is checked separately because its Bioconda package pins
        # Python 2.7.
        pass
    if getattr(args, "quality_control") == "checkm":
        names.append("checkm")
    if getattr(args, "gunc"):
        names.append("gunc")
    if getattr(args, "trna") or getattr(args, "rrna"):
        names.append("gtdbtk")
    if getattr(args, "trna"):
        names.append("tRNAscan-SE")
    if getattr(args, "rrna"):
        names.append("barrnap")
    names.append("galah" if getattr(args, "dereplication_tool") == "galah" else "dRep")
    return unique_requirements(names)


def annotation_requirements(_args: object | None = None) -> list[SoftwareRequirement]:
    return unique_requirements(("bash", "gtdbtk", "coverm", "minimap2"))


def all_requirements() -> list[SoftwareRequirement]:
    return unique_requirements(SOFTWARE)


def missing_software(
    requirements: Sequence[SoftwareRequirement],
) -> list[SoftwareRequirement]:
    return [
        requirement
        for requirement in requirements
        if shutil.which(requirement.executable) is None
    ]


def software_runtime_issues(
    requirements: Sequence[SoftwareRequirement],
) -> list[SoftwareRuntimeIssue]:
    executables = {requirement.executable for requirement in requirements}
    if "barrnap" not in executables:
        return []
    barrnap = shutil.which("barrnap")
    if barrnap is None:
        return []
    try:
        process = subprocess.run(
            [barrnap, "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [
            SoftwareRuntimeIssue(
                "barrnap",
                f"runtime probe could not start: {exc}",
                ("barrnap", "perl-path-tiny"),
            )
        ]
    if process.returncode == 0:
        return []
    detail = " | ".join(
        line.strip()
        for line in (process.stderr or process.stdout).splitlines()
        if line.strip()
    )
    if len(detail) > 500:
        detail = detail[:497] + "..."
    return [
        SoftwareRuntimeIssue(
            "barrnap",
            (
                f"`{barrnap} --version` exited with code "
                f"{process.returncode}: {detail or 'no diagnostic text'}"
            ),
            ("barrnap", "perl-path-tiny"),
        )
    ]


def conda_frontend() -> str | None:
    for executable in ("mamba", "micromamba", "conda"):
        if shutil.which(executable):
            return executable
    return None


def conda_environment_selector(environment: str) -> tuple[str, str]:
    value = environment.strip()
    if not value:
        raise ValueError("Isolated environment cannot be empty")
    is_prefix = (
        Path(value).is_absolute()
        or value.startswith((".", "~"))
        or "/" in value
        or "\\" in value
    )
    if is_prefix:
        return "--prefix", str(Path(value).expanduser().resolve())
    return "--name", value


def isolated_environment_config_path() -> Path:
    explicit = os.environ.get("METABAW_CONFIG_FILE")
    if explicit:
        return Path(explicit).expanduser().resolve()
    config_home = os.environ.get("XDG_CONFIG_HOME")
    root = Path(config_home).expanduser() if config_home else Path.home() / ".config"
    return root / "metabaw" / "config.json"


def _runtime_config() -> dict[str, object]:
    path = isolated_environment_config_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def saved_isolated_environment(key: str) -> str | None:
    if key not in ISOLATED_TOOLS:
        raise ValueError(f"Unknown isolated tool: {key}")
    environments = _runtime_config().get("isolated_environments")
    if not isinstance(environments, dict):
        return None
    value = environments.get(key)
    return str(value).strip() if value else None


def configured_isolated_environment(key: str) -> str:
    if key not in ISOLATED_TOOLS:
        raise ValueError(f"Unknown isolated tool: {key}")
    environment_name = f"METABAW_{key.upper()}_ENV"
    return (
        os.environ.get(environment_name)
        or saved_isolated_environment(key)
        or ISOLATED_TOOLS[key].default_environment
    )


def save_isolated_environment(key: str, environment: str) -> str:
    if key not in ISOLATED_TOOLS:
        raise ValueError(f"Unknown isolated tool: {key}")
    selector, normalized = conda_environment_selector(environment)
    saved_value = normalized if selector == "--prefix" else environment.strip()
    path = isolated_environment_config_path()
    config = _runtime_config()
    environments = config.get("isolated_environments")
    if not isinstance(environments, dict):
        environments = {}
        config["isolated_environments"] = environments
    environments[key] = saved_value
    config["version"] = 1
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return saved_value


def configured_magscot_directory() -> Path:
    configured = (
        os.environ.get("MAGSCOT_DIR")
        or _runtime_config().get("magscot_directory")
        or "~/.cache/metabaw/MAGScoT"
    )
    return Path(str(configured)).expanduser().resolve()


def save_magscot_directory(directory: Path) -> Path:
    normalized = directory.expanduser().resolve()
    path = isolated_environment_config_path()
    config = _runtime_config()
    config["magscot_directory"] = str(normalized)
    config["version"] = 1
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return normalized


def isolated_run_prefix(
    environment: str,
    frontend: str | None = None,
) -> tuple[str, ...]:
    selected_frontend = frontend or conda_frontend() or "conda"
    return (
        selected_frontend,
        "run",
        *conda_environment_selector(environment),
    )


def host_cuda_status() -> HostCudaStatus:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return HostCudaStatus(
            None,
            (),
            None,
            "nvidia-smi is not available",
        )
    query = subprocess.run(
        [
            executable,
            "--query-gpu=index,name,memory.total,driver_version,memory.free",
            "--format=csv,noheader,nounits",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if query.returncode != 0:
        detail = (query.stderr or query.stdout).strip()
        return HostCudaStatus(
            executable,
            (),
            None,
            detail or f"nvidia-smi exited with code {query.returncode}",
        )
    devices: list[CudaDevice] = []
    try:
        for row in csv.reader(query.stdout.splitlines()):
            values = [value.strip() for value in row]
            if len(values) < 4:
                continue
            devices.append(
                CudaDevice(
                    index=values[0],
                    name=values[1],
                    memory_total_mib=max(0, int(float(values[2]))),
                    driver_version=values[3],
                    memory_free_mib=(
                        max(0, int(float(values[4])))
                        if len(values) >= 5 and values[4]
                        else None
                    ),
                )
            )
    except ValueError as exc:
        return HostCudaStatus(
            executable,
            (),
            None,
            f"cannot parse nvidia-smi GPU information: {exc}",
        )
    banner = subprocess.run(
        [executable],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    advertised = None
    if banner.returncode == 0:
        match = re.search(
            r"CUDA(?: UMD)? Version:\s*([0-9.]+)",
            banner.stdout,
        )
        if match:
            advertised = match.group(1)
    if not devices:
        return HostCudaStatus(
            executable,
            (),
            advertised,
            "nvidia-smi reported no NVIDIA GPUs",
        )
    return HostCudaStatus(executable, tuple(devices), advertised)


def cuda_runtime_status(
    label: str,
    python_command: Sequence[str],
) -> CudaRuntimeStatus:
    marker = "METABAW_CUDA_STATUS="
    probe = (
        "from __future__ import print_function\n"
        "import json, os, sys\n"
        "result = {"
        "'python': '%s.%s' % (sys.version_info[0], sys.version_info[1]), "
        "'torch': None, 'torch_cuda': None, "
        "'visible': os.environ.get('CUDA_VISIBLE_DEVICES'), "
        "'available': False, 'count': 0, 'devices': [], "
        "'allocation': False, 'error': None}\n"
        "try:\n"
        "    import torch\n"
        "    result['torch'] = str(torch.__version__)\n"
        "    result['torch_cuda'] = str(torch.version.cuda) if torch.version.cuda else None\n"
        "    result['available'] = bool(torch.cuda.is_available())\n"
        "    if result['available']:\n"
        "        result['count'] = int(torch.cuda.device_count())\n"
        "        result['devices'] = [str(torch.cuda.get_device_name(i)) "
        "for i in range(result['count'])]\n"
        "        test = torch.zeros(1, device='cuda')\n"
        "        torch.cuda.synchronize()\n"
        "        del test\n"
        "        result['allocation'] = True\n"
        "    else:\n"
        "        result['error'] = 'torch.cuda.is_available() returned False'\n"
        "except Exception as exc:\n"
        "    result['error'] = '%s: %s' % (type(exc).__name__, exc)\n"
        f"print({marker!r} + json.dumps(result, sort_keys=True))\n"
    )
    try:
        process = subprocess.run(
            [*python_command, "-c", probe],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except OSError as exc:
        return CudaRuntimeStatus(
            label,
            None,
            None,
            None,
            os.environ.get("CUDA_VISIBLE_DEVICES"),
            0,
            (),
            False,
            f"cannot start CUDA probe: {exc}",
        )
    payload: dict[str, object] | None = None
    for line in reversed(process.stdout.splitlines()):
        if not line.startswith(marker):
            continue
        try:
            candidate = json.loads(line[len(marker) :])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            payload = candidate
            break
    if payload is None:
        detail = (process.stderr or process.stdout).strip()
        if detail:
            detail = detail.splitlines()[-1]
        return CudaRuntimeStatus(
            label,
            None,
            None,
            None,
            os.environ.get("CUDA_VISIBLE_DEVICES"),
            0,
            (),
            False,
            detail or f"CUDA probe exited with code {process.returncode}",
        )
    error_value = payload.get("error")
    error = str(error_value) if error_value else None
    if process.returncode != 0 and error is None:
        error = f"CUDA probe exited with code {process.returncode}"
    devices_value = payload.get("devices")
    devices = (
        tuple(str(value) for value in devices_value)
        if isinstance(devices_value, list)
        else ()
    )
    try:
        device_count = int(payload.get("count", 0))
    except (TypeError, ValueError):
        device_count = 0
    return CudaRuntimeStatus(
        label=label,
        python_version=str(payload.get("python") or "") or None,
        torch_version=str(payload.get("torch") or "") or None,
        torch_cuda_version=str(payload.get("torch_cuda") or "") or None,
        cuda_visible_devices=(
            str(payload["visible"]) if payload.get("visible") is not None else None
        ),
        device_count=device_count,
        devices=devices,
        allocation_test=bool(payload.get("allocation")),
        error=error,
    )


def print_host_cuda_status(status: HostCudaStatus) -> None:
    if not status.available:
        print(
            f"[CUDA MISSING] NVIDIA driver: {status.error or 'unavailable'}",
            flush=True,
        )
        return
    drivers = ", ".join(sorted({device.driver_version for device in status.devices}))
    cuda = status.advertised_cuda_version or "unknown"
    devices = "; ".join(
        (
            f"{device.index}:{device.name} "
            f"({device.memory_total_mib} MiB total, "
            f"{device.memory_free_mib} MiB free)"
            if device.memory_free_mib is not None
            else f"{device.index}:{device.name} ({device.memory_total_mib} MiB)"
        )
        for device in status.devices
    )
    print(
        f"[CUDA OK] NVIDIA driver {drivers}; advertised CUDA {cuda}; GPUs: {devices}",
        flush=True,
    )


def print_cuda_runtime_status(status: CudaRuntimeStatus) -> None:
    state = "CUDA OK" if status.available else "CUDA MISSING"
    visible = (
        status.cuda_visible_devices
        if status.cuda_visible_devices is not None
        else "not set"
    )
    detail = (
        f"Python {status.python_version or 'unknown'}; "
        f"PyTorch {status.torch_version or 'not installed'}; "
        f"PyTorch CUDA {status.torch_cuda_version or 'none'}; "
        f"CUDA_VISIBLE_DEVICES={visible}"
    )
    if status.available:
        detail += f"; devices={', '.join(status.devices)}; allocation test passed"
    else:
        detail += f"; {status.error or 'CUDA allocation test failed'}"
    print(f"[{state}] {status.label}: {detail}", flush=True)


def isolated_environment_status(
    spec: IsolatedToolSpec,
    environment: str,
    frontend: str | None = None,
) -> IsolatedEnvironmentStatus:
    selected_frontend = frontend or conda_frontend()
    if selected_frontend is None:
        return IsolatedEnvironmentStatus(
            spec,
            environment,
            None,
            None,
            None,
            "mamba, micromamba, or conda is not available",
        )
    probe = (
        "import json, sys\n"
        "try:\n"
        "    from shutil import which\n"
        "except ImportError:\n"
        "    from distutils.spawn import find_executable as which\n"
        "print(json.dumps({"
        "'python': '%s.%s' % (sys.version_info[0], sys.version_info[1]), "
        f"'executable': which({spec.executable!r})"
        "}))\n"
    )
    command = [
        *isolated_run_prefix(environment, selected_frontend),
        "python",
        "-c",
        probe,
    ]
    process = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        detail = (process.stderr or process.stdout).strip()
        if detail:
            detail = detail.splitlines()[-1]
        return IsolatedEnvironmentStatus(
            spec,
            environment,
            selected_frontend,
            None,
            None,
            detail or f"environment probe exited with code {process.returncode}",
        )
    payload: dict[str, object] | None = None
    for line in reversed(process.stdout.splitlines()):
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and "python" in candidate:
            payload = candidate
            break
    if payload is None:
        return IsolatedEnvironmentStatus(
            spec,
            environment,
            selected_frontend,
            None,
            None,
            "environment probe returned no readable status",
        )
    python_version = str(payload["python"])
    executable_value = payload.get("executable")
    executable = str(executable_value) if executable_value else None
    error = None
    if python_version != spec.python_version:
        error = (
            f"Python {python_version} is incompatible; "
            f"{spec.display_name} requires Python {spec.python_version}"
        )
    elif executable is None:
        error = f"{spec.executable} is not installed"
    return IsolatedEnvironmentStatus(
        spec,
        environment,
        selected_frontend,
        python_version,
        executable,
        error,
    )


def install_isolated_environment(
    spec: IsolatedToolSpec,
    environment: str,
    frontend: str | None = None,
) -> IsolatedEnvironmentStatus:
    selected_frontend = frontend or conda_frontend()
    if selected_frontend is None:
        raise RuntimeError(
            f"Cannot install {spec.display_name} because mamba, micromamba, "
            "or conda is not available"
        )
    status = isolated_environment_status(spec, environment, selected_frontend)
    if (
        status.python_version is not None
        and status.python_version != spec.python_version
    ):
        raise RuntimeError(
            f"{spec.display_name} environment {environment!r} uses Python "
            f"{status.python_version}. Select a different environment with "
            f"{spec.option}; MetaBAW will not change Python in an existing environment."
        )
    action = "install" if status.python_version == spec.python_version else "create"
    channel_arguments = [
        argument
        for channel in spec.channels
        for argument in ("--channel", channel)
    ]
    command = [
        selected_frontend,
        action,
        "--yes",
        # Legacy pinned stacks (pytorch 1.11, Python 2.7) only resolve when
        # builds from multiple channels can be mixed; a user-level strict
        # channel_priority would make them unsolvable.
        "--channel-priority",
        "flexible",
        *conda_environment_selector(environment),
        *channel_arguments,
        f"python={spec.python_version}",
        *(spec.conda_packages or (spec.package,)),
    ]
    print(
        f"Installing {spec.display_name} in isolated Python {spec.python_version} environment: "
        f"{environment}",
        flush=True,
    )
    process = subprocess.run(command, check=False)
    if process.returncode != 0:
        raise RuntimeError(
            f"{spec.display_name} environment installation failed with exit code "
            f"{process.returncode}: {' '.join(command)}"
        )
    if spec.pip_source:
        pip_command = [
            *isolated_run_prefix(environment, selected_frontend),
            "python",
            "-m",
            "pip",
            "install",
            "--no-deps",
            spec.pip_source,
        ]
        print(
            f"Installing {spec.display_name} from its pinned official source",
            flush=True,
        )
        pip_process = subprocess.run(pip_command, check=False)
        if pip_process.returncode != 0:
            raise RuntimeError(
                f"{spec.display_name} source installation failed with exit code "
                f"{pip_process.returncode}: {' '.join(pip_command)}"
            )
    installed = isolated_environment_status(spec, environment, selected_frontend)
    if not installed.available:
        raise RuntimeError(
            f"{spec.display_name} environment {environment!r} is not usable after installation: "
            f"{installed.error or 'unknown error'}"
        )
    return installed


def install_isolated_cuda_runtime(
    key: str,
    environment: str,
    frontend: str | None = None,
) -> None:
    if key not in ISOLATED_CUDA_PACKAGES:
        raise RuntimeError(f"No isolated CUDA repair profile is available for {key}")
    selected_frontend = frontend or conda_frontend()
    if selected_frontend is None:
        raise RuntimeError(
            "Cannot install CUDA-enabled PyTorch because mamba, micromamba, "
            "or conda is not available"
        )
    channels, packages = ISOLATED_CUDA_PACKAGES[key]
    channel_arguments = [
        argument
        for channel in channels
        for argument in ("--channel", channel)
    ]
    command = [
        selected_frontend,
        "install",
        "--yes",
        "--channel-priority",
        "flexible",
        *conda_environment_selector(environment),
        *channel_arguments,
        *packages,
    ]
    print(
        f"Installing the CUDA-enabled PyTorch profile for "
        f"{ISOLATED_TOOLS[key].display_name}: {environment}",
        flush=True,
    )
    process = subprocess.run(command, check=False)
    if process.returncode != 0:
        raise RuntimeError(
            f"CUDA-enabled PyTorch installation failed with exit code "
            f"{process.returncode}: {' '.join(command)}"
        )


def install_main_cuda_runtime(frontend: str | None = None) -> None:
    selected_frontend = frontend or conda_frontend()
    if selected_frontend is None:
        raise RuntimeError(
            "Cannot install CUDA-enabled PyTorch because mamba, micromamba, "
            "or conda is not available"
        )
    prefix = Path(sys.prefix).resolve()
    if not (prefix / "conda-meta").is_dir():
        raise RuntimeError(
            f"The MetaBAW Python environment is not a Conda environment: {prefix}. "
            "Install a CUDA-enabled PyTorch build in this environment manually."
        )
    command = [
        selected_frontend,
        "install",
        "--yes",
        "--channel-priority",
        "flexible",
        "--prefix",
        str(prefix),
        "--channel",
        "pytorch",
        "--channel",
        "nvidia",
        "--channel",
        "conda-forge",
        "pytorch",
        "pytorch-cuda=11.8",
    ]
    print(
        f"Installing CUDA-enabled PyTorch in the MetaBAW environment: {prefix}",
        flush=True,
    )
    process = subprocess.run(command, check=False)
    if process.returncode != 0:
        raise RuntimeError(
            f"CUDA-enabled PyTorch installation failed with exit code "
            f"{process.returncode}: {' '.join(command)}"
        )


def print_isolated_status(status: IsolatedEnvironmentStatus) -> None:
    state = "OK" if status.available else "MISSING"
    python_version = status.python_version or "unknown"
    detail = status.executable or status.error or f"{status.spec.executable} not found"
    print(
        f"[{state:<7}] {status.spec.display_name} environment {status.environment}; "
        f"Python {python_version}; {detail}",
        flush=True,
    )


def comebin_run_prefix(
    environment: str,
    frontend: str | None = None,
) -> tuple[str, ...]:
    return isolated_run_prefix(environment, frontend)


def comebin_environment_status(
    environment: str,
    frontend: str | None = None,
) -> IsolatedEnvironmentStatus:
    return isolated_environment_status(ISOLATED_TOOLS["comebin"], environment, frontend)


def install_comebin_environment(
    environment: str,
    frontend: str | None = None,
) -> IsolatedEnvironmentStatus:
    return install_isolated_environment(ISOLATED_TOOLS["comebin"], environment, frontend)


def print_comebin_status(status: IsolatedEnvironmentStatus) -> None:
    print_isolated_status(status)


def install_software(
    requirements: Sequence[SoftwareRequirement],
    include_magscot_r_packages: bool = False,
) -> list[SoftwareRequirement]:
    missing = missing_software(requirements)
    profiles = required_r_package_profiles(requirements)
    if include_magscot_r_packages and "MAGScoT" not in profiles:
        profiles.append("MAGScoT")
    missing_r = {
        profile: missing_r_packages(profile)
        for profile in profiles
    }
    runtime_issues = software_runtime_issues(requirements)
    if (
        not missing
        and not any(missing_r.values())
        and not runtime_issues
    ):
        return []
    pip_requirements = [
        requirement
        for requirement in missing
        if requirement.executable in PIP_INSTALL_URLS
    ]
    conda_requirements = [
        requirement
        for requirement in missing
        if requirement.executable not in PIP_INSTALL_URLS
    ]
    packages = list(
        dict.fromkeys(requirement.package for requirement in conda_requirements)
    )
    for issue in runtime_issues:
        packages.extend(
            package
            for package in issue.repair_packages
            if package not in packages
        )
    if packages:
        frontend = conda_frontend()
        if frontend is None:
            raise RuntimeError(
                "Cannot install missing software because mamba, micromamba, "
                "or conda is not available"
            )
        command = [
            frontend,
            "install",
            "--yes",
            "--channel",
            "conda-forge",
            "--channel",
            "bioconda",
            *packages,
        ]
        print("Installing missing software:", " ".join(packages), flush=True)
        process = subprocess.run(command, check=False)
        if process.returncode != 0:
            raise RuntimeError(
                f"Software installation failed with exit code {process.returncode}: "
                f"{' '.join(command)}"
            )
    for requirement in pip_requirements:
        url = PIP_INSTALL_URLS[requirement.executable]
        command = [sys.executable, "-m", "pip", "install", "--upgrade", url]
        print(f"Installing {requirement.executable} from its upstream wheel", flush=True)
        process = subprocess.run(command, check=False)
        if process.returncode != 0:
            raise RuntimeError(
                f"{requirement.executable} installation failed with exit code "
                f"{process.returncode}: {' '.join(command)}"
            )
    r_requirements = [
        requirement
        for profile, names in missing_r.items()
        for requirement in R_PACKAGE_PROFILES[profile]
        if requirement.name in names
    ]
    if r_requirements:
        install_r_packages(r_requirements)
    return missing_software(requirements)


def required_r_package_profiles(
    requirements: Sequence[SoftwareRequirement],
) -> list[str]:
    profiles: list[str] = []
    if any("MAGScoT" in requirement.purpose for requirement in requirements):
        profiles.append("MAGScoT")
    if any(
        requirement.executable == "DAS_Tool"
        or "DAS Tool" in requirement.purpose
        for requirement in requirements
    ):
        profiles.append("DAS Tool")
    return profiles


def missing_r_packages(profile: str) -> list[str]:
    if profile not in R_PACKAGE_PROFILES:
        raise ValueError(f"Unknown R package profile: {profile}")
    rscript = shutil.which("Rscript")
    names = tuple(requirement.name for requirement in R_PACKAGE_PROFILES[profile])
    if rscript is None:
        return list(names)
    missing: list[str] = []
    for package in names:
        process = subprocess.run(
            [
                rscript,
                "-e",
                (
                    f"quit(status=ifelse(requireNamespace('{package}', quietly=TRUE),"
                    "0,1))"
                ),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if process.returncode != 0:
            missing.append(package)
    return missing


def _r_character_vector(values: Sequence[str]) -> str:
    quoted = ", ".join(
        '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
        for value in values
    )
    return f"c({quoted})"


def install_r_packages(
    requirements: Sequence[RPackageRequirement],
) -> None:
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise RuntimeError(
            "Rscript is unavailable after installing external software; "
            "cannot install required R packages"
        )
    unique = {
        (requirement.name, requirement.source.lower()): requirement
        for requirement in requirements
    }
    unsupported = sorted(
        source
        for _name, source in unique
        if source not in {"cran", "bioconductor"}
    )
    if unsupported:
        raise ValueError(
            "Unsupported R package source(s): " + ", ".join(unsupported)
        )
    cran = sorted(name for name, source in unique if source == "cran")
    bioconductor = sorted(
        name for name, source in unique if source == "bioconductor"
    )
    commands: list[tuple[str, list[str]]] = []
    if cran:
        expression = (
            "options(repos=c(CRAN='https://cloud.r-project.org')); "
            f"install.packages({_r_character_vector(cran)}, dependencies=TRUE)"
        )
        commands.append(("CRAN", [rscript, "-e", expression]))
    if bioconductor:
        expression = (
            "options(repos=c(CRAN='https://cloud.r-project.org')); "
            "if (!requireNamespace('BiocManager', quietly=TRUE)) "
            "install.packages('BiocManager', dependencies=TRUE); "
            "if (!requireNamespace('BiocManager', quietly=TRUE)) "
            "quit(status=1); "
            f"BiocManager::install({_r_character_vector(bioconductor)}, "
            "ask=FALSE, update=FALSE)"
        )
        commands.append(("Bioconductor", [rscript, "-e", expression]))
    for source, command in commands:
        print(
            f"Installing missing {source} R packages with Rscript",
            flush=True,
        )
        process = subprocess.run(command, check=False)
        if process.returncode != 0:
            raise RuntimeError(
                f"{source} R package installation failed with exit code "
                f"{process.returncode}"
            )


def missing_magscot_r_packages() -> list[str]:
    return missing_r_packages("MAGScoT")


def missing_dastool_r_packages() -> list[str]:
    return missing_r_packages("DAS Tool")


def magscot_files(directory: Path) -> tuple[Path, Path, Path]:
    return (
        directory / "MAGScoT.R",
        directory / "hmm" / "gtdbtk_rel207_Pfam-A.hmm",
        directory / "hmm" / "gtdbtk_rel207_tigrfam.hmm",
    )


def install_magscot(directory: Path) -> None:
    missing = [path for path in magscot_files(directory) if not path.is_file()]
    if not missing:
        return
    if directory.exists() and any(directory.iterdir()):
        raise RuntimeError(
            f"MAGScoT directory exists but is incomplete: {directory}; missing={missing}"
        )
    directory.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/ikmb/MAGScoT.git",
            str(directory),
        ],
        check=False,
    )
    if process.returncode != 0 or any(not path.is_file() for path in magscot_files(directory)):
        raise RuntimeError(f"MAGScoT installation failed: {directory}")


def _run_database_command(command: list[str]) -> None:
    print("Installing database:", " ".join(command), flush=True)
    process = subprocess.run(command, check=False)
    if process.returncode != 0:
        raise RuntimeError(
            f"Database installation failed with exit code {process.returncode}: {' '.join(command)}"
        )


def install_checkm2_database(
    directory: Path,
    run_prefix: Sequence[str] = (),
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    _run_database_command(
        [
            *run_prefix,
            "checkm2",
            "database",
            "--download",
            "--path",
            str(directory),
        ]
    )
    matches = sorted(directory.rglob("*.dmnd"))
    if not matches:
        raise RuntimeError(f"CheckM2 database download created no .dmnd file under {directory}")
    return matches[0]


def install_gunc_database(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    _run_database_command(["gunc", "download_db", str(directory)])
    matches = sorted(directory.rglob("*.dmnd"))
    if not matches:
        raise RuntimeError(f"GUNC database download created no .dmnd file under {directory}")
    return matches[0]


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    downloaded = 0
    with urllib.request.urlopen(url) as response, partial.open("wb") as output:
        while True:
            block = response.read(8 * 1024 * 1024)
            if not block:
                break
            output.write(block)
            downloaded += len(block)
            if downloaded % (1024 * 1024 * 1024) < len(block):
                print(f"Downloaded {downloaded / (1024 ** 3):.1f} GiB", flush=True)
    partial.replace(destination)


def _safe_extract_tar(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle:
            if member.issym() or member.islnk():
                raise RuntimeError(f"Refusing link in GTDB-Tk archive: {member.name}")
            target = (root / member.name).resolve()
            try:
                target.relative_to(root)
            except ValueError as exc:
                raise RuntimeError(
                    f"Refusing unsafe GTDB-Tk archive member: {member.name}"
                ) from exc
            handle.extract(member, root)


def install_gtdbtk_database(directory: Path) -> Path:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        archive = directory / "gtdbtk_data.tar.gz"
        url = (
            "https://data.gtdb.ecogenomic.org/releases/latest/auxillary_files/"
            "gtdbtk_package/full_package/gtdbtk_data.tar.gz"
        )
        _download_file(url, archive)
        extracted = directory / "data"
        _safe_extract_tar(archive, extracted)
        candidates = [
            path
            for path in (extracted, *extracted.iterdir())
            if path.is_dir() and (path / "metadata").is_dir()
        ]
    except (OSError, tarfile.TarError) as exc:
        raise RuntimeError(
            f"GTDB-Tk database installation failed under {directory}: {exc}"
        ) from exc
    if not candidates:
        raise RuntimeError(f"Cannot locate extracted GTDB-Tk data under {extracted}")
    return candidates[0]


def _database_config_path() -> Path:
    return Path.home() / ".config" / "metabaw" / "databases.json"


def _database_config() -> dict[str, str]:
    path = _database_config_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {str(key): str(value) for key, value in data.items()}


def save_database_path(name: str, path: Path) -> None:
    config_path = _database_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config = _database_config()
    config[name] = str(path.resolve())
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def configured_database_path(
    explicit: str | None,
    environment_name: str,
    config_name: str | None = None,
    artifact_suffix: str | None = None,
) -> Path | None:
    def resolve_candidate(value: str) -> Path:
        path = Path(value).expanduser().resolve()
        if artifact_suffix and path.is_dir():
            matches = sorted(path.rglob(f"*{artifact_suffix}"))
            if matches:
                return matches[0]
        return path

    def is_valid_candidate(path: Path) -> bool:
        if artifact_suffix:
            return path.is_file() and path.name.endswith(artifact_suffix)
        return path.exists()

    if explicit:
        return resolve_candidate(explicit)

    environment_value = os.environ.get(environment_name)
    saved_value = _database_config().get(config_name) if config_name else None
    environment_path = (
        resolve_candidate(environment_value) if environment_value else None
    )
    saved_path = resolve_candidate(saved_value) if saved_value else None

    if environment_path is not None and is_valid_candidate(environment_path):
        return environment_path
    if saved_path is not None and is_valid_candidate(saved_path):
        if environment_path is not None:
            print(
                f"[WARNING] Ignoring invalid {environment_name} path "
                f"{environment_path}; using saved database path {saved_path}",
                file=sys.stderr,
                flush=True,
            )
        return saved_path
    return environment_path or saved_path


def print_software_status(requirements: Sequence[SoftwareRequirement]) -> None:
    for requirement in requirements:
        location = shutil.which(requirement.executable)
        state = "OK" if location else "MISSING"
        if location:
            detail = location
        elif requirement.executable in PIP_INSTALL_URLS:
            detail = "upstream Python wheel"
        else:
            detail = f"Conda package: {requirement.package}"
        print(
            f"[{state}] {requirement.executable:<34} {requirement.purpose}; {detail}",
            flush=True,
        )


def confirm_install(prompt: str) -> bool:
    if not sys.stdin.isatty():
        return False
    response = input(f"{prompt} [y/N]: ").strip().lower()
    return response in {"y", "yes"}
