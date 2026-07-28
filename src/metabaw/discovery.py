from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


FASTA_SUFFIXES = (".fa", ".fna", ".fasta", ".fa.gz", ".fna.gz", ".fasta.gz")
PAIR_PATTERNS = (
    re.compile(r"^(?P<sample>.+?)(?:_R)(?P<mate>[12])(?P<tail>_001)?$", re.IGNORECASE),
    re.compile(r"^(?P<sample>.+?)(?:_)(?P<mate>[12])$", re.IGNORECASE),
    re.compile(r"^(?P<sample>.+?)(?:\.)(?P<mate>[12])$", re.IGNORECASE),
)
ASSOCIATION_TRAILING_TOKENS = {
    "assembly",
    "assemblies",
    "clean",
    "cleaned",
    "contig",
    "contigs",
    "filtered",
    "final",
    "ok",
    "qc",
    "scaffold",
    "scaffolds",
    "trim",
    "trimmed",
}
SAMPLE_NAME_TRAILING_TOKENS = {"clean", "cleaned"}


def _strip_trailing_tokens(value: str, trailing_tokens: set[str]) -> str:
    """Remove recognized trailing tokens while preserving original separators."""
    normalized = value.strip().strip("._-")
    while normalized:
        match = re.search(r"[._-]+(?P<token>[^._-]+)$", normalized)
        if match is None or match.group("token").lower() not in trailing_tokens:
            break
        normalized = normalized[: match.start()].rstrip("._-")
    return normalized


def normalize_sample_name(value: str) -> str:
    """Remove trailing clean/cleaned processing tokens from a sample label."""
    normalized = _strip_trailing_tokens(value, SAMPLE_NAME_TRAILING_TOKENS)
    if not normalized:
        raise ValueError(f"Sample name becomes empty after normalization: {value!r}")
    return normalized


@dataclass(frozen=True)
class ReadSample:
    name: str
    read1: Path
    read2: Path | None = None
    contigs: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", normalize_sample_name(self.name))

    @property
    def reads(self) -> tuple[Path, ...]:
        return (self.read1,) if self.read2 is None else (self.read1, self.read2)


@dataclass(frozen=True)
class Analysis:
    name: str
    samples: tuple[ReadSample, ...]
    contigs: tuple[Path, ...]
    combined: bool
    cross_mapped: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", normalize_sample_name(self.name))


def _trim_suffix(name: str, suffix: str) -> str:
    normalized = suffix.lstrip(".")
    lower = name.lower()
    marker = "." + normalized.lower()
    if not lower.endswith(marker):
        return name
    return name[: -len(marker)]


def _sample_and_mate(stem: str) -> tuple[str, int | None]:
    for pattern in PAIR_PATTERNS:
        match = pattern.fullmatch(stem)
        if match:
            return match.group("sample"), int(match.group("mate"))
    return stem, None


def _sample_and_mate_with_separator(stem: str, separator: str) -> tuple[str, int | None]:
    """Resolve sample and mate, splitting STEM at the first SEPARATOR.

    Standard _R1/_R2, _1/_2, and .1/.2 pair patterns take precedence. For
    other names, the part before the first SEPARATOR becomes the sample name
    and a trailing 1/2 (or R1/R2, also after processing labels such as
    "clean_R1") marks the read mate; any other trailing content is treated as
    a processing label and the read is accepted as unpaired for that sample.
    """
    sample, mate = _sample_and_mate(stem)
    if mate is not None:
        return sample, mate
    prefix, found, rest = stem.partition(separator)
    if not found or not prefix:
        return stem, None
    mate_match = re.search(r"(?:^|[._-])[rR]?([12])$", rest)
    if mate_match:
        return prefix, int(mate_match.group(1))
    return prefix, None


def discover_reads(
    path: str | Path,
    suffix: str,
    read_type: str,
    separator: str | None = None,
) -> list[ReadSample]:
    source = Path(path).expanduser().resolve()
    if source.is_file():
        files = [source]
    elif source.is_dir():
        marker = "." + suffix.lstrip(".").lower()
        files = sorted(candidate.resolve() for candidate in source.iterdir() if candidate.name.lower().endswith(marker))
    else:
        raise FileNotFoundError(f"Reads path not found: {source}")
    if not files:
        raise ValueError(f"No *.{suffix.lstrip('.')} reads found under {source}")

    if read_type == "long":
        samples = []
        for item in files:
            stem = _trim_suffix(item.name, suffix)
            name = (
                _sample_and_mate_with_separator(stem, separator)[0] if separator else stem
            )
            samples.append(ReadSample(name, item))
        return _validate_unique_sample_names(samples)

    grouped: dict[str, dict[int | None, Path]] = {}
    for item in files:
        stem = _trim_suffix(item.name, suffix)
        if separator:
            sample, mate = _sample_and_mate_with_separator(stem, separator)
        else:
            sample, mate = _sample_and_mate(stem)
        sample = normalize_sample_name(sample)
        slots = grouped.setdefault(sample, {})
        if mate in slots:
            raise ValueError(f"Ambiguous read files for sample {sample!r}: {slots[mate]} and {item}")
        slots[mate] = item

    samples: list[ReadSample] = []
    for name, slots in sorted(grouped.items()):
        if 2 in slots and 1 not in slots:
            raise ValueError(f"Read 2 exists without read 1 for sample {name!r}")
        if None in slots and len(slots) > 1:
            raise ValueError(f"Mixed paired and unpaired naming for sample {name!r}")
        read1 = slots.get(1) or slots.get(None)
        if read1 is None:
            raise ValueError(f"Cannot determine read 1 for sample {name!r}")
        samples.append(ReadSample(name, read1, slots.get(2)))
    return samples


def discover_contigs(path: str | Path, suffix: str) -> list[Path]:
    source = Path(path).expanduser().resolve()
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"Contig path not found: {source}")
    marker = "." + suffix.lstrip(".").lower()
    files = sorted(candidate.resolve() for candidate in source.iterdir() if candidate.name.lower().endswith(marker))
    if not files:
        raise ValueError(f"No *.{suffix.lstrip('.')} contigs found under {source}")
    return files


def _contig_key(path: Path, suffix: str) -> str:
    return _association_key(_trim_suffix(path.name, suffix))


def _association_key(value: str) -> str:
    tokens = [token for token in re.split(r"[._-]+", value) if token]
    while len(tokens) > 1 and tokens[-1].lower() in ASSOCIATION_TRAILING_TOKENS:
        tokens.pop()
    return ".".join(tokens).lower()


def public_sample_name(value: str) -> str:
    """Return a stable public sample label without processing suffixes."""
    return _strip_trailing_tokens(value, ASSOCIATION_TRAILING_TOKENS) or value


def _validate_unique_sample_names(samples: list[ReadSample]) -> list[ReadSample]:
    seen: dict[str, Path] = {}
    for sample in samples:
        previous = seen.get(sample.name)
        if previous is not None:
            raise ValueError(
                f"Read filenames are ambiguous after sample-name normalization: "
                f"{previous} and {sample.read1} both map to {sample.name!r}"
            )
        seen[sample.name] = sample.read1
    return samples


def attach_contigs(samples: list[ReadSample], contigs: list[Path], suffix: str) -> list[ReadSample]:
    if len(contigs) == 1:
        return [ReadSample(item.name, item.read1, item.read2, contigs[0]) for item in samples]
    by_key: dict[str, Path] = {}
    for path in contigs:
        key = _contig_key(path, suffix)
        if key in by_key:
            raise ValueError(f"Contig filenames are ambiguous after normalization: {by_key[key]} and {path}")
        by_key[key] = path
    attached: list[ReadSample] = []
    missing: list[str] = []
    for sample in samples:
        sample_key = _association_key(sample.name)
        candidate = by_key.get(sample_key)
        if candidate is None:
            matching = [
                path
                for key, path in by_key.items()
                if sample_key in key or key in sample_key
            ]
            if len(matching) == 1:
                candidate = matching[0]
        if candidate is None:
            missing.append(sample.name)
        else:
            attached.append(ReadSample(sample.name, sample.read1, sample.read2, candidate))
    if missing:
        raise ValueError(
            "Could not match contigs to samples: "
            + ", ".join(missing)
            + ". Use one shared contig file or --multi-files for explicit mapping."
        )
    return attached


def read_multi_files(path: str | Path) -> list[ReadSample]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Multi-sample file not found: {source}")
    samples: list[ReadSample] = []
    seen: set[str] = set()
    for line_number, raw in enumerate(source.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = [field.strip() for field in line.split("\t")]
        if len(fields) == 2:
            reads_field, contig_field = fields
            read_paths = [item.strip() for item in reads_field.split(",") if item.strip()]
            if not read_paths:
                raise ValueError(f"{source}:{line_number}: missing read path")
            sample = _sample_and_mate(_strip_known_suffix(Path(read_paths[0]).name))[0]
        elif len(fields) == 3:
            sample, reads_field, contig_field = fields
            read_paths = [item.strip() for item in reads_field.split(",") if item.strip()]
        else:
            raise ValueError(
                f"{source}:{line_number}: expected READ1[,READ2]<TAB>CONTIGS or "
                "SAMPLE<TAB>READ1[,READ2]<TAB>CONTIGS"
            )
        if len(read_paths) not in {1, 2}:
            raise ValueError(f"{source}:{line_number}: expected one or two read files")
        sample = normalize_sample_name(sample)
        if not sample or sample in seen:
            raise ValueError(f"{source}:{line_number}: empty or duplicate sample name {sample!r}")
        reads = [_resolve_from(item, source.parent) for item in read_paths]
        contig = _resolve_from(contig_field, source.parent)
        for item in (*reads, contig):
            if not item.is_file():
                raise FileNotFoundError(f"{source}:{line_number}: file not found: {item}")
        samples.append(ReadSample(sample, reads[0], reads[1] if len(reads) == 2 else None, contig))
        seen.add(sample)
    if not samples:
        raise ValueError(f"Multi-sample file contains no data rows: {source}")
    return samples


def build_analyses(samples: list[ReadSample], mode: str, selected: list[ReadSample] | None = None) -> list[Analysis]:
    owners_by_assembly: dict[Path, list[str]] = {}
    for sample in samples:
        if sample.contigs is None:
            raise ValueError(f"Contigs are not assigned for sample {sample.name!r}")
        owners_by_assembly.setdefault(sample.contigs, []).append(sample.name)
    shared = {
        path: owners
        for path, owners in owners_by_assembly.items()
        if len(owners) > 1
    }
    if shared:
        detail = "; ".join(
            f"{path}: {','.join(owners)}"
            for path, owners in sorted(shared.items(), key=lambda item: str(item[0]))
        )
        raise ValueError(
            "The current MetaBAW workflow supports single-sample and "
            "multi-sample binning with one independently assembled contig file "
            f"per sample. Shared co-assembly input is not enabled: {detail}"
        )
    if mode == "multi":
        targets = selected or samples
        remainder = (
            [sample for sample in samples if sample.name not in {item.name for item in selected}]
            if selected
            else []
        )
        analyses = _multi_analyses(targets, remainder)
        analyses.extend(_single_analysis(sample) for sample in remainder)
        return analyses
    return [_single_analysis(sample) for sample in samples]


def _multi_analyses(targets: list[ReadSample], remainder: list[ReadSample]) -> list[Analysis]:
    """Build multi-sample analyses with cross mapping.

    Targets with per-sample assemblies each get their own analysis named
    after the owning sample, and every target's reads are mapped to every
    assembly (cross mapping), so bins keep the owning sample's name while
    binners still see differential coverage from all samples.
    """
    if len(targets) == 1:
        return [_single_analysis(targets[0])]
    groups: dict[Path, list[ReadSample]] = {}
    for sample in targets:
        if sample.contigs is None:
            raise ValueError(f"Contigs are not assigned for sample {sample.name!r}")
        groups.setdefault(sample.contigs, []).append(sample)
    mapped = tuple(targets)
    analyses: list[Analysis] = []
    for contigs, owners in groups.items():
        analyses.append(Analysis(owners[0].name, mapped, (contigs,), False, True))
    return analyses


def _single_analysis(sample: ReadSample) -> Analysis:
    if sample.contigs is None:
        raise ValueError(f"Contigs are not assigned for sample {sample.name!r}")
    return Analysis(sample.name, (sample,), (sample.contigs,), False)


def _resolve_from(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _strip_known_suffix(name: str) -> str:
    value = name
    for suffix in (".fastq.gz", ".fq.gz", ".fastq", ".fq"):
        if value.lower().endswith(suffix):
            return value[: -len(suffix)]
    return Path(value).stem


def discover_mag_files(path: str | Path, suffix: str) -> list[Path]:
    source = Path(path).expanduser().resolve()
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"MAG path not found: {source}")
    marker = "." + suffix.lstrip(".").lower()
    files = sorted(candidate.resolve() for candidate in source.iterdir() if candidate.name.lower().endswith(marker))
    if not files:
        fallback = sorted(
            candidate.resolve()
            for candidate in source.iterdir()
            if any(candidate.name.lower().endswith(item) for item in FASTA_SUFFIXES)
        )
        if fallback:
            return fallback
        raise ValueError(f"No MAG files with suffix .{suffix.lstrip('.')} found under {source}")
    return files
