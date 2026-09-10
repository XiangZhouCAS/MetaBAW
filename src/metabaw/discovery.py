from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


READ_SUFFIXES = (".fastq.gz", ".fq.gz", ".fastq", ".fq")
FASTA_SUFFIXES = (".fa.gz", ".fna.gz", ".fasta.gz", ".fa", ".fna", ".fasta")
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
    assembly_samples: tuple[ReadSample, ...] = ()
    plan_id: str | None = None
    # Public MAG naming is deliberately separated from ``name``.  The latter
    # remains stable for task IDs, work directories, and resume state, while
    # explicit reads-to-contigs mappings can expose that provenance in files.
    public_name: str | None = None
    explicit_mapping: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", normalize_sample_name(self.name))


@dataclass(frozen=True)
class CoassemblyPlan:
    """One Bin Chicken-selected coassembly and its recovery samples."""

    identifier: str
    assembly_sample_names: tuple[str, ...]
    recovery_sample_names: tuple[str, ...]


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

    When SEPARATOR occurs, the part before its first occurrence is always the
    sample name. A trailing 1/2 (or R1/R2, also after processing labels such
    as "clean_R1") in the remaining text marks the read mate. Standard
    _R1/_R2, _1/_2, and .1/.2 pair patterns are used only when SEPARATOR is
    absent. This makes the default separator, '.', consistently reduce names
    such as "SRR1.clean.rehost.1" to sample "SRR1".
    """
    prefix, found, rest = stem.partition(separator)
    if not found or not prefix:
        return _sample_and_mate(stem)
    mate_match = re.search(r"(?:^|[._-])[rR]?([12])$", rest)
    if mate_match:
        return prefix, int(mate_match.group(1))
    return prefix, None


def _samples_from_read_files(
    files: list[Path],
    read_type: str,
    separator: str | None,
    suffix: str | None,
) -> list[ReadSample]:
    """Build samples from explicit read files using the shared naming rules.

    ``suffix`` is supplied for legacy path/directory discovery.  A value of
    ``None`` means that each explicitly listed read has one of the recognized
    FASTQ suffixes and that suffix should be detected independently.
    """
    if read_type == "long":
        samples = []
        for item in files:
            stem = (
                _trim_suffix(item.name, suffix)
                if suffix is not None
                else _strip_known_read_suffix(item.name)
            )
            # A trailing _1/_2 is a valid part of a single-end long-read
            # sample name (for example HMI_1).  Do not apply paired-end mate
            # detection when the configured separator is absent.  When the
            # separator is present, retain the documented prefix behaviour.
            prefix, found, _rest = (
                stem.partition(separator) if separator else (stem, "", "")
            )
            name = prefix if found and prefix else stem
            samples.append(ReadSample(name, item))
        return _validate_unique_sample_names(samples)

    grouped: dict[str, dict[int | None, Path]] = {}
    for item in files:
        stem = (
            _trim_suffix(item.name, suffix)
            if suffix is not None
            else _strip_known_read_suffix(item.name)
        )
        if separator:
            sample, mate = _sample_and_mate_with_separator(stem, separator)
        else:
            sample, mate = _sample_and_mate(stem)
        sample = normalize_sample_name(sample)
        slots = grouped.setdefault(sample, {})
        if mate in slots:
            raise ValueError(
                f"Ambiguous read files for sample {sample!r}: "
                f"{slots[mate]} and {item}"
            )
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


def _paths_from_file(
    path: str | Path,
    *,
    label: str,
    accepted_suffixes: tuple[str, ...],
) -> list[Path]:
    """Read one input path per non-comment line from a UTF-8 manifest."""
    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"{label} file not found: {source}")
    if not source.is_file():
        raise ValueError(f"{label} must be a regular file: {source}")

    files: list[Path] = []
    first_line_by_path: dict[Path, int] = {}
    suffix_text = ", ".join(accepted_suffixes)
    for line_number, raw in enumerate(
        source.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        item = _resolve_from(value, source.parent)
        if not item.exists():
            raise FileNotFoundError(
                f"{source}:{line_number}: listed path not found: {item}"
            )
        if not item.is_file():
            raise ValueError(
                f"{source}:{line_number}: listed path is not a regular file: {item}"
            )
        if not any(item.name.lower().endswith(suffix) for suffix in accepted_suffixes):
            raise ValueError(
                f"{source}:{line_number}: unsupported file suffix for {item}; "
                f"expected one of: {suffix_text}"
            )
        previous_line = first_line_by_path.get(item)
        if previous_line is not None:
            raise ValueError(
                f"{source}:{line_number}: duplicate listed path {item}; "
                f"first listed on line {previous_line}"
            )
        first_line_by_path[item] = line_number
        files.append(item)
    if not files:
        raise ValueError(f"{label} contains no data rows: {source}")
    return files


def discover_reads(
    path: str | Path,
    suffix: str,
    read_type: str,
    separator: str | None = ".",
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

    return _samples_from_read_files(files, read_type, separator, suffix)


def discover_reads_from_file(
    path: str | Path,
    read_type: str,
    separator: str | None = ".",
) -> list[ReadSample]:
    """Discover read samples from a one-path-per-line FASTQ manifest."""
    files = _paths_from_file(
        path,
        label="Read input list",
        accepted_suffixes=READ_SUFFIXES,
    )
    return _samples_from_read_files(files, read_type, separator, None)


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


def discover_contigs_from_file(path: str | Path) -> list[Path]:
    """Discover contigs from a one-path-per-line FASTA manifest."""
    return _paths_from_file(
        path,
        label="Contig input list",
        accepted_suffixes=FASTA_SUFFIXES,
    )


def _contig_key(path: Path, suffix: str | None) -> str:
    stem = (
        _trim_suffix(path.name, suffix)
        if suffix is not None
        else _strip_known_fasta_suffix(path.name)
    )
    return _association_key(stem)


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


def attach_contigs(
    samples: list[ReadSample],
    contigs: list[Path],
    suffix: str | None = None,
) -> list[ReadSample]:
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
            sample = _sample_and_mate_with_separator(
                _strip_known_suffix(Path(read_paths[0]).name), "."
            )[0]
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


def build_analyses(
    samples: list[ReadSample],
    mode: str,
    selected: list[ReadSample] | None = None,
    *,
    explicit_group_name: str | None = None,
) -> list[Analysis]:
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
    if shared and mode != "multi":
        detail = "; ".join(
            f"{path}: {','.join(owners)}"
            for path, owners in sorted(shared.items(), key=lambda item: str(item[0]))
        )
        raise ValueError(
            "A contig assembly is assigned to more than one read sample. "
            "Shared co-assemblies must be processed with --multi so all sample "
            f"reads are mapped to one binning analysis: {detail}"
        )
    if mode == "multi":
        targets = selected or samples
        remainder = (
            [sample for sample in samples if sample.name not in {item.name for item in selected}]
            if selected
            else []
        )
        analyses = _multi_analyses(
            targets,
            remainder,
            explicit_group_name=explicit_group_name,
        )
        analyses.extend(_single_analysis(sample) for sample in remainder)
        return analyses
    return [_single_analysis(sample) for sample in samples]


def build_cohort_analyses(
    samples: list[ReadSample],
    mode: str,
    cohort_size: int,
) -> list[Analysis]:
    """Split multi-sample analyses into deterministic cohorts when requested."""
    if mode != "multi" or len(samples) <= cohort_size:
        return build_analyses(samples, mode)
    analyses: list[Analysis] = []
    for start in range(0, len(samples), cohort_size):
        cohort = samples[start : start + cohort_size]
        analyses.extend(
            build_analyses(cohort, "multi" if len(cohort) > 1 else "single")
        )
    return analyses


def build_bin_chicken_analyses(
    samples: list[ReadSample],
    plans: list[CoassemblyPlan],
    assembly_root: Path,
    remaining_mode: str = "single",
) -> list[Analysis]:
    """Create MetaBAW analyses for Bin Chicken-selected coassemblies.

    Bin Chicken's ``samples`` column identifies reads used to create a shared
    assembly, whereas ``recover_samples`` identifies all reads that should be
    mapped back for differential-abundance binning.  Keeping the two sets
    separate is essential: a recovery sample need not contribute reads to the
    assembly itself.
    """
    by_name = {sample.name: sample for sample in samples}
    analyses: list[Analysis] = []
    coassembled_sample_names: set[str] = set()
    seen_identifiers: set[str] = set()
    seen_analysis_names: set[str] = set()
    for plan in plans:
        if plan.identifier in seen_identifiers:
            raise ValueError(
                f"Bin Chicken plan contains duplicate coassembly ID: {plan.identifier}"
            )
        seen_identifiers.add(plan.identifier)
        if plan.identifier in by_name:
            raise ValueError(
                f"Bin Chicken coassembly ID {plan.identifier!r} conflicts with "
                "an input sample name"
            )
        missing_assembly = sorted(set(plan.assembly_sample_names) - set(by_name))
        missing_recovery = sorted(set(plan.recovery_sample_names) - set(by_name))
        if missing_assembly or missing_recovery:
            details: list[str] = []
            if missing_assembly:
                details.append("assembly=" + ",".join(missing_assembly))
            if missing_recovery:
                details.append("recovery=" + ",".join(missing_recovery))
            raise ValueError(
                f"Bin Chicken plan {plan.identifier!r} refers to unknown samples: "
                + "; ".join(details)
            )
        assembly_samples = tuple(by_name[name] for name in plan.assembly_sample_names)
        coassembled_sample_names.update(plan.assembly_sample_names)
        recovery_names = plan.recovery_sample_names or plan.assembly_sample_names
        recovery_samples = tuple(by_name[name] for name in recovery_names)
        if len(assembly_samples) < 2:
            raise ValueError(
                f"Bin Chicken plan {plan.identifier!r} contains fewer than two "
                "assembly samples"
            )
        analysis_name = "coassembly_" + "_".join(
            sample.name for sample in assembly_samples
        )
        if analysis_name in seen_analysis_names:
            raise ValueError(
                "Bin Chicken plan creates a duplicate sample-derived coassembly "
                f"name: {analysis_name}"
            )
        if analysis_name in by_name:
            raise ValueError(
                f"Sample-derived coassembly name {analysis_name!r} conflicts "
                "with an input sample name"
            )
        seen_analysis_names.add(analysis_name)
        analyses.append(
            Analysis(
                analysis_name,
                recovery_samples,
                (assembly_root / analysis_name / "final.contigs.fa",),
                True,
                False,
                assembly_samples,
                plan.identifier,
            )
        )
    # Samples that Bin Chicken did not select for coassembly remain valuable.
    # They retain MetaBAW's standard independent-assembly workflow, allowing
    # the final provenance report to distinguish coassembled and individual MAGs.
    remaining = [
        sample for sample in samples if sample.name not in coassembled_sample_names
    ]
    analyses.extend(build_analyses(remaining, remaining_mode) if remaining else [])
    return analyses


def _multi_analyses(
    targets: list[ReadSample],
    remainder: list[ReadSample],
    *,
    explicit_group_name: str | None = None,
) -> list[Analysis]:
    """Build multi-sample analyses with cross mapping.

    Targets with per-sample assemblies each get their own analysis named
    after the owning sample, and every target's reads are mapped to every
    assembly (cross mapping), so bins keep the owning sample's name while
    binners still see differential coverage from all samples.
    """
    if len(targets) == 1:
        analysis = _single_analysis(targets[0])
        if explicit_group_name is None:
            return [analysis]
        assert targets[0].contigs is not None
        contig_label = public_sample_name(
            _strip_known_fasta_suffix(targets[0].contigs.name)
        )
        return [
            Analysis(
                analysis.name,
                analysis.samples,
                analysis.contigs,
                analysis.combined,
                public_name=f"{targets[0].name}_to_{contig_label}",
                explicit_mapping=True,
            )
        ]
    groups: dict[Path, list[ReadSample]] = {}
    for sample in targets:
        if sample.contigs is None:
            raise ValueError(f"Contigs are not assigned for sample {sample.name!r}")
        groups.setdefault(sample.contigs, []).append(sample)
    mapped = tuple(targets)
    analyses: list[Analysis] = []
    analysis_names: set[str] = set()
    public_names: set[str] = set()
    for contigs, owners in groups.items():
        name = (
            owners[0].name
            if len(owners) == 1
            else public_sample_name(_strip_known_fasta_suffix(contigs.name))
        )
        if name in analysis_names:
            raise ValueError(
                "Multi-sample contig inputs create a duplicate analysis name "
                f"{name!r}; rename the contig files to unique dataset names"
            )
        analysis_names.add(name)
        public_name = None
        if explicit_group_name is not None:
            read_label = owners[0].name if len(owners) == 1 else explicit_group_name
            contig_label = public_sample_name(_strip_known_fasta_suffix(contigs.name))
            public_name = f"{read_label}_to_{contig_label}"
            if public_name in public_names:
                raise ValueError(
                    "Explicit multi-sample mappings create a duplicate public "
                    f"MAG prefix {public_name!r}; use uniquely named contig files"
                )
            public_names.add(public_name)
        analyses.append(
            Analysis(
                name,
                mapped,
                (contigs,),
                False,
                True,
                public_name=public_name,
                explicit_mapping=explicit_group_name is not None,
            )
        )
    return analyses


def _single_analysis(sample: ReadSample) -> Analysis:
    if sample.contigs is None:
        raise ValueError(f"Contigs are not assigned for sample {sample.name!r}")
    return Analysis(sample.name, (sample,), (sample.contigs,), False)


def _resolve_from(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _strip_known_read_suffix(name: str) -> str:
    value = name
    for suffix in READ_SUFFIXES:
        if value.lower().endswith(suffix):
            return value[: -len(suffix)]
    return Path(value).stem


def _strip_known_fasta_suffix(name: str) -> str:
    value = name
    for suffix in FASTA_SUFFIXES:
        if value.lower().endswith(suffix):
            return value[: -len(suffix)]
    return Path(value).stem


def _strip_known_suffix(name: str) -> str:
    """Backward-compatible alias for FASTQ sample-name inference."""
    return _strip_known_read_suffix(name)


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
