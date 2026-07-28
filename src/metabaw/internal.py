from __future__ import annotations

import argparse
from collections import Counter, OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import gzip
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
from typing import Iterable, Iterator, TextIO


FASTA_SUFFIXES = (".fa", ".fna", ".fasta", ".fa.gz", ".fna.gz", ".fasta.gz")


def apply_memory_limit_from_environment() -> None:
    """Apply the executor-provided address-space ceiling to internal tasks."""
    raw = os.environ.get("METABAW_MAX_MEMORY_GB", "").strip()
    if not raw or os.name != "posix":
        return
    try:
        import resource

        requested = max(1, int(float(raw) * 1024**3))
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        target = requested
        if soft != resource.RLIM_INFINITY:
            target = min(target, soft)
        if hard != resource.RLIM_INFINITY:
            target = min(target, hard)
        resource.setrlimit(resource.RLIMIT_AS, (target, hard))
    except (ImportError, OSError, ValueError):
        return


def open_text(path: Path, mode: str = "rt") -> TextIO:
    if path.name.endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8")  # type: ignore[return-value]
    return path.open(mode.replace("t", ""), encoding="utf-8")


def iter_fasta(path: Path) -> Iterator[tuple[str, str, str]]:
    header: str | None = None
    chunks: list[str] = []
    with open_text(path) as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.rstrip("\r\n")
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header.split()[0], header, "".join(chunks)
                header = line[1:].strip()
                if not header:
                    raise ValueError(f"Empty FASTA header in {path} at line {line_number}")
                chunks = []
            else:
                if header is None:
                    raise ValueError(f"Sequence found before first header in {path} at line {line_number}")
                chunks.append(line.strip())
    if header is not None:
        yield header.split()[0], header, "".join(chunks)


def write_record(handle: TextIO, identifier: str, sequence: str, width: int = 80) -> None:
    handle.write(f">{identifier}\n")
    for start in range(0, len(sequence), width):
        handle.write(sequence[start : start + width] + "\n")


def fasta_filter(input_path: Path, output_path: Path, minimum_length: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    kept = 0
    with output_path.open("w", encoding="utf-8") as output:
        for identifier, _header, sequence in iter_fasta(input_path):
            if identifier in seen:
                raise ValueError(f"Duplicate contig identifier {identifier!r} in {input_path}")
            seen.add(identifier)
            if len(sequence) >= minimum_length:
                write_record(output, identifier, sequence)
                kept += 1
    if kept == 0:
        raise ValueError(f"No contigs >= {minimum_length} bp in {input_path}")


def concatenate_fastas(
    items: list[str],
    output_path: Path,
    minimum_length: int,
    separator: str = "__",
) -> None:
    """Concatenate assemblies while making contig identifiers globally unique."""
    if not separator:
        raise ValueError("FASTA identifier separator cannot be empty")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    kept = 0
    with output_path.open("w", encoding="utf-8") as output:
        for item in items:
            label, raw_path = item.split("=", 1)
            path = Path(raw_path)
            safe_label = _safe_filename(label)
            if separator in safe_label:
                raise ValueError(
                    f"Sample label {safe_label!r} contains the FASTA separator {separator!r}"
                )
            for identifier, _header, sequence in iter_fasta(path):
                if len(items) > 1 and separator in identifier:
                    raise ValueError(
                        f"Contig identifier {identifier!r} contains the FASTA separator "
                        f"{separator!r}"
                    )
                renamed = (
                    f"{safe_label}{separator}{identifier}"
                    if len(items) > 1
                    else identifier
                )
                if renamed in seen:
                    raise ValueError(f"Duplicate contig identifier after concatenation: {renamed!r}")
                seen.add(renamed)
                if len(sequence) >= minimum_length:
                    write_record(output, renamed, sequence)
                    kept += 1
    if kept == 0:
        raise ValueError(f"No contigs >= {minimum_length} bp in the selected assemblies")


def _fasta_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and any(path.name.lower().endswith(suffix) for suffix in FASTA_SUFFIXES)
    )


def _fasta_stem(path: Path) -> str:
    name = path.name
    if name.endswith(".gz"):
        name = name[:-3]
    for suffix in (".fasta", ".fna", ".fa"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def bins_to_map(
    bins_dir: Path,
    output: Path,
    label: str,
    minimum_bin_bp: int = 0,
) -> None:
    files = _fasta_files(bins_dir)
    if not files:
        raise ValueError(f"No FASTA bins found in {bins_dir}")
    output.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        for fasta in files:
            records = list(iter_fasta(fasta))
            if sum(len(sequence) for _contig, _header, sequence in records) < minimum_bin_bp:
                continue
            bin_id = f"{label}__{_fasta_stem(fasta)}"
            for contig, _header, _sequence in records:
                writer.writerow((bin_id, contig, label))
                rows_written += 1
    if rows_written == 0:
        raise ValueError(
            f"No bins under {bins_dir} passed the minimum size of "
            f"{minimum_bin_bp} bp"
        )


def normalize_map(input_path: Path, output: Path, label: str, order: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with input_path.open("r", encoding="utf-8-sig", newline="") as source, output.open(
        "w", encoding="utf-8", newline=""
    ) as destination:
        writer = csv.writer(destination, delimiter="\t", lineterminator="\n")
        for row_number, row in enumerate(csv.reader(source, delimiter="\t"), start=1):
            if not row or all(not value.strip() for value in row):
                continue
            if len(row) < 2:
                raise ValueError(f"Expected at least two columns in {input_path}, row {row_number}")
            lowered = {row[0].strip().lower(), row[1].strip().lower()}
            if row_number == 1 and lowered & {"clustername", "contigname", "bin", "contig"}:
                continue
            if order == "bin-contig":
                bin_id, contig = row[0].strip(), row[1].strip()
            else:
                contig, bin_id = row[0].strip(), row[1].strip()
            writer.writerow((f"{label}__{bin_id}", contig, label))
            rows_written += 1
    if rows_written == 0:
        raise ValueError(f"No assignments found in {input_path}")


def combine_maps(inputs: list[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as destination:
        for input_path in inputs:
            if not input_path.is_file():
                print(f"Skipping unavailable binner assignment map: {input_path}", flush=True)
                continue
            with input_path.open("r", encoding="utf-8") as source:
                for line in source:
                    if line.strip():
                        destination.write(line if line.endswith("\n") else line + "\n")
                        count += 1
    if count == 0:
        raise ValueError("No bin assignments were available to combine")


def map_for_dastool(input_path: Path, output: Path) -> None:
    """Convert metaBAW's bin-contig-label map to DAS Tool's contig-bin table."""
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with input_path.open("r", encoding="utf-8-sig", newline="") as source, output.open(
        "w", encoding="utf-8", newline=""
    ) as destination:
        writer = csv.writer(destination, delimiter="\t", lineterminator="\n")
        for row in csv.reader(source, delimiter="\t"):
            if len(row) >= 2 and row[0].strip() and row[1].strip():
                writer.writerow((row[1].strip(), row[0].strip()))
                count += 1
    if count == 0:
        raise ValueError(f"No assignments found in {input_path}")


def merge_aemb(items: list[str], output: Path) -> None:
    """Merge strobealign AEMB files into VAMB's contigname abundance matrix."""
    samples: list[str] = []
    values: dict[str, dict[str, str]] = defaultdict(dict)
    contig_order: list[str] = []
    for item in items:
        sample, raw_path = item.split("=", 1)
        if sample in samples:
            raise ValueError(f"Duplicate AEMB sample: {sample}")
        samples.append(sample)
        with Path(raw_path).open("r", encoding="utf-8-sig", newline="") as handle:
            for row_number, row in enumerate(csv.reader(handle, delimiter="\t"), start=1):
                if len(row) < 2:
                    continue
                contig, abundance = row[0].strip(), row[1].strip()
                if row_number == 1 and contig.lower() in {"contigname", "contig", "reference"}:
                    continue
                if contig not in values:
                    contig_order.append(contig)
                values[contig][sample] = abundance
    if not contig_order:
        raise ValueError("No contig abundance rows found in strobealign AEMB files")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("contigname", *samples))
        for contig in contig_order:
            writer.writerow((contig, *(values[contig].get(sample, "0") for sample in samples)))


def parse_magscot_hmm(pfam: Path, tigr: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination, delimiter="\t", lineterminator="\n")
        for path, use_accession in ((pfam, True), (tigr, False)):
            with path.open("r", encoding="utf-8", errors="replace") as source:
                for line in source:
                    if not line.strip() or line.startswith("#"):
                        continue
                    fields = line.split()
                    if len(fields) < 5:
                        continue
                    protein = fields[0]
                    marker = fields[3] if use_accession and fields[3] != "-" else fields[2]
                    writer.writerow((protein, marker, fields[4]))
                    count += 1
    if count == 0:
        raise ValueError("No MAGScoT marker hits were parsed from HMMER output")


def _safe_filename(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "._-" else "_" for char in value)
    return cleaned.strip("._") or "bin"


def materialize_bins(assembly: Path, mapping: Path, output_dir: Path, prefix: str, suffix: str = "fa") -> None:
    contig_to_bin: dict[str, str] = {}
    with mapping.open("r", encoding="utf-8-sig", newline="") as handle:
        for row_number, row in enumerate(csv.reader(handle, delimiter="\t"), start=1):
            if not row or len(row) < 2:
                continue
            bin_id, contig = row[0].strip(), row[1].strip()
            if row_number == 1 and bin_id.lower() in {"bin", "binnew", "clustername"}:
                continue
            previous = contig_to_bin.setdefault(contig, bin_id)
            if previous != bin_id:
                raise ValueError(f"Contig {contig!r} is assigned to multiple refined bins")
    if not contig_to_bin:
        raise ValueError(f"No assignments found in {mapping}")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    handles: OrderedDict[str, TextIO] = OrderedDict()
    filenames = {
        bin_id: f"{_safe_filename(prefix)}_{number}.{suffix}"
        for number, bin_id in enumerate(sorted(set(contig_to_bin.values())), start=1)
    }
    counts: defaultdict[str, int] = defaultdict(int)

    def get_handle(bin_id: str) -> TextIO:
        if bin_id in handles:
            handle = handles.pop(bin_id)
            handles[bin_id] = handle
            return handle
        if len(handles) >= 128:
            _, old_handle = handles.popitem(last=False)
            old_handle.close()
        filename = filenames[bin_id]
        handle = (output_dir / filename).open("a", encoding="utf-8")
        handles[bin_id] = handle
        return handle

    try:
        for contig, _header, sequence in iter_fasta(assembly):
            bin_id = contig_to_bin.get(contig)
            if bin_id is None:
                continue
            write_record(get_handle(bin_id), contig, sequence)
            counts[bin_id] += 1
    finally:
        for handle in handles.values():
            handle.close()
    missing = sorted(set(contig_to_bin.values()) - set(counts))
    if missing:
        raise ValueError(f"Assignments refer to bins with no matching contigs: {missing[:10]}")
    with (output_dir / "manifest.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("bin", "file", "contigs"))
        for bin_id in sorted(counts):
            writer.writerow((bin_id, filenames[bin_id], counts[bin_id]))


def _bin_assignments(mapping: Path) -> dict[str, str]:
    contig_to_bin: dict[str, str] = {}
    with mapping.open("r", encoding="utf-8-sig", newline="") as handle:
        for row_number, row in enumerate(csv.reader(handle, delimiter="\t"), start=1):
            if not row or len(row) < 2:
                continue
            bin_id, contig = row[0].strip(), row[1].strip()
            if row_number == 1 and bin_id.lower() in {"bin", "binnew", "clustername"}:
                continue
            previous = contig_to_bin.setdefault(contig, bin_id)
            if previous != bin_id:
                raise ValueError(f"Contig {contig!r} is assigned to multiple refined bins")
    if not contig_to_bin:
        raise ValueError(f"No assignments found in {mapping}")
    return contig_to_bin


def _write_materialized_bins(
    assembly: Path,
    contig_to_bin: dict[str, str],
    filenames: dict[str, str],
    output_dir: Path,
) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    handles: OrderedDict[str, TextIO] = OrderedDict()

    def get_handle(bin_id: str) -> TextIO:
        if bin_id in handles:
            handle = handles.pop(bin_id)
            handles[bin_id] = handle
            return handle
        if len(handles) >= 128:
            _, old_handle = handles.popitem(last=False)
            old_handle.close()
        handle = (output_dir / filenames[bin_id]).open("a", encoding="utf-8")
        handles[bin_id] = handle
        return handle

    counts: defaultdict[str, int] = defaultdict(int)
    try:
        for contig, _header, sequence in iter_fasta(assembly):
            bin_id = contig_to_bin.get(contig)
            if bin_id is None:
                continue
            write_record(get_handle(bin_id), contig, sequence)
            counts[bin_id] += 1
    finally:
        for handle in handles.values():
            handle.close()
    missing = sorted(set(contig_to_bin.values()) - set(counts))
    if missing:
        raise ValueError(f"Assignments refer to bins with no matching contigs: {missing[:10]}")
    with (output_dir / "manifest.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("bin", "file", "contigs"))
        for bin_id in sorted(counts):
            writer.writerow((bin_id, filenames[bin_id], counts[bin_id]))


def materialize_refined(
    assembly: Path,
    mapping: Path,
    output_dir: Path,
    prefix: str,
    sources: list[Path],
    suffix: str = "fa",
) -> None:
    """Write refined bins while keeping original binner names for unchanged bins.

    A refined bin whose contig set is identical to a bin already published by an
    individual binner keeps that bin's sample_tool_sequence filename. Only bins
    created or modified by the refinement step receive new prefix_sequence names.
    """
    contig_to_bin = _bin_assignments(mapping)
    source_names: dict[frozenset[str], str] = {}
    for directory in sources:
        if not directory.is_dir():
            print(f"Skipping unavailable binner bins directory: {directory}", flush=True)
            continue
        for fasta in _fasta_files(directory):
            contigs = frozenset(contig for contig, _header, _sequence in iter_fasta(fasta))
            if contigs:
                source_names.setdefault(contigs, _fasta_stem(fasta))
    bin_to_contigs: dict[str, set[str]] = defaultdict(set)
    for contig, bin_id in contig_to_bin.items():
        bin_to_contigs[bin_id].add(contig)
    filenames: dict[str, str] = {}
    refined_number = 0
    for bin_id in sorted(bin_to_contigs):
        kept = source_names.get(frozenset(bin_to_contigs[bin_id]))
        if kept is not None:
            filenames[bin_id] = f"{kept}.{suffix}"
        else:
            refined_number += 1
            filenames[bin_id] = f"{_safe_filename(prefix)}_{refined_number}.{suffix}"
    _write_materialized_bins(assembly, contig_to_bin, filenames, output_dir)


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def stage_bams(items: list[str], output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    for item in items:
        sample, raw_path = item.split("=", 1)
        source = Path(raw_path)
        destination = output_dir / f"{_safe_filename(sample)}.bam"
        _link_or_copy(source, destination)
        bai_candidates = [Path(str(source) + ".bai"), source.with_suffix(".bai")]
        bai = next((path for path in bai_candidates if path.exists()), None)
        if bai is not None:
            _link_or_copy(bai, Path(str(destination) + ".bai"))


def collect_fasta(sources: list[Path], output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    seen: set[str] = set()
    collected: list[tuple[str, str]] = []
    for directory in sources:
        for source in _fasta_files(directory):
            if source.name in seen:
                raise ValueError(f"Duplicate candidate bin filename: {source.name}")
            seen.add(source.name)
            _link_or_copy(source, output_dir / source.name)
            collected.append((str(directory), source.name))
    if not seen:
        raise ValueError("No refined FASTA bins were collected")
    with (output_dir / "manifest.tsv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as table:
        writer = csv.writer(table, delimiter="\t", lineterminator="\n")
        writer.writerow(("source_directory", "file"))
        writer.writerows(collected)


def publish_fasta(source_dir: Path, output_dir: Path, prefix: str, suffix: str = "fa") -> None:
    """Publish bins under stable sample_tool_sequence .fa filenames."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    files = _fasta_files(source_dir)
    if not files:
        raise ValueError(f"No FASTA bins found in {source_dir}")
    manifest = output_dir / "manifest.tsv"
    with manifest.open("w", encoding="utf-8", newline="") as table:
        writer = csv.writer(table, delimiter="\t", lineterminator="\n")
        writer.writerow(("source", "published"))
        for number, source in enumerate(files, start=1):
            name = f"{_safe_filename(prefix)}_{number}.{suffix}"
            with (output_dir / name).open("w", encoding="utf-8") as output:
                for identifier, _header, sequence in iter_fasta(source):
                    write_record(output, identifier, sequence)
            writer.writerow((source.name, name))


def normalize_refined_fasta(
    source_dir: Path,
    output_dir: Path,
    prefix: str,
    sources: list[Path],
    suffix: str = "fa",
) -> None:
    """Normalize refined FASTA names while retaining unchanged-bin provenance."""
    refined_files = _fasta_files(source_dir)
    if not refined_files:
        raise ValueError(f"No refined FASTA bins found in {source_dir}")

    source_names: dict[frozenset[str], str] = {}
    for directory in sources:
        if not directory.is_dir():
            print(f"Skipping unavailable binner bins directory: {directory}", flush=True)
            continue
        for fasta in _fasta_files(directory):
            contigs = frozenset(
                identifier
                for identifier, _header, _sequence in iter_fasta(fasta)
            )
            if contigs:
                source_names.setdefault(contigs, _fasta_stem(fasta))

    planned: list[tuple[Path, str, str]] = []
    used_names: set[str] = set()
    refined_number = 0
    for fasta in refined_files:
        contigs = frozenset(
            identifier
            for identifier, _header, _sequence in iter_fasta(fasta)
        )
        if not contigs:
            raise ValueError(f"Refined FASTA contains no sequences: {fasta}")
        kept = source_names.get(contigs)
        if kept is None:
            refined_number += 1
            published = f"{_safe_filename(prefix)}_{refined_number}.{suffix}"
            provenance = "refined"
        else:
            published = f"{kept}.{suffix}"
            provenance = "unchanged"
        if published in used_names:
            raise ValueError(
                f"Duplicate normalized refined MAG filename: {published}"
            )
        used_names.add(published)
        planned.append((fasta, published, provenance))

    staging = output_dir.with_name(f".{output_dir.name}.metabaw-normalizing")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        manifest = staging / "manifest.tsv"
        with manifest.open("w", encoding="utf-8", newline="") as table:
            writer = csv.writer(table, delimiter="\t", lineterminator="\n")
            writer.writerow(("source", "published", "provenance"))
            for source, published, provenance in planned:
                with (staging / published).open("w", encoding="utf-8") as output:
                    for identifier, _header, sequence in iter_fasta(source):
                        write_record(output, identifier, sequence)
                writer.writerow((source.name, published, provenance))
        if output_dir.exists():
            shutil.rmtree(output_dir)
        staging.replace(output_dir)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def tag_contigs(input_dir: Path, output_dir: Path, suffix: str = "fa") -> None:
    """Copy bins and replace each contig ID by BIN_1, BIN_2, ..."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    count = 0
    for source in _fasta_files(input_dir):
        bin_name = _safe_filename(_fasta_stem(source))
        destination = output_dir / f"{bin_name}.{suffix}"
        with destination.open("w", encoding="utf-8") as handle:
            for number, (_identifier, _header, sequence) in enumerate(iter_fasta(source), start=1):
                write_record(handle, f"{bin_name}_{number}", sequence)
                count += 1
    if count == 0:
        raise ValueError(f"No FASTA records found in {input_dir}")


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"true", "t", "1", "yes", "pass"}


def _read_checkm2(path: Path) -> tuple[dict[str, dict[str, str]], str, str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = reader.fieldnames or []
        normalized = {field.lower().replace(" ", "_"): field for field in fields}
        name_col = normalized.get("name") or normalized.get("genome") or normalized.get("bin_id")
        completeness_col = normalized.get("completeness")
        contamination_col = normalized.get("contamination")
        if not name_col or not completeness_col or not contamination_col:
            raise ValueError(f"Unrecognized CheckM2 report columns: {fields}")
        rows = {_canonical_genome(row[name_col]): row for row in reader}
    return rows, name_col, completeness_col, contamination_col


def filter_quality(
    bins_dir: Path,
    checkm2_report: Path,
    output_dir: Path,
    summary: Path,
    min_completeness: float,
    max_contamination: float,
    gunc_dir: Path | None,
    gunc_min_reference_representation: float = 0.30,
    fail_gunc_unscored: bool = False,
    min_quality_score: float | None = None,
    rna_summary: Path | None = None,
) -> None:
    checkm_rows, _name_col, completeness_col, contamination_col = _read_checkm2(checkm2_report)
    gunc: dict[str, bool | None] = {}
    if gunc_dir is not None:
        matches = sorted(gunc_dir.glob("GUNC.*maxCSS_level.tsv"))
        if not matches:
            raise ValueError(f"No GUNC maxCSS summary found in {gunc_dir}")
        with matches[0].open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                genome = _canonical_genome(row.get("genome", ""))
                pass_value = row.get("pass.GUNC", row.get("pass_gunc", ""))
                normalized = str(pass_value).strip().lower()
                if normalized in {"true", "t", "yes", "y", "1"}:
                    call: bool | None = True
                elif normalized in {"false", "f", "no", "n", "0"}:
                    call = False
                else:
                    call = None
                representation = row.get("reference_representation_score", "")
                if representation not in {None, ""}:
                    try:
                        if float(representation) < gunc_min_reference_representation:
                            call = None
                    except ValueError:
                        call = None
                gunc[genome] = call
    rna: dict[str, dict[str, str]] = {}
    if rna_summary is not None:
        with rna_summary.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            rna = {_canonical_genome(row.get("genome", "")): row for row in reader}
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    summary.parent.mkdir(parents=True, exist_ok=True)
    passed = 0
    evaluated = 0
    reason_counts: Counter[str] = Counter()
    with summary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            ("bin", "completeness", "contamination", "pass_gunc", "selected", "reason", "quality_score", "pass_rna")
        )
        for fasta in _fasta_files(bins_dir):
            evaluated += 1
            name = _fasta_stem(fasta)
            row = checkm_rows.get(name)
            if row is None:
                reason_counts["missing_checkm2"] += 1
                writer.writerow(
                    (name, "", "", "", "false", "missing_checkm2")
                )
                continue
            completeness = float(row[completeness_col])
            contamination = float(row[contamination_col])
            quality_score = completeness - 5 * contamination
            pass_gunc: bool | None = gunc.get(name) if gunc else True
            rna_row = rna.get(name)
            pass_rna = True if not rna else bool(rna_row and _truthy(rna_row.get("pass", "")))
            reasons: list[str] = []
            if completeness < min_completeness:
                reasons.append("low_completeness")
            if contamination > max_contamination:
                reasons.append("high_contamination")
            if min_quality_score is not None and quality_score < min_quality_score:
                reasons.append("low_quality_score")
            if pass_gunc is False:
                reasons.append("gunc_failed")
            elif pass_gunc is None and fail_gunc_unscored:
                reasons.append("gunc_unscored")
            if not pass_rna:
                reasons.append(
                    "rna_tool_error"
                    if rna_row and rna_row.get("error", "").strip()
                    else "rna_failed"
                )
            selected = not reasons
            reason_counts.update(reasons)
            if selected:
                _link_or_copy(fasta, output_dir / f"{name}.fa")
                passed += 1
            writer.writerow(
                (
                    name,
                    completeness,
                    contamination,
                    (
                        "unscored"
                        if pass_gunc is None
                        else str(pass_gunc).lower()
                        if gunc
                        else "not_run"
                    ),
                    str(selected).lower(),
                    ";".join(reasons) or "pass",
                    f"{quality_score:.6f}",
                    str(pass_rna).lower() if rna else "not_run",
                )
            )
    if passed == 0:
        breakdown = ", ".join(
            f"{reason}={count}"
            for reason, count in sorted(reason_counts.items())
        )
        raise ValueError(
            "No bins passed the configured quality thresholds; "
            f"evaluated={evaluated}; reasons: {breakdown or 'unknown'}; "
            f"details={summary}"
        )


def _marker_domains(directory: Path) -> dict[str, str]:
    scores: dict[str, dict[str, float]] = defaultdict(dict)
    for domain, total, pattern in (
        ("bacteria", 120, "*bac120.markers_summary.tsv"),
        ("archaea", 53, "*ar53.markers_summary.tsv"),
    ):
        for path in directory.rglob(pattern):
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                for row in reader:
                    lowered = {key.lower().replace(" ", "_"): key for key in row}
                    name_col = lowered.get("genome") or lowered.get("name")
                    if not name_col:
                        continue
                    unique_col = lowered.get("number_unique_genes") or lowered.get("unique")
                    multiple_col = lowered.get("number_multiple_genes") or lowered.get("multiple")
                    try:
                        present = float(row.get(unique_col or "", 0) or 0) + float(
                            row.get(multiple_col or "", 0) or 0
                        )
                    except ValueError:
                        present = 0
                    scores[_canonical_genome(row[name_col])][domain] = present / total
    return {
        genome: max(values, key=values.get)
        for genome, values in scores.items()
        if values
    }


def _run_logged(command: list[str], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    stderr_path = output.with_name(output.name + ".stderr.log")
    with (
        output.open("w", encoding="utf-8", errors="replace") as stdout,
        stderr_path.open("w", encoding="utf-8", errors="replace") as stderr,
    ):
        process = subprocess.run(
            command,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    if process.returncode != 0:
        detail = " ".join(
            stderr_path.read_text(encoding="utf-8", errors="replace").split()
        )
        if not detail:
            detail = " ".join(
                output.read_text(encoding="utf-8", errors="replace").split()
            )
        if len(detail) > 500:
            detail = detail[:497] + "..."
        message = (
            f"Command failed ({process.returncode}): {' '.join(command)}; "
            f"stderr: {detail or 'no error text'}; see {stderr_path}"
        )
        raise RuntimeError(message)
    try:
        if stderr_path.stat().st_size == 0:
            stderr_path.unlink()
    except OSError:
        pass


def rna_qc(
    bins_dir: Path,
    marker_dir: Path,
    output_dir: Path,
    summary: Path,
    trnascan: str,
    barrnap: str,
    threads: int,
    run_trna: bool,
    trna_pass: int | None,
    run_rrna: bool,
    rrna_pass: bool,
    completion_marker: Path | None = None,
) -> None:
    run_trna = run_trna or trna_pass is not None
    run_rrna = run_rrna or rrna_pass
    if bins_dir.resolve().is_relative_to(output_dir.resolve()):
        raise ValueError(
            "RNA QC output directory would delete the input bins directory: "
            f"output={output_dir}, bins={bins_dir}"
        )
    domains = _marker_domains(marker_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    if completion_marker is not None:
        completion_marker.unlink(missing_ok=True)
    summary.parent.mkdir(parents=True, exist_ok=True)
    fastas = _fasta_files(bins_dir)
    if not fastas:
        raise ValueError(f"No candidate FASTA bins found in {bins_dir}")
    available_threads = max(1, threads)
    workers = min(len(fastas), available_threads)
    threads_per_worker = max(1, available_threads // workers)
    print(
        f"RNA QC scheduling: genomes={len(fastas)}, workers={workers}, "
        f"threads_per_worker={threads_per_worker}, "
        f"maximum_threads={workers * threads_per_worker}",
        flush=True,
    )

    def analyze(
        fasta: Path,
    ) -> tuple[str, str, set[str], set[str], str, str]:
        genome = _fasta_stem(fasta)
        domain = domains.get(genome, "bacteria")
        trna_types: set[str] = set()
        rrna_types: set[str] = set()
        trna_error = ""
        rrna_error = ""
        if run_trna:
            trna_table = output_dir / "tRNA" / f"{genome}.tsv"
            mode = "-A" if domain == "archaea" else "-B"
            try:
                _run_logged(
                    [
                        trnascan,
                        mode,
                        "--thread",
                        str(threads_per_worker),
                        "-o",
                        str(trna_table),
                        str(fasta),
                    ],
                    output_dir / "tRNA" / f"{genome}.log",
                )
                with trna_table.open(
                    "r",
                    encoding="utf-8",
                    errors="replace",
                ) as source:
                    for line in source:
                        fields = line.rstrip().split("\t")
                        if (
                            len(fields) >= 5
                            and fields[4] not in {"Type", "Undet", "Sup"}
                        ):
                            trna_types.add(fields[4])
            except (OSError, RuntimeError) as exc:
                trna_error = str(exc)
        if run_rrna:
            gff = output_dir / "rRNA" / f"{genome}.gff"
            kingdom = "arc" if domain == "archaea" else "bac"
            try:
                _run_logged(
                    [
                        barrnap,
                        "--kingdom",
                        kingdom,
                        "--threads",
                        str(threads_per_worker),
                        str(fasta),
                    ],
                    gff,
                )
                with gff.open(
                    "r",
                    encoding="utf-8",
                    errors="replace",
                ) as source:
                    for line in source:
                        if line.startswith("#"):
                            continue
                        for kind in ("5S", "16S", "23S"):
                            if (
                                f"Name={kind}_rRNA" in line
                                or f"{kind} ribosomal RNA" in line
                            ):
                                rrna_types.add(kind)
            except (OSError, RuntimeError) as exc:
                rrna_error = str(exc)
        return (
            genome,
            domain,
            trna_types,
            rrna_types,
            trna_error,
            rrna_error,
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(analyze, fastas))

    with summary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            (
                "genome",
                "domain",
                "trna_types",
                "rrna_5S",
                "rrna_16S",
                "rrna_23S",
                "pass",
                "error",
            )
        )
        for (
            genome,
            domain,
            trna_types,
            rrna_types,
            trna_error,
            rrna_error,
        ) in results:
            errors = []
            if trna_error:
                errors.append(f"tRNA: {trna_error}")
            if rrna_error:
                errors.append(f"rRNA: {rrna_error}")
            trna_filter_passed = (
                trna_pass is None
                or (not trna_error and len(trna_types) >= trna_pass)
            )
            rrna_filter_passed = (
                not rrna_pass
                or (
                    not rrna_error
                    and {"5S", "16S", "23S"} <= rrna_types
                )
            )
            passed = trna_filter_passed and rrna_filter_passed
            if errors:
                print(
                    f"RNA QC warning for {genome}: {'; '.join(errors)}",
                    file=sys.stderr,
                    flush=True,
                )
            writer.writerow(
                (
                    genome,
                    domain,
                    len(trna_types) if run_trna else "not_run",
                    str("5S" in rrna_types).lower() if run_rrna else "not_run",
                    str("16S" in rrna_types).lower() if run_rrna else "not_run",
                    str("23S" in rrna_types).lower() if run_rrna else "not_run",
                    str(passed).lower(),
                    "; ".join(errors),
                )
            )
    requested_invocations = len(results) * (int(run_trna) + int(run_rrna))
    failed_invocations = sum(
        int(bool(trna_error)) + int(bool(rrna_error))
        for (
            _genome,
            _domain,
            _trna_types,
            _rrna_types,
            trna_error,
            rrna_error,
        ) in results
    )
    if requested_invocations and failed_invocations == requested_invocations:
        (
            first_genome,
            _first_domain,
            _first_trna_types,
            _first_rrna_types,
            first_trna_error,
            first_rrna_error,
        ) = results[0]
        first_errors = [
            error
            for error in (
                f"tRNA: {first_trna_error}" if first_trna_error else "",
                f"rRNA: {first_rrna_error}" if first_rrna_error else "",
            )
            if error
        ]
        raise RuntimeError(
            f"Every requested RNA tool invocation failed across "
            f"{len(results)} candidate MAGs; "
            f"first failure ({first_genome}): {'; '.join(first_errors)}; "
            f"details={summary}"
        )
    if completion_marker is not None:
        completion_marker.parent.mkdir(parents=True, exist_ok=True)
        completion_marker.write_text("complete\n", encoding="utf-8")


COVERM_METRICS = {
    "relative_abundance": ("relative abundance (%)", "relative_abundance", "rel_abd"),
    "rpkm": ("rpkm",),
    "tpm": ("tpm",),
    "mean": ("mean",),
    "count": ("read count", "count"),
    "trimmed_mean": ("trimmed mean",),
    "covered_fraction": ("covered fraction",),
    "covered_bases": ("covered bases",),
    "reads_per_base": ("reads per base",),
    "variance": ("variance",),
    "length": ("length",),
}


def _coverm_metric(field: str) -> str | None:
    normalized = field.strip().lower()
    for key, aliases in sorted(COVERM_METRICS.items(), key=lambda item: -max(map(len, item[1]))):
        if any(normalized == alias or normalized.endswith(" " + alias) for alias in aliases):
            return key
    return None


def merge_coverm_taxonomy(
    items: list[str],
    taxonomy_dir: Path,
    output_dir: Path,
    output_suffix: str,
) -> None:
    """Write a full CoverM matrix plus one taxonomy-aware matrix per metric."""
    taxonomy = _read_gtdb_taxonomy(taxonomy_dir)
    samples: list[str] = []
    data: dict[str, dict[str, dict[str, str]]] = defaultdict(lambda: defaultdict(dict))
    metrics: set[str] = set()
    genomes: set[str] = set()
    for item in items:
        sample, raw_path = item.split("=", 1)
        if sample in samples:
            raise ValueError(f"Duplicate CoverM sample: {sample}")
        samples.append(sample)
        with Path(raw_path).open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            fields = reader.fieldnames or []
            if not fields:
                raise ValueError(f"Empty CoverM table: {raw_path}")
            genome_col = fields[0]
            field_metrics = {field: _coverm_metric(field) for field in fields[1:]}
            for row in reader:
                genome = _canonical_genome(row.get(genome_col, ""))
                if not genome:
                    continue
                genomes.add(genome)
                for field, metric in field_metrics.items():
                    if metric:
                        data[metric][genome][sample] = row.get(field, "0") or "0"
                        metrics.add(metric)
    if not metrics:
        raise ValueError("No recognized CoverM metrics were found")
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = output_suffix if output_suffix.startswith(".") else "." + output_suffix
    rank_names = ("domain", "phylum", "class", "order", "family", "genus", "species")

    def taxonomy_values(genome: str) -> list[str]:
        classification = taxonomy.get(genome, "")
        return [_rank_taxon(genome, classification, rank) for rank in rank_names]

    filenames = {
        "relative_abundance": "coverm_rel_abd",
        "rpkm": "coverm_rpkm_abd",
        "tpm": "coverm_tpm_abd",
        "mean": "coverm_mean_abd",
        "count": "coverm_counts",
    }
    manifest: dict[str, str] = {}
    for metric in sorted(metrics):
        path = output_dir / f"{filenames.get(metric, 'coverm_' + metric)}{suffix}"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(("Genome", *rank_names, *samples))
            for genome in sorted(genomes):
                writer.writerow(
                    (genome, *taxonomy_values(genome), *(data[metric][genome].get(sample, "0") for sample in samples))
                )
        manifest[metric] = path.name

    combined = output_dir / f"coverm_all_metrics{suffix}"
    with combined.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        header = ["Genome", *rank_names]
        for sample in samples:
            header.extend(f"{sample}|{metric}" for metric in sorted(metrics))
        writer.writerow(header)
        for genome in sorted(genomes):
            row: list[str] = [genome, *taxonomy_values(genome)]
            for sample in samples:
                row.extend(data[metric][genome].get(sample, "0") for metric in sorted(metrics))
            writer.writerow(row)
    manifest["all_metrics"] = combined.name
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


RANK_PREFIXES = {
    "domain": "d__",
    "phylum": "p__",
    "class": "c__",
    "order": "o__",
    "family": "f__",
    "genus": "g__",
    "species": "s__",
}


def _canonical_genome(value: str) -> str:
    name = Path(value.strip()).name
    if name.endswith(".gz"):
        name = name[:-3]
    for suffix in (".fasta", ".fna", ".fa"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


def _read_gtdb_taxonomy(directory: Path) -> dict[str, str]:
    taxonomy: dict[str, str] = {}
    summaries = sorted(directory.rglob("gtdbtk.*.summary.tsv"))
    if not summaries:
        raise ValueError(f"No GTDB-Tk summary files found in {directory}")
    for summary in summaries:
        with summary.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                genome = row.get("user_genome", "").strip()
                classification = row.get("classification", "").strip()
                if genome and classification:
                    taxonomy[_canonical_genome(genome)] = classification
    return taxonomy


def _rank_taxon(genome: str, classification: str, rank: str) -> str:
    if rank in {"mag", "strain"}:
        return genome
    prefix = RANK_PREFIXES[rank]
    value = next((part.strip() for part in classification.split(";") if part.strip().startswith(prefix)), prefix)
    if value == prefix:
        return f"{prefix}unclassified::{genome}"
    return value


def classify_niche_literature(
    abundance: Path,
    taxonomy_dir: Path,
    output: Path,
    aggregated_output: Path,
    assignments_output: Path,
    provenance_output: Path,
    sample_datasets: list[str],
    rank: str,
    detection_percent: float,
    min_total_reads: int,
    min_prevalence: float,
    core_prevalence: float,
    min_core_datasets: int,
) -> None:
    """Classify niches by combining Chen 2021 SI and Tovar-Herrera 2025 core prevalence.

    Chen's corrected specialization index is CV(density) - sqrt(K/N), where K is
    the number of sampled habitat profiles and N is the total taxon read count.
    Tovar-Herrera's core evidence is prevalence >=80% with support across datasets.
    The three-class synthesis is explicit: concordant core+low-SI is generalist,
    concordant non-core+high-SI is specialist, and discordant evidence is intermediate.
    """
    if rank not in {*RANK_PREFIXES, "strain"}:
        raise ValueError(f"Unsupported niche rank: {rank}")
    sample_to_dataset: dict[str, str] = {}
    for item in sample_datasets:
        sample, dataset = item.split("=", 1)
        if sample in sample_to_dataset:
            raise ValueError(f"Duplicate sample metadata: {sample}")
        sample_to_dataset[sample] = dataset
    if len(sample_to_dataset) < 2:
        raise ValueError("Literature-based niche classification requires at least two samples")
    taxonomy = _read_gtdb_taxonomy(taxonomy_dir) if rank not in {"mag", "strain"} else {}

    with abundance.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = reader.fieldnames or []
        if not fields or fields[0] != "Genome":
            raise ValueError("The merged CoverM matrix must start with a Genome column")
        relative_columns: dict[str, str] = {}
        count_columns: dict[str, str] = {}
        for sample in sample_to_dataset:
            sample_fields = [field for field in fields if field.startswith(f"{sample}|")]
            relative = next((field for field in sample_fields if "relative abundance" in field.lower()), None)
            count = next((field for field in sample_fields if "count" in field.lower()), None)
            if relative is None or count is None:
                raise ValueError(
                    f"CoverM matrix lacks relative_abundance or count for {sample}; available={sample_fields}"
                )
            relative_columns[sample] = relative
            count_columns[sample] = count
        genome_rows = list(reader)

    taxon_relative: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    taxon_counts: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    taxon_members: dict[str, set[str]] = defaultdict(set)
    genome_to_taxon: dict[str, str] = {}
    for row in genome_rows:
        genome = _canonical_genome(row["Genome"])
        classification = taxonomy.get(genome, "")
        taxon = _rank_taxon(genome, classification, rank)
        genome_to_taxon[genome] = taxon
        taxon_members[taxon].add(genome)
        for sample in sample_to_dataset:
            taxon_relative[taxon][sample] += float(row.get(relative_columns[sample]) or 0)
            taxon_counts[taxon][sample] += float(row.get(count_columns[sample]) or 0)

    datasets = sorted(set(sample_to_dataset.values()))
    dataset_samples = {
        dataset: sorted(sample for sample, value in sample_to_dataset.items() if value == dataset)
        for dataset in datasets
    }
    required_core_datasets = min(min_core_datasets, len(datasets))
    rows: list[dict[str, object]] = []
    k_habitat_classes = len(sample_to_dataset)
    for taxon in sorted(taxon_members):
        densities = [taxon_relative[taxon][sample] for sample in sample_to_dataset]
        counts = [taxon_counts[taxon][sample] for sample in sample_to_dataset]
        overall_prevalence = sum(value >= detection_percent for value in densities) / len(densities)
        total_reads = sum(counts)
        mean_density = statistics.fmean(densities)
        raw_si = statistics.stdev(densities) / mean_density if mean_density > 0 else math.inf
        si_bias = math.sqrt(k_habitat_classes / total_reads) if total_reads > 0 else math.inf
        corrected_si = raw_si - si_bias if math.isfinite(raw_si) and math.isfinite(si_bias) else math.inf
        prevalences = {
            dataset: sum(taxon_relative[taxon][sample] >= detection_percent for sample in samples) / len(samples)
            for dataset, samples in dataset_samples.items()
        }
        core_dataset_count = sum(value >= core_prevalence for value in prevalences.values())
        eligible = total_reads >= min_total_reads and overall_prevalence >= min_prevalence
        rows.append(
            {
                "taxon": taxon,
                "members": len(taxon_members[taxon]),
                "overall_prevalence": overall_prevalence,
                "total_reads": total_reads,
                "raw_si": raw_si,
                "si_bias": si_bias,
                "corrected_si": corrected_si,
                "core_dataset_count": core_dataset_count,
                "eligible": eligible,
                "prevalences": prevalences,
            }
        )
    eligible_si = [float(row["corrected_si"]) for row in rows if row["eligible"] and math.isfinite(float(row["corrected_si"]))]
    if not eligible_si:
        raise ValueError("No taxa passed the minimum read and prevalence filters for niche classification")
    community_mean_si = statistics.fmean(eligible_si)
    taxon_niche: dict[str, str] = {}
    for row in rows:
        if not row["eligible"]:
            niche = "unclassified_low_support"
            reason = "below_min_total_reads_or_prevalence"
        else:
            core_supported = int(row["core_dataset_count"]) >= required_core_datasets
            low_si = float(row["corrected_si"]) <= community_mean_si
            if core_supported and low_si:
                niche = "generalist"
                reason = "core_prevalence_and_below_mean_corrected_si"
            elif not core_supported and not low_si:
                niche = "specialist"
                reason = "non_core_prevalence_and_above_mean_corrected_si"
            else:
                niche = "intermediate"
                reason = "discordant_prevalence_and_specialization_evidence"
        row["niche"] = niche
        row["reason"] = reason
        taxon_niche[str(row["taxon"])] = niche

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            (
                "taxon",
                "rank",
                "member_mags",
                "overall_prevalence",
                "total_reads",
                "raw_si_cv",
                "expected_si_bias_sqrt_K_over_N",
                "corrected_si",
                "community_mean_corrected_si",
                "core_datasets",
                "required_core_datasets",
                *(f"prevalence::{dataset}" for dataset in datasets),
                "eligible",
                "niche",
                "reason",
            )
        )
        for row in rows:
            prevalences = row["prevalences"]
            writer.writerow(
                (
                    row["taxon"],
                    rank,
                    row["members"],
                    f"{float(row['overall_prevalence']):.6f}",
                    f"{float(row['total_reads']):.3f}",
                    f"{float(row['raw_si']):.8f}",
                    f"{float(row['si_bias']):.8f}",
                    f"{float(row['corrected_si']):.8f}",
                    f"{community_mean_si:.8f}",
                    row["core_dataset_count"],
                    required_core_datasets,
                    *(f"{float(prevalences[dataset]):.6f}" for dataset in datasets),  # type: ignore[index]
                    str(row["eligible"]).lower(),
                    row["niche"],
                    row["reason"],
                )
            )

    aggregated_output.parent.mkdir(parents=True, exist_ok=True)
    with aggregated_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        header = ["Taxon"]
        for sample in sample_to_dataset:
            header.extend((f"{sample}|Relative Abundance (%)", f"{sample}|Count"))
        writer.writerow(header)
        for taxon in sorted(taxon_members):
            values: list[object] = [taxon]
            for sample in sample_to_dataset:
                values.extend((taxon_relative[taxon][sample], taxon_counts[taxon][sample]))
            writer.writerow(values)

    assignments_output.parent.mkdir(parents=True, exist_ok=True)
    with assignments_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("Genome", "Taxon", "Rank", "Niche"))
        for genome, taxon in sorted(genome_to_taxon.items()):
            writer.writerow((genome, taxon, rank, taxon_niche[taxon]))

    provenance_output.parent.mkdir(parents=True, exist_ok=True)
    provenance_output.write_text(
        json.dumps(
            {
                "method": "chen_tovar_synthesis_v1",
                "rank": rank,
                "strain_definition": "dereplicated representative MAG proxy" if rank in {"strain", "mag"} else None,
                "references": [
                    {
                        "citation": "Chen et al. 2021, The ISME Journal",
                        "doi": "10.1038/s41396-021-00988-w",
                        "implemented_evidence": "corrected_SI = CV(taxon density across samples) - sqrt(K/N)",
                    },
                    {
                        "citation": "Tovar-Herrera et al. 2025, Nature Ecology & Evolution",
                        "doi": "10.1038/s41559-025-02904-3",
                        "implemented_evidence": "core prevalence threshold with cross-dataset support",
                    },
                ],
                "parameters": {
                    "detection_percent": detection_percent,
                    "min_total_reads": min_total_reads,
                    "min_prevalence": min_prevalence,
                    "core_prevalence": core_prevalence,
                    "configured_min_core_datasets": min_core_datasets,
                    "effective_min_core_datasets": required_core_datasets,
                    "datasets": datasets,
                    "K_sample_profiles": k_habitat_classes,
                    "community_mean_corrected_si": community_mean_si,
                },
                "classification_rules": {
                    "generalist": "core-supported AND corrected_SI <= community mean",
                    "specialist": "not core-supported AND corrected_SI > community mean",
                    "intermediate": "eligible but prevalence and SI evidence are discordant",
                    "unclassified_low_support": "below minimum total reads or prevalence",
                },
                "interpretation_note": (
                    "The three-class rule is an explicit synthesis implemented by metaBAW; "
                    "neither source paper originally defined this exact three-class decision table."
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m metabaw.internal")
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("fasta-filter")
    command.add_argument("--input", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--min-length", type=int, required=True)
    command = sub.add_parser("concat-fasta")
    command.add_argument("--input", action="append", required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--min-length", type=int, required=True)
    command.add_argument("--separator", default="__")
    command = sub.add_parser("bins-to-map")
    command.add_argument("--bins-dir", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--label", required=True)
    command.add_argument("--min-bin-bp", type=int, default=0)
    command = sub.add_parser("normalize-map")
    command.add_argument("--input", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--label", required=True)
    command.add_argument("--order", choices=("bin-contig", "contig-bin"), default="bin-contig")
    command = sub.add_parser("combine-maps")
    command.add_argument("--input", type=Path, action="append", required=True)
    command.add_argument("--output", type=Path, required=True)
    command = sub.add_parser("map-for-dastool")
    command.add_argument("--input", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command = sub.add_parser("merge-aemb")
    command.add_argument("--input", action="append", required=True)
    command.add_argument("--output", type=Path, required=True)
    command = sub.add_parser("parse-magscot-hmm")
    command.add_argument("--pfam", type=Path, required=True)
    command.add_argument("--tigr", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command = sub.add_parser("materialize-bins")
    command.add_argument("--assembly", type=Path, required=True)
    command.add_argument("--mapping", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--prefix", required=True)
    command.add_argument("--suffix", default="fa", help="MAG file extension (default: fa)")
    command = sub.add_parser("materialize-refined")
    command.add_argument("--assembly", type=Path, required=True)
    command.add_argument("--mapping", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--prefix", required=True)
    command.add_argument("--source", type=Path, action="append", default=[])
    command.add_argument("--suffix", default="fa", help="MAG file extension (default: fa)")
    command = sub.add_parser("stage-bams")
    command.add_argument("--bam", action="append", required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command = sub.add_parser("collect-fasta")
    command.add_argument("--source", type=Path, action="append", required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command = sub.add_parser("publish-fasta")
    command.add_argument("--source-dir", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--prefix", required=True)
    command.add_argument("--suffix", default="fa", help="MAG file extension (default: fa)")
    command = sub.add_parser("normalize-refined-fasta")
    command.add_argument("--source-dir", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--prefix", required=True)
    command.add_argument("--source", type=Path, action="append", default=[])
    command.add_argument("--suffix", default="fa", help="MAG file extension (default: fa)")
    command = sub.add_parser("tag-contigs")
    command.add_argument("--input-dir", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--suffix", default="fa", help="MAG file extension (default: fa)")
    command = sub.add_parser("filter-quality")
    command.add_argument("--bins-dir", type=Path, required=True)
    command.add_argument("--checkm2", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--summary", type=Path, required=True)
    command.add_argument("--min-completeness", type=float, required=True)
    command.add_argument("--max-contamination", type=float, required=True)
    command.add_argument("--gunc-dir", type=Path)
    command.add_argument("--gunc-min-reference-representation", type=float, default=0.30)
    command.add_argument("--fail-gunc-unscored", action="store_true")
    command.add_argument("--min-quality-score", type=float)
    command.add_argument("--rna-summary", type=Path)
    command = sub.add_parser("rna-qc")
    command.add_argument("--bins-dir", type=Path, required=True)
    command.add_argument("--marker-dir", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--summary", type=Path, required=True)
    command.add_argument("--trnascan", default="tRNAscan-SE")
    command.add_argument("--barrnap", default="barrnap")
    command.add_argument("--threads", type=int, default=1)
    command.add_argument("--trna", action="store_true")
    command.add_argument("--trna-pass", type=int)
    command.add_argument("--rrna", action="store_true")
    command.add_argument("--rrna-pass", action="store_true")
    command.add_argument("--completion-marker", type=Path)
    command = sub.add_parser("merge-coverm-taxonomy")
    command.add_argument("--input", action="append", required=True)
    command.add_argument("--taxonomy-dir", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--output-suffix", default=".tsv")
    command = sub.add_parser("classify-niche-literature")
    command.add_argument("--abundance", type=Path, required=True)
    command.add_argument("--taxonomy-dir", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--aggregated-output", type=Path, required=True)
    command.add_argument("--assignments-output", type=Path, required=True)
    command.add_argument("--provenance-output", type=Path, required=True)
    command.add_argument("--sample-dataset", action="append", required=True)
    command.add_argument(
        "--rank",
        choices=("domain", "phylum", "class", "order", "family", "genus", "species", "strain"),
        required=True,
    )
    command.add_argument("--detection-percent", type=float, required=True)
    command.add_argument("--min-total-reads", type=int, required=True)
    command.add_argument("--min-prevalence", type=float, required=True)
    command.add_argument("--core-prevalence", type=float, required=True)
    command.add_argument("--min-core-datasets", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    apply_memory_limit_from_environment()
    args = build_parser().parse_args(argv)
    dispatch = {
        "fasta-filter": lambda: fasta_filter(args.input, args.output, args.min_length),
        "concat-fasta": lambda: concatenate_fastas(
            args.input,
            args.output,
            args.min_length,
            args.separator,
        ),
        "bins-to-map": lambda: bins_to_map(
            args.bins_dir,
            args.output,
            args.label,
            args.min_bin_bp,
        ),
        "normalize-map": lambda: normalize_map(args.input, args.output, args.label, args.order),
        "combine-maps": lambda: combine_maps(args.input, args.output),
        "map-for-dastool": lambda: map_for_dastool(args.input, args.output),
        "merge-aemb": lambda: merge_aemb(args.input, args.output),
        "parse-magscot-hmm": lambda: parse_magscot_hmm(args.pfam, args.tigr, args.output),
        "materialize-bins": lambda: materialize_bins(
            args.assembly, args.mapping, args.output_dir, args.prefix, args.suffix
        ),
        "materialize-refined": lambda: materialize_refined(
            args.assembly, args.mapping, args.output_dir, args.prefix, args.source, args.suffix
        ),
        "stage-bams": lambda: stage_bams(args.bam, args.output_dir),
        "collect-fasta": lambda: collect_fasta(args.source, args.output_dir),
        "publish-fasta": lambda: publish_fasta(
            args.source_dir, args.output_dir, args.prefix, args.suffix
        ),
        "normalize-refined-fasta": lambda: normalize_refined_fasta(
            args.source_dir,
            args.output_dir,
            args.prefix,
            args.source,
            args.suffix,
        ),
        "tag-contigs": lambda: tag_contigs(args.input_dir, args.output_dir, args.suffix),
        "filter-quality": lambda: filter_quality(
            args.bins_dir,
            args.checkm2,
            args.output_dir,
            args.summary,
            args.min_completeness,
            args.max_contamination,
            args.gunc_dir,
            args.gunc_min_reference_representation,
            args.fail_gunc_unscored,
            args.min_quality_score,
            args.rna_summary,
        ),
        "rna-qc": lambda: rna_qc(
            args.bins_dir,
            args.marker_dir,
            args.output_dir,
            args.summary,
            args.trnascan,
            args.barrnap,
            args.threads,
            args.trna,
            args.trna_pass,
            args.rrna,
            args.rrna_pass,
            args.completion_marker,
        ),
        "merge-coverm-taxonomy": lambda: merge_coverm_taxonomy(
            args.input, args.taxonomy_dir, args.output_dir, args.output_suffix
        ),
        "classify-niche-literature": lambda: classify_niche_literature(
            args.abundance,
            args.taxonomy_dir,
            args.output,
            args.aggregated_output,
            args.assignments_output,
            args.provenance_output,
            args.sample_dataset,
            args.rank,
            args.detection_percent,
            args.min_total_reads,
            args.min_prevalence,
            args.core_prevalence,
            args.min_core_datasets,
        ),
    }
    try:
        dispatch[args.command]()
    except Exception as exc:
        print(f"metaBAW internal error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
