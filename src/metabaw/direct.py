from __future__ import annotations

from dataclasses import dataclass, replace
import os
from pathlib import Path
import shlex
import sys
from typing import Iterable

from .discovery import Analysis, ReadSample, public_sample_name
from .model import Task, topological_order


BINNER_ORDER = (
    "metabat2",
    "metadecoder",
    "vamb",
    "comebin",
    "semibin2",
    "lorbin",
)
BINNER_PRIORITY = {name: (position + 1) * 10 for position, name in enumerate(BINNER_ORDER)}
BINNER_OUTPUT_LABELS = {
    "metabat2": "MetaBAT2",
    "metadecoder": "MetaDecoder",
    "vamb": "VAMB",
    "comebin": "COMEBin",
    "semibin2": "SemiBin2",
    "lorbin": "LorBin",
}
CHECKM2_THREAD_ENV = {
    "BLIS_NUM_THREADS": "1",
    "MALLOC_ARENA_MAX": "2",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "TF_NUM_INTEROP_THREADS": "1",
    "TF_NUM_INTRAOP_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


def q(value: str | Path) -> str:
    return shlex.quote(str(value))


def safe_path_component(value: str) -> str:
    cleaned = "".join(
        char if char.isalnum() or char in "._-" else "_"
        for char in value
    )
    return cleaned.strip("._") or "sample"


def internal(*arguments: str | Path) -> tuple[str, ...]:
    return (sys.executable, "-m", "metabaw.internal", *(str(argument) for argument in arguments))


def parse_choices(values: Iterable[str]) -> list[str]:
    parsed: list[str] = []
    for value in values:
        parsed.extend(item.strip().lower() for item in value.split(",") if item.strip())
    return list(dict.fromkeys(parsed))


@dataclass(frozen=True)
class BinOptions:
    outdir: Path
    workdir: Path
    threads: int
    read_type: str
    align_tool: str
    long_read_preset: str
    binners: tuple[str, ...]
    min_contig_length: int
    min_fasta_kbs: int
    batch_size: int
    refiner: str
    quality_control: str
    min_completeness: float
    max_contamination: float
    min_quality_score: float | None
    run_gunc: bool
    run_trna: bool
    trna_pass: int | None
    run_rrna: bool
    rrna_pass: bool
    dereplicator: str
    environment: str | None
    tag_contigs: bool
    gpu: bool
    max_gpu_memory: str
    magscot_dir: Path
    checkm2_db: Path | None
    gunc_db: Path | None
    gtdbtk_data: Path | None
    extra_args: dict[str, str]
    ani: float = 99.0
    min_aligned_fraction: float = 30.0
    mag_suffix: str = "fa"
    comebin_run_prefix: tuple[str, ...] = (
        "conda",
        "run",
        "--name",
        "metabaw-comebin-py37",
    )
    checkm2_run_prefix: tuple[str, ...] = (
        "conda",
        "run",
        "--name",
        "metabaw-checkm2-py312",
    )
    metawrap_run_prefix: tuple[str, ...] = (
        "conda",
        "run",
        "--name",
        "metabaw-metawrap-py27",
    )
    lorbin_run_prefix: tuple[str, ...] = (
        "conda",
        "run",
        "--name",
        "metabaw-lorbin-py310",
    )


class DirectBinBuilder:
    def __init__(self, analyses: list[Analysis], options: BinOptions, root: Path):
        self.analyses = analyses
        self.options = options
        self.root = root
        self.tasks: list[Task] = []
        self._semibin_multisample_outputs: dict[str, tuple[Task, Path]] = {}
        self._active_sample: str | None = None

    def add(self, task: Task) -> Task:
        if task.sample is None and self._active_sample is not None:
            task = replace(task, sample=self._active_sample)
        self.tasks.append(task)
        return task

    def build(self) -> list[Task]:
        self._prepare_semibin_multisample()
        refined: list[tuple[Task, Path]] = []
        for analysis in self.analyses:
            refined.append(self._analysis(analysis))
        self._add_binning_barrier()
        catalog = self._catalog(refined)
        check_task, check_report = self._quality(catalog)
        gunc_task = self._gunc(catalog)
        rna_task, rna_summary = self._rna(catalog)
        filtered_task, filtered = self._filter(
            catalog, check_task, check_report, gunc_task, rna_task, rna_summary
        )
        final_task, final_dir = self._dereplicate(filtered_task, filtered, check_report)
        if self.options.tag_contigs:
            tagged = self.options.outdir / "non_redundant_bins"
            self.add(
                Task(
                    id="07.tag_contigs",
                    stage="07_catalog",
                    command=internal("tag-contigs", "--input-dir", final_dir, "--output-dir", tagged, "--suffix", self.options.mag_suffix),
                    cwd=self.root,
                    deps=(final_task.id,),
                    inputs=(final_dir,),
                    outputs=(tagged,),
                    description="Rename contigs to BIN_1, BIN_2, ...",
                )
            )
        return topological_order(self.tasks)

    def _add_binning_barrier(self) -> None:
        """Prevent refinement until every sample-specific binning task is terminal."""
        binning_dependencies = tuple(
            task.id for task in self.tasks if task.stage == "03_binning"
        )
        if not binning_dependencies:
            return
        marker = self.options.workdir / "binning" / "binning.complete"
        barrier = self.add(
            Task(
                id="03.binning.complete",
                stage="03_binning",
                command=(
                    sys.executable,
                    "-c",
                    (
                        "from pathlib import Path; import sys; "
                        "path=Path(sys.argv[1]); "
                        "path.parent.mkdir(parents=True, exist_ok=True); "
                        "path.write_text('complete\\n')"
                    ),
                    str(marker),
                ),
                cwd=self.root,
                deps=binning_dependencies,
                outputs=(marker,),
                allow_failed_deps=True,
                failure_tolerated=True,
                priority=1000,
                description=(
                    "Confirm all selected binners finished across all samples"
                ),
            )
        )
        self.tasks = [
            (
                replace(task, wait_for=(*task.wait_for, barrier.id))
                if task.stage == "04_refinement"
                and barrier.id not in task.wait_for
                else task
            )
            for task in self.tasks
        ]

    def _analysis(self, analysis: Analysis) -> tuple[Task, Path]:
        self._active_sample = analysis.name
        prepared = self.options.workdir / "contigs" / f"{analysis.name}.fna"
        prepare_args: list[str | Path] = ["concat-fasta"]
        labels: list[str] = []
        if len(analysis.contigs) == 1:
            labels = [analysis.name]
        else:
            by_contig: dict[Path, str] = {}
            for sample in analysis.samples:
                if sample.contigs is not None:
                    by_contig.setdefault(sample.contigs, sample.name)
            labels = [by_contig[path] for path in analysis.contigs]
        for label, path in zip(labels, analysis.contigs, strict=True):
            prepare_args.extend(("--input", f"{label}={path}"))
        prepare_args.extend(("--output", prepared, "--min-length", str(self.options.min_contig_length)))
        prepare = self.add(
            Task(
                id=f"01.prepare.{analysis.name}",
                stage="01_prepare",
                command=internal(*prepare_args),
                cwd=self.root,
                inputs=analysis.contigs,
                outputs=(prepared,),
                description=f"Filter and prepare contigs for {analysis.name}",
            )
        )
        bam_task, bams, bam_dir = self._mapping(analysis, prepare, prepared)
        maps = self._binners(analysis, prepare, prepared, bam_task, bams, bam_dir)
        semibin_multisample = self._semibin_multisample_outputs.get(
            analysis.name
        )
        if semibin_multisample is not None:
            run, bins = semibin_multisample
            mapping = (
                self.options.workdir
                / "binning"
                / analysis.name
                / "semibin2"
                / "contig_to_bin.tsv"
            )
            map_task = self._map_bins(analysis, "semibin2", run, bins, mapping)
            maps.append(
                self._publish_binner(
                    analysis,
                    "semibin2",
                    prepare,
                    prepared,
                    map_task,
                    mapping,
                )
            )
        result = self._refine(analysis, prepare, prepared, maps)
        self._active_sample = None
        return result

    def _prepare_semibin_multisample(self) -> None:
        """Build SemiBin2's concatenated input from independently assembled samples."""
        if "semibin2" not in self.options.binners:
            return
        analyses = [analysis for analysis in self.analyses if analysis.cross_mapped]
        if not analyses:
            return
        cohorts: dict[tuple[str, ...], list[Analysis]] = {}
        for analysis in analyses:
            cohort = tuple(sample.name for sample in analysis.samples)
            cohorts.setdefault(cohort, []).append(analysis)

        multiple_cohorts = len(cohorts) > 1
        for cohort_number, cohort_analyses in enumerate(
            cohorts.values(),
            start=1,
        ):
            combined_name = "semibin2_multisample_input"
            if multiple_cohorts:
                combined_name += f"_{cohort_number:03d}"
            self._active_sample = combined_name
            combined = (
                self.options.workdir
                / "contigs"
                / f"{combined_name}.fna"
            )
            prepare_args: list[str | Path] = ["concat-fasta"]
            contig_paths: list[Path] = []
            for analysis in cohort_analyses:
                if len(analysis.contigs) != 1:
                    raise ValueError(
                        "SemiBin2 multi-sample mode requires one assembly "
                        "per analysis"
                    )
                contig = analysis.contigs[0]
                contig_paths.append(contig)
                prepare_args.extend(("--input", f"{analysis.name}={contig}"))
            prepare_args.extend(
                (
                    "--output",
                    combined,
                    "--min-length",
                    str(self.options.min_contig_length),
                    "--separator",
                    ":",
                )
            )
            prepare = self.add(
                Task(
                    id=f"01.prepare.{combined_name}",
                    stage="01_prepare",
                    command=internal(*prepare_args),
                    cwd=self.root,
                    inputs=tuple(contig_paths),
                    outputs=(combined,),
                    description=(
                        "Prepare sample-labelled contigs for SemiBin2 "
                        "multi-sample binning"
                    ),
                )
            )

            mapping_analysis = Analysis(
                combined_name,
                cohort_analyses[0].samples,
                tuple(contig_paths),
                False,
            )
            bam_task, bams, _bam_dir = self._mapping(
                mapping_analysis,
                prepare,
                combined,
            )
            tool_root = (
                self.options.workdir
                / "binning"
                / combined_name
                / "semibin2"
            )
            sequence_type = (
                " --sequencing-type long_read"
                if self.options.read_type == "long"
                else ""
            )
            engine = " --engine gpu" if self.options.gpu else " --engine cpu"
            sample_bins = tuple(
                tool_root / "samples" / analysis.name / "output_bins"
                for analysis in cohort_analyses
            )
            run_id = "03.bin.semibin2.multi"
            if multiple_cohorts:
                run_id += f".{cohort_number:03d}"
            run = self.add(
                Task(
                    id=run_id,
                    stage="03_binning",
                    command=(
                        f"rm -rf {q(tool_root)} && SemiBin2 multi_easy_bin "
                        f"-i {q(combined)} "
                        f"-b {' '.join(q(path) for path in bams)} "
                        f"-o {q(tool_root)} --threads {self.options.threads} "
                        f"--min-len {self.options.min_contig_length} "
                        f"--separator : --self-supervised"
                        f"{sequence_type}{engine}{self._extra('semibin2')}"
                    ),
                    cwd=self.root,
                    deps=(prepare.id, bam_task.id),
                    inputs=(combined, *bams),
                    outputs=sample_bins,
                    fasta_output_dirs=sample_bins,
                    automatic_retries=1,
                    cpus=self.options.threads,
                    gpus=1 if self.options.gpu else 0,
                    priority=BINNER_PRIORITY["semibin2"],
                    failure_tolerated=True,
                    env=self._gpu_env(),
                    description=(
                        "Run SemiBin2 multi_easy_bin for "
                        f"{len(cohort_analyses)} independently assembled samples"
                    ),
                )
            )
            self._semibin_multisample_outputs.update(
                {
                    analysis.name: (run, bins)
                    for analysis, bins in zip(
                        cohort_analyses,
                        sample_bins,
                        strict=True,
                    )
                }
            )
        self._active_sample = None

    def _mapping(
        self, analysis: Analysis, prepare: Task, contigs: Path
    ) -> tuple[Task, list[Path], Path]:
        root = self.options.workdir / "mapping" / analysis.name
        index_marker = root / f"{self.options.align_tool}.index.done"
        index_outputs: tuple[Path, ...] = (index_marker,)
        index_alternatives: tuple[tuple[Path, ...], ...] = ()
        if self.options.align_tool == "bowtie2":
            prefix = root / "index" / contigs.stem
            index_alternatives = tuple(
                tuple(
                    Path(f"{prefix}.{part}.{extension}")
                    for part in ("1", "2", "3", "4", "rev.1", "rev.2")
                )
                for extension in ("bt2", "bt2l")
            )
            index_command = (
                f"rm -rf {q(prefix.parent)} && mkdir -p {q(prefix.parent)} && "
                f"bowtie2-build --threads {self.options.threads} "
                f"{q(contigs)} {q(prefix)} && touch {q(index_marker)}"
            )
        elif self.options.align_tool == "minimap2":
            prefix = root / "index" / f"{contigs.stem}.mmi"
            index_outputs = (index_marker, prefix)
            index_command = (
                f"rm -rf {q(prefix.parent)} && mkdir -p {q(prefix.parent)} && "
                f"minimap2 -t {self.options.threads} -d {q(prefix)} "
                f"{q(contigs)} && touch {q(index_marker)}"
            )
        else:
            prefix = root / "index" / contigs.stem
            index_command = (
                f"rm -rf {q(prefix.parent)} && mkdir -p {q(prefix.parent)} && "
                f"minibwa index -t {self.options.threads} "
                f"{q(contigs)} {q(prefix)} && touch {q(index_marker)}"
            )
        index = self.add(
            Task(
                id=f"02.index.{analysis.name}",
                stage="02_mapping",
                command=index_command,
                cwd=self.root,
                deps=(prepare.id,),
                inputs=(contigs,),
                outputs=index_outputs,
                output_alternatives=index_alternatives,
                cpus=self.options.threads,
                description=f"Build {self.options.align_tool} index for {analysis.name}",
            )
        )

        bams: list[Path] = []
        map_tasks: list[Task] = []
        for sample in analysis.samples:
            bam_name = (
                f"{sample.name}_to_{analysis.name}" if analysis.cross_mapped else sample.name
            )
            bam = root / "bam" / f"{bam_name}.bam"
            reads = " ".join(q(path) for path in sample.reads)
            filter_pipe = (
                f"samtools view -@ {self.options.threads} -b -q 10 -F 3588 - | "
                f"samtools sort -@ {self.options.threads} -o {q(bam)} - && "
                f"samtools index -@ {self.options.threads} {q(bam)}"
            )
            if self.options.align_tool == "bowtie2":
                read_args = (
                    f"-1 {q(sample.read1)} -2 {q(sample.read2)}"
                    if sample.read2
                    else f"-U {q(sample.read1)}"
                )
                mapper = (
                    f"bowtie2 --very-sensitive -p {self.options.threads} -x {q(prefix)} "
                    f"{read_args} 2> {q(root / 'bam' / (bam_name + '.bowtie2.log'))}"
                )
            elif self.options.align_tool == "minimap2":
                preset = "sr" if self.options.read_type == "short" else self.options.long_read_preset
                mapper = f"minimap2 -ax {preset} -t {self.options.threads} {q(prefix)} {reads}"
            else:
                mapper = f"minibwa map -t {self.options.threads} {q(prefix)} {reads}"
            task = self.add(
                Task(
                    id=f"02.map.{analysis.name}.{sample.name}",
                    stage="02_mapping",
                    command=f"mkdir -p {q(bam.parent)} && {mapper} | {filter_pipe}",
                    cwd=self.root,
                    deps=(index.id,),
                    inputs=(contigs, *sample.reads),
                    outputs=(bam, Path(str(bam) + ".bai")),
                    cpus=self.options.threads,
                    description=f"Map {sample.name} reads to {analysis.name}",
                )
            )
            bams.append(bam)
            map_tasks.append(task)
        bam_dir = root / "bamset"
        stage_args: list[str | Path] = ["stage-bams"]
        staged_outputs: list[Path] = [bam_dir]
        for sample, bam in zip(analysis.samples, bams, strict=True):
            stage_args.extend(("--bam", f"{sample.name}={bam}"))
            staged_bam = bam_dir / f"{safe_path_component(sample.name)}.bam"
            staged_outputs.extend((staged_bam, Path(str(staged_bam) + ".bai")))
        stage_args.extend(("--output-dir", bam_dir))
        bam_task = self.add(
            Task(
                id=f"02.bamset.{analysis.name}",
                stage="02_mapping",
                command=internal(*stage_args),
                cwd=self.root,
                deps=tuple(task.id for task in map_tasks),
                inputs=tuple(bams),
                outputs=tuple(staged_outputs),
                description=f"Stage BAMs for {analysis.name}",
            )
        )
        return bam_task, bams, bam_dir

    def _binners(
        self,
        analysis: Analysis,
        prepare: Task,
        contigs: Path,
        bam_task: Task,
        bams: list[Path],
        bam_dir: Path,
    ) -> list[tuple[str, Task, Path, Path]]:
        results: list[tuple[str, Task, Path, Path]] = []
        root = self.options.workdir / "binning" / analysis.name
        for binner in self.options.binners:
            priority = BINNER_PRIORITY[binner]
            if binner == "metabat2":
                depth = root / binner / "depth.tsv"
                depth_task = self.add(
                    Task(
                        id=f"03.depth.{binner}.{analysis.name}",
                        stage="03_binning",
                        command=(
                            f"mkdir -p {q(depth.parent)} && jgi_summarize_bam_contig_depths "
                            f"--outputDepth {q(depth)} {' '.join(q(path) for path in bams)}"
                        ),
                        cwd=self.root,
                        deps=(bam_task.id,),
                        inputs=tuple(bams),
                        outputs=(depth,),
                        priority=priority,
                        failure_tolerated=True,
                        description="Calculate MetaBAT2 depth matrix",
                    )
                )
                bins = root / binner / "bins"
                run = self.add(
                    Task(
                        id=f"03.bin.{binner}.{analysis.name}",
                        stage="03_binning",
                        command=(
                            f"rm -rf {q(bins)} && mkdir -p {q(bins)} && metabat2 -i {q(contigs)} "
                            f"-a {q(depth)} -o {q(bins / (analysis.name + '.metabat2'))} "
                            f"-m {self.options.min_contig_length} -s {self.options.min_fasta_kbs * 1000} "
                            f"--seed 42 -t {self.options.threads}{self._extra(binner)}"
                        ),
                        cwd=self.root,
                        deps=(prepare.id, depth_task.id),
                        inputs=(contigs, depth),
                        outputs=(bins,),
                        fasta_output_dirs=(bins,),
                        automatic_retries=1,
                        cpus=self.options.threads,
                        priority=priority,
                        failure_tolerated=True,
                        description=f"Run MetaBAT2 for {analysis.name}",
                    )
                )
                mapping = root / binner / "contig_to_bin.tsv"
                map_task = self._map_bins(analysis, binner, run, bins, mapping)
            elif binner == "vamb":
                vamb_root = root / binner / "run"
                deps = [prepare.id]
                inputs: list[Path] = [contigs]
                abundance_arg: str
                if self.options.read_type == "short":
                    aemb_tasks: list[Task] = []
                    aemb_paths: list[Path] = []
                    for sample in analysis.samples:
                        aemb = root / binner / "aemb" / f"{sample.name}.tsv"
                        reads = " ".join(q(path) for path in sample.reads)
                        aemb_tasks.append(
                            self.add(
                                Task(
                                    id=f"03.aemb.{analysis.name}.{sample.name}",
                                    stage="03_binning",
                                    command=(
                                        f"mkdir -p {q(aemb.parent)} && strobealign --aemb -t "
                                        f"{self.options.threads} {q(contigs)} {reads} > {q(aemb)}"
                                    ),
                                    cwd=self.root,
                                    deps=(prepare.id,),
                                    inputs=(contigs, *sample.reads),
                                    outputs=(aemb,),
                                    cpus=self.options.threads,
                                    priority=priority,
                                    failure_tolerated=True,
                                    description=f"Estimate {sample.name} abundance with strobealign AEMB",
                                )
                            )
                        )
                        aemb_paths.append(aemb)
                    abundance = root / binner / "abundance.tsv"
                    merge_args: list[str | Path] = ["merge-aemb"]
                    for sample, path in zip(analysis.samples, aemb_paths, strict=True):
                        merge_args.extend(("--input", f"{sample.name}={path}"))
                    merge_args.extend(("--output", abundance))
                    merge = self.add(
                        Task(
                            id=f"03.aemb.merge.{analysis.name}",
                            stage="03_binning",
                            command=internal(*merge_args),
                            cwd=self.root,
                            deps=tuple(task.id for task in aemb_tasks),
                            inputs=tuple(aemb_paths),
                            outputs=(abundance,),
                            priority=priority,
                            failure_tolerated=True,
                            description="Merge AEMB profiles with mandatory contigname header",
                        )
                    )
                    deps.append(merge.id)
                    inputs.append(abundance)
                    abundance_arg = f"--abundance_tsv {q(abundance)}"
                else:
                    deps.append(bam_task.id)
                    inputs.append(bam_dir)
                    abundance_arg = f"--bamdir {q(bam_dir)}"
                cuda = " --cuda" if self.options.gpu else ""
                bins = vamb_root / "bins"
                run = self.add(
                    Task(
                        id=f"03.bin.{binner}.{analysis.name}",
                        stage="03_binning",
                        command=(
                            f"rm -rf {q(vamb_root)} && mkdir -p {q(vamb_root.parent)} && "
                            f"vamb bin default --outdir {q(vamb_root)} --fasta {q(contigs)} "
                            f"{abundance_arg} -m {self.options.min_contig_length} "
                            f"--minfasta {self.options.min_fasta_kbs * 1000} --seed 42 "
                            f"-p {self.options.threads}{cuda}{self._extra(binner)}"
                        ),
                        cwd=self.root,
                        deps=tuple(deps),
                        inputs=tuple(inputs),
                        outputs=(bins,),
                        fasta_output_dirs=(bins,),
                        automatic_retries=1,
                        cpus=self.options.threads,
                        gpus=1 if self.options.gpu else 0,
                        priority=priority,
                        failure_tolerated=True,
                        env=self._gpu_env(),
                        description=f"Run VAMB for {analysis.name}",
                    )
                )
                mapping = root / binner / "contig_to_bin.tsv"
                map_task = self._map_bins(
                    analysis,
                    binner,
                    run,
                    bins,
                    mapping,
                    self.options.min_fasta_kbs * 1000,
                )
            elif binner == "metadecoder":
                tool_root = root / binner
                coverage = tool_root / "coverage.tsv"
                seeds = tool_root / "seed.tsv"
                cov_task = self.add(
                    Task(
                        id=f"03.coverage.{binner}.{analysis.name}",
                        stage="03_binning",
                        command=(
                            f"mkdir -p {q(tool_root)} && metadecoder coverage "
                            f"-b {' '.join(q(path) for path in bams)} -o {q(coverage)}"
                        ),
                        cwd=self.root,
                        deps=(bam_task.id,),
                        inputs=tuple(bams),
                        outputs=(coverage,),
                        priority=priority,
                        failure_tolerated=True,
                        description="Calculate MetaDecoder coverage",
                    )
                )
                seed_task = self.add(
                    Task(
                        id=f"03.seed.{binner}.{analysis.name}",
                        stage="03_binning",
                        command=(
                            f"mkdir -p {q(tool_root)} && metadecoder seed --threads {self.options.threads} "
                            f"-f {q(contigs)} -o {q(seeds)}"
                        ),
                        cwd=self.root,
                        deps=(prepare.id,),
                        inputs=(contigs,),
                        outputs=(seeds,),
                        cpus=self.options.threads,
                        priority=priority,
                        failure_tolerated=True,
                        description="Find MetaDecoder marker seeds",
                    )
                )
                bins = tool_root / "bins"
                run = self.add(
                    Task(
                        id=f"03.bin.{binner}.{analysis.name}",
                        stage="03_binning",
                        command=(
                            f"rm -rf {q(bins)} && mkdir -p {q(bins)} && metadecoder cluster "
                            f"-f {q(contigs)} -c {q(coverage)} -s {q(seeds)} "
                            f"-o {q(bins / (analysis.name + '.metadecoder'))} "
                            f"--threads {self.options.threads}{self._extra(binner)}"
                        ),
                        cwd=self.root,
                        deps=(cov_task.id, seed_task.id),
                        inputs=(contigs, coverage, seeds),
                        outputs=(bins,),
                        fasta_output_dirs=(bins,),
                        automatic_retries=1,
                        cpus=self.options.threads,
                        priority=priority,
                        failure_tolerated=True,
                        description=f"Run MetaDecoder for {analysis.name}",
                    )
                )
                mapping = tool_root / "contig_to_bin.tsv"
                map_task = self._map_bins(analysis, binner, run, bins, mapping)
            elif binner == "comebin":
                tool_root = root / binner
                bins = tool_root / "comebin_res" / "comebin_res_bins"
                runner = " ".join(q(argument) for argument in self.options.comebin_run_prefix)
                run = self.add(
                    Task(
                        id=f"03.bin.{binner}.{analysis.name}",
                        stage="03_binning",
                        command=(
                            f"rm -rf {q(tool_root)} && mkdir -p {q(tool_root)} && "
                            f"{runner} run_comebin.sh "
                            f"-a {q(contigs)} -p {q(bam_dir)} "
                            f"-o {q(tool_root)} -t {self.options.threads} -b {self.options.batch_size}"
                            f"{self._extra(binner)}"
                        ),
                        cwd=self.root,
                        deps=(prepare.id, bam_task.id),
                        inputs=(contigs, bam_dir),
                        outputs=(bins,),
                        fasta_output_dirs=(bins,),
                        automatic_retries=1,
                        cpus=self.options.threads,
                        gpus=1 if self.options.gpu else 0,
                        priority=priority,
                        failure_tolerated=True,
                        env=self._gpu_env(),
                        description=f"Run COMEBin for {analysis.name}",
                    )
                )
                mapping = tool_root / "contig_to_bin.tsv"
                map_task = self._map_bins(analysis, binner, run, bins, mapping)
            elif binner == "semibin2":
                if analysis.cross_mapped:
                    continue
                tool_root = root / binner
                mode = "single_easy_bin"
                has_multiple_bams = len(bams) > 1
                environment = (
                    f" --environment {q(self.options.environment)}"
                    if self.options.environment and not has_multiple_bams
                    else ""
                )
                training = (
                    " --self-supervised"
                    if has_multiple_bams or not self.options.environment
                    else ""
                )
                sequence_type = " --sequencing-type long_read" if self.options.read_type == "long" else ""
                engine = " --engine gpu" if self.options.gpu else " --engine cpu"
                bins = tool_root / "output_bins"
                run = self.add(
                    Task(
                        id=f"03.bin.{binner}.{analysis.name}",
                        stage="03_binning",
                        command=(
                            f"rm -rf {q(tool_root)} && SemiBin2 {mode} -i {q(contigs)} -b "
                            f"{' '.join(q(path) for path in bams)} -o {q(tool_root)} "
                            f"--threads {self.options.threads} --min-len {self.options.min_contig_length}"
                            f"{environment}{training}{sequence_type}{engine}{self._extra(binner)}"
                        ),
                        cwd=self.root,
                        deps=(prepare.id, bam_task.id),
                        inputs=(contigs, *bams),
                        outputs=(bins,),
                        fasta_output_dirs=(bins,),
                        automatic_retries=1,
                        cpus=self.options.threads,
                        gpus=1 if self.options.gpu else 0,
                        priority=priority,
                        failure_tolerated=True,
                        env=self._gpu_env(),
                        description=f"Run SemiBin2 for {analysis.name}",
                    )
                )
                mapping = tool_root / "contig_to_bin.tsv"
                map_task = self._map_bins(analysis, binner, run, bins, mapping)
            elif binner == "lorbin":
                tool_root = root / binner
                bins = tool_root / "output_bins"
                runner = " ".join(
                    q(argument) for argument in self.options.lorbin_run_prefix
                )
                cuda = " --cuda" if self.options.gpu else ""
                run = self.add(
                    Task(
                        id=f"03.bin.{binner}.{analysis.name}",
                        stage="03_binning",
                        command=(
                            f"rm -rf {q(tool_root)} && {runner} LorBin bin "
                            f"-o {q(tool_root)} -fa {q(contigs)} "
                            f"-b {' '.join(q(path) for path in bams)} "
                            f"--num_process {self.options.threads}"
                            f"{cuda}{self._extra(binner)}"
                        ),
                        cwd=self.root,
                        deps=(prepare.id, bam_task.id),
                        inputs=(contigs, *bams),
                        outputs=(bins,),
                        fasta_output_dirs=(bins,),
                        automatic_retries=1,
                        cpus=self.options.threads,
                        gpus=1 if self.options.gpu else 0,
                        priority=priority,
                        failure_tolerated=True,
                        env=self._gpu_env(),
                        description=f"Run LorBin for {analysis.name}",
                    )
                )
                mapping = tool_root / "contig_to_bin.tsv"
                map_task = self._map_bins(
                    analysis,
                    binner,
                    run,
                    bins,
                    mapping,
                    self.options.min_fasta_kbs * 1000,
                )
            else:
                raise ValueError(f"Unsupported binner: {binner}")

            results.append(
                self._publish_binner(
                    analysis,
                    binner,
                    prepare,
                    contigs,
                    map_task,
                    mapping,
                )
            )
        return results

    def _publish_binner(
        self,
        analysis: Analysis,
        binner: str,
        prepare: Task,
        contigs: Path,
        map_task: Task,
        mapping: Path,
    ) -> tuple[str, Task, Path, Path]:
        public_bins = self.options.outdir / "bin_files" / binner / analysis.name
        output_prefix = (
            f"{public_sample_name(analysis.name)}_{BINNER_OUTPUT_LABELS[binner]}"
        )
        materialize = self.add(
            Task(
                id=f"03.publish.{binner}.{analysis.name}",
                stage="03_binning",
                command=internal(
                    "materialize-bins",
                    "--assembly",
                    contigs,
                    "--mapping",
                    mapping,
                    "--output-dir",
                    public_bins,
                    "--prefix",
                    output_prefix,
                    "--suffix",
                    self.options.mag_suffix,
                ),
                cwd=self.root,
                deps=(prepare.id, map_task.id),
                inputs=(contigs, mapping),
                outputs=(public_bins, public_bins / "manifest.tsv"),
                priority=BINNER_PRIORITY[binner],
                failure_tolerated=True,
                description=f"Publish normalized {binner} bins",
            )
        )
        return binner, materialize, mapping, public_bins

    def _map_bins(
        self,
        analysis: Analysis,
        binner: str,
        run: Task,
        bins: Path,
        mapping: Path,
        minimum_bin_bp: int = 0,
    ) -> Task:
        arguments: list[str | Path] = [
            "bins-to-map",
            "--bins-dir",
            bins,
            "--output",
            mapping,
            "--label",
            binner,
        ]
        if minimum_bin_bp:
            arguments.extend(("--min-bin-bp", str(minimum_bin_bp)))
        return self.add(
            Task(
                id=f"03.map.{binner}.{analysis.name}",
                stage="03_binning",
                command=internal(*arguments),
                cwd=self.root,
                deps=(run.id,),
                inputs=(bins,),
                outputs=(mapping,),
                priority=BINNER_PRIORITY[binner],
                failure_tolerated=True,
                description=f"Normalize {binner} assignments",
            )
        )

    def _refine(
        self,
        analysis: Analysis,
        prepare: Task,
        contigs: Path,
        maps: list[tuple[str, Task, Path, Path]],
    ) -> tuple[Task, Path]:
        root = self.options.workdir / "refinement" / analysis.name
        sample_name = public_sample_name(analysis.name)
        if self.options.refiner == "magscot":
            combined = root / "all_binners.tsv"
            combine_args: list[str | Path] = ["combine-maps"]
            for _name, _task, path, _bins in maps:
                combine_args.extend(("--input", path))
            combine_args.extend(("--output", combined))
            combined_task = self.add(
                Task(
                    id=f"04.combine.{analysis.name}",
                    stage="04_refinement",
                    command=internal(*combine_args),
                    cwd=self.root,
                    deps=tuple(task.id for _name, task, _path, _bins in maps),
                    inputs=tuple(path for _name, _task, path, _bins in maps),
                    outputs=(combined,),
                    allow_failed_deps=True,
                    description="Combine binner assignments for MAGScoT",
                )
            )
            proteins = root / "markers" / "proteins.faa"
            genes = root / "markers" / "genes.ffn"
            prodigal_gff = root / "markers" / "prodigal.gff"
            prodigal_parts = root / "markers" / "prodigal_parts"
            prodigal_template = (
                f"prodigal -p meta "
                f"-a {q(prodigal_parts / 'proteins_{#}.faa')} "
                f"-d {q(prodigal_parts / 'genes_{#}.ffn')} "
                f"-o {q(prodigal_parts / 'prodigal_{#}.gff')}"
            )
            prodigal = self.add(
                Task(
                    id=f"04.prodigal.{analysis.name}",
                    stage="04_refinement",
                    command=(
                        f"rm -rf {q(prodigal_parts)} && mkdir -p {q(prodigal_parts)} && "
                        f"parallel --jobs {self.options.threads} --block 999k --recstart '>' "
                        f"--pipe {q(prodigal_template)} < {q(contigs)} && "
                        f"cat {q(prodigal_parts)}/proteins_*.faa > {q(proteins)} && "
                        f"cat {q(prodigal_parts)}/genes_*.ffn > "
                        f"{q(genes)} && "
                        f"cat {q(prodigal_parts)}/prodigal_*.gff > "
                        f"{q(prodigal_gff)}"
                    ),
                    cwd=self.root,
                    deps=(prepare.id,),
                    inputs=(contigs,),
                    outputs=(proteins, genes, prodigal_gff),
                    cpus=self.options.threads,
                    description="Predict genes for MAGScoT markers",
                )
            )
            hmm_tasks: list[Task] = []
            tables: list[Path] = []
            for name, hmm in (
                ("pfam", "gtdbtk_rel207_Pfam-A.hmm"),
                ("tigr", "gtdbtk_rel207_tigrfam.hmm"),
            ):
                table = root / "markers" / f"{name}.tbl"
                report = root / "markers" / f"{name}.out"
                hmm_tasks.append(
                    self.add(
                        Task(
                            id=f"04.hmm.{name}.{analysis.name}",
                            stage="04_refinement",
                            command=(
                                f"hmmsearch -o {q(report)} --tblout {q(table)} "
                                f"--noali --notextw --cut_nc --cpu {self.options.threads} "
                                f"{q(self.options.magscot_dir / 'hmm' / hmm)} {q(proteins)}"
                            ),
                            cwd=self.root,
                            deps=(prodigal.id,),
                            inputs=(proteins,),
                            outputs=(table, report),
                            cpus=self.options.threads,
                            description=f"Search {name} markers for MAGScoT",
                        )
                    )
                )
                tables.append(table)
            markers = root / "markers" / "magscot_markers.tsv"
            marker_task = self.add(
                Task(
                    id=f"04.markers.{analysis.name}",
                    stage="04_refinement",
                    command=internal(
                        "parse-magscot-hmm",
                        "--pfam",
                        tables[0],
                        "--tigr",
                        tables[1],
                        "--output",
                        markers,
                    ),
                    cwd=self.root,
                    deps=tuple(task.id for task in hmm_tasks),
                    inputs=tuple(tables),
                    outputs=(markers,),
                    description="Format marker hits for MAGScoT",
                )
            )
            refined_map = root / f"{analysis.name}.refined.contig_to_bin.out"
            refine = self.add(
                Task(
                    id=f"04.refine.magscot.{analysis.name}",
                    stage="04_refinement",
                    command=(
                        f"Rscript {q(self.options.magscot_dir / 'MAGScoT.R')} -i {q(combined)} "
                        f"--hmm {q(markers)} -o {q(analysis.name)} --score_a 1 --score_b 0.5 "
                        f"--score_c 0.5 --threshold {self.options.min_completeness / 100:g} "
                        f"--max_cont {self.options.max_contamination / 100:g} --min_markers 25 "
                        f"--min_sharing 0.8 --n_iterations 2{self._extra('magscot')}"
                    ),
                    cwd=root,
                    deps=(combined_task.id, marker_task.id),
                    inputs=(combined, markers),
                    outputs=(refined_map,),
                    description=f"Refine {analysis.name} bins with MAGScoT",
                )
            )
            bins = root / "bins"
            refined_args: list[str | Path] = [
                "materialize-refined",
                "--assembly",
                contigs,
                "--mapping",
                refined_map,
                "--output-dir",
                bins,
                "--prefix",
                f"{sample_name}_MAGScoT",
                "--suffix",
                self.options.mag_suffix,
            ]
            for _name, _task, _mapping, public_bins in maps:
                refined_args.extend(("--source", public_bins))
            result = self.add(
                Task(
                    id=f"04.bins.{analysis.name}",
                    stage="04_refinement",
                    command=internal(*refined_args),
                    cwd=self.root,
                    deps=(refine.id, *(task.id for _name, task, _mapping, _bins in maps)),
                    inputs=(contigs, refined_map, *(public for _name, _task, _mapping, public in maps)),
                    outputs=(bins, bins / "manifest.tsv"),
                    allow_failed_deps=True,
                    description="Write refined MAGs; unchanged bins keep original binner names",
                )
            )
            return result, bins

        if self.options.refiner == "das_tool":
            converted: list[tuple[str, Task, Path]] = []
            for name, task, mapping, _bins in maps:
                path = root / f"{name}.contig2bin.tsv"
                converted.append(
                    (
                        name,
                        self.add(
                            Task(
                                id=f"04.dastool.map.{name}.{analysis.name}",
                                stage="04_refinement",
                                command=internal(
                                    "map-for-dastool", "--input", mapping, "--output", path
                                ),
                                cwd=self.root,
                                deps=(task.id,),
                                inputs=(mapping,),
                                outputs=(path,),
                                description=f"Convert {name} assignments for DAS Tool",
                            )
                        ),
                        path,
                    )
                )
            prefix = root / analysis.name
            bins = Path(str(prefix) + "_DASTool_bins")
            completion = root / "dastool.complete"
            run = self.add(
                Task(
                    id=f"04.refine.das_tool.{analysis.name}",
                    stage="04_refinement",
                    command=(
                        f"mkdir -p {q(root)} && rm -rf {q(bins)} "
                        f"{q(completion)} && DAS_Tool -i "
                        f"{q(','.join(str(path) for _name, _task, path in converted))} -l "
                        f"{q(','.join(name for name, _task, _path in converted))} -c {q(contigs)} "
                        f"-o {q(prefix)} --search_engine diamond --write_bins "
                        f"-t {self.options.threads}{self._extra('das_tool')} && "
                        f"touch {q(completion)}"
                    ),
                    cwd=self.root,
                    deps=tuple(task.id for _name, task, _path in converted),
                    inputs=(contigs, *(path for _name, _task, path in converted)),
                    outputs=(completion,),
                    fasta_output_dirs=(bins,),
                    cpus=self.options.threads,
                    description=f"Refine {analysis.name} bins with DAS Tool",
                )
            )
            normalize_args: list[str | Path] = [
                "normalize-refined-fasta",
                "--source-dir",
                bins,
                "--output-dir",
                bins,
                "--prefix",
                f"{sample_name}_DASTool",
                "--suffix",
                self.options.mag_suffix,
            ]
            for _name, _task, _mapping, public_bins in maps:
                normalize_args.extend(("--source", public_bins))
            publish = self.add(
                Task(
                    id=f"04.bins.{analysis.name}",
                    stage="04_refinement",
                    command=internal(*normalize_args),
                    cwd=self.root,
                    deps=(
                        run.id,
                        *(task.id for _name, task, _mapping, _bins in maps),
                    ),
                    inputs=(
                        bins,
                        *(public_bins for _name, _task, _mapping, public_bins in maps),
                    ),
                    outputs=(bins, bins / "manifest.tsv"),
                    fasta_output_dirs=(bins,),
                    description=(
                        "Normalize DAS Tool MAG names and retain unchanged "
                        "source-binner provenance"
                    ),
                )
            )
            return publish, bins

        if len(maps) > 3:
            raise ValueError("MetaWRAP bin_refinement accepts at most three binner directories")
        letters = ("A", "B", "C")
        inputs = " ".join(
            f"-{letter} {q(bins)}" for letter, (_name, _task, _map, bins) in zip(letters, maps, strict=False)
        )
        threshold_name = (
            f"metawrap_{self.options.min_completeness:g}_{self.options.max_contamination:g}_bins"
        )
        bins = root / threshold_name
        completion = root / "metawrap.complete"
        runner = " ".join(q(argument) for argument in self.options.metawrap_run_prefix)
        run = self.add(
            Task(
                id=f"04.refine.metawrap.{analysis.name}",
                stage="04_refinement",
                command=(
                    f"mkdir -p {q(root.parent)} && rm -rf {q(root)} && "
                    f"{runner} metawrap bin_refinement "
                    f"-o {q(root)} -t {self.options.threads} "
                    f"{inputs} -c {self.options.min_completeness:g} "
                    f"-x {self.options.max_contamination:g}"
                    f"{self._extra('metawrap')} && touch {q(completion)}"
                ),
                cwd=self.root,
                deps=tuple(task.id for _name, task, _mapping, _bins in maps),
                inputs=tuple(bins_dir for _name, _task, _mapping, bins_dir in maps),
                outputs=(completion,),
                fasta_output_dirs=(bins,),
                cpus=self.options.threads,
                description=f"Refine {analysis.name} bins with MetaWRAP",
            )
        )
        normalize_args: list[str | Path] = [
            "normalize-refined-fasta",
            "--source-dir",
            bins,
            "--output-dir",
            bins,
            "--prefix",
            f"{sample_name}_MetaWRAP",
            "--suffix",
            self.options.mag_suffix,
        ]
        for _name, _task, _mapping, public_bins in maps:
            normalize_args.extend(("--source", public_bins))
        publish = self.add(
            Task(
                id=f"04.bins.{analysis.name}",
                stage="04_refinement",
                command=internal(*normalize_args),
                cwd=self.root,
                deps=(
                    run.id,
                    *(task.id for _name, task, _mapping, _bins in maps),
                ),
                inputs=(
                    bins,
                    *(public_bins for _name, _task, _mapping, public_bins in maps),
                ),
                outputs=(bins, bins / "manifest.tsv"),
                fasta_output_dirs=(bins,),
                description=(
                    "Normalize MetaWRAP MAG names and retain unchanged "
                    "source-binner provenance"
                ),
            )
        )
        return publish, bins

    def _catalog(self, refined: list[tuple[Task, Path]]) -> tuple[Task, Path]:
        candidates = self.options.outdir / "quality_control_files" / "candidate_bins"
        args: list[str | Path] = ["collect-fasta"]
        for _task, directory in refined:
            args.extend(("--source", directory))
        args.extend(("--output-dir", candidates))
        task = self.add(
            Task(
                id="05.catalog.candidates",
                stage="05_quality",
                command=internal(*args),
                cwd=self.root,
                deps=tuple(task.id for task, _path in refined),
                inputs=tuple(path for _task, path in refined),
                outputs=(candidates, candidates / "manifest.tsv"),
                description="Collect refined candidate MAGs",
            )
        )
        return task, candidates

    def _quality(self, catalog: tuple[Task, Path]) -> tuple[Task, Path]:
        catalog_task, candidates = catalog
        root = self.options.outdir / "quality_control_files"
        task_environment: dict[str, str] = {}
        if self.options.quality_control == "checkm2":
            output = root / "checkm2"
            report = output / "quality_report.tsv"
            database = f" --database_path {q(self.options.checkm2_db)}" if self.options.checkm2_db else ""
            runner = " ".join(q(argument) for argument in self.options.checkm2_run_prefix)
            task_environment = CHECKM2_THREAD_ENV
            command = (
                f"rm -rf {q(output)} && {runner} checkm2 predict "
                f"--threads {self.options.threads} "
                f"--input {q(candidates)} --output-directory {q(output)} "
                f"-x {q(self.options.mag_suffix)}{database}"
                f"{self._extra('checkm2')}"
            )
        else:
            output = root / "checkm"
            report = output / "quality_report.tsv"
            work = self.options.workdir / "quality" / "checkm"
            command = (
                f"rm -rf {q(output)} {q(work)} && mkdir -p {q(output)} {q(work)} && "
                f"checkm lineage_wf -x {q(self.options.mag_suffix)} {q(candidates)} {q(work)} -t {self.options.threads} "
                f"--tab_table -f {q(report)}{self._extra('checkm')}"
            )
        task = self.add(
            Task(
                id=f"05.qc.{self.options.quality_control}",
                stage="05_quality",
                command=command,
                cwd=self.root,
                deps=(catalog_task.id,),
                inputs=(candidates,),
                outputs=(report,),
                cpus=self.options.threads,
                env=task_environment,
                description=f"Estimate MAG quality with {self.options.quality_control}",
            )
        )
        return task, report

    def _gunc(self, catalog: tuple[Task, Path]) -> Task | None:
        if not self.options.run_gunc:
            return None
        catalog_task, candidates = catalog
        output = self.options.outdir / "quality_control_files" / "GUNC"
        temp = self.options.workdir / "quality" / "gunc"
        database = f" --db_file {q(self.options.gunc_db)}" if self.options.gunc_db else ""
        return self.add(
            Task(
                id="05.qc.gunc",
                stage="05_quality",
                command=(
                    f"rm -rf {q(output)} {q(temp)} && mkdir -p {q(output)} {q(temp)} && gunc run "
                    f"--input_dir {q(candidates)} --file_suffix .{self.options.mag_suffix} --threads {self.options.threads} "
                    f"--out_dir {q(output)} --temp_dir {q(temp)} "
                    f"--min_mapped_genes 11{database}{self._extra('gunc')}"
                ),
                cwd=self.root,
                deps=(catalog_task.id,),
                inputs=(candidates,),
                outputs=(output,),
                cpus=self.options.threads,
                description="Detect chimeric MAGs with GUNC",
            )
        )

    def _rna(self, catalog: tuple[Task, Path]) -> tuple[Task | None, Path | None]:
        run_trna = self.options.run_trna or self.options.trna_pass is not None
        run_rrna = self.options.run_rrna or self.options.rrna_pass
        if not (run_trna or run_rrna):
            return None, None
        catalog_task, candidates = catalog
        marker_dir = self.options.workdir / "quality" / "gtdbtk_identify"
        data_env = {"GTDBTK_DATA_PATH": str(self.options.gtdbtk_data)} if self.options.gtdbtk_data else {}
        identify = self.add(
            Task(
                id="05.qc.rna_domain",
                stage="05_quality",
                command=(
                    f"rm -rf {q(marker_dir)} && gtdbtk identify --genome_dir {q(candidates)} "
                    f"--out_dir {q(marker_dir)} --extension {q(self.options.mag_suffix)} --cpus {self.options.threads}"
                ),
                cwd=self.root,
                deps=(catalog_task.id,),
                inputs=(candidates,),
                outputs=(marker_dir,),
                cpus=self.options.threads,
                env=data_env,
                description="Identify bacterial/archaeal marker domains before RNA QC",
            )
        )
        quality_root = self.options.outdir / "quality_control_files"
        output = quality_root / "rna"
        summary = quality_root / "rna_quality.tsv"
        completion_marker = output / "rna_qc.v2.complete"
        args: list[str | Path] = [
            "rna-qc",
            "--bins-dir",
            candidates,
            "--marker-dir",
            marker_dir,
            "--output-dir",
            output,
            "--summary",
            summary,
            "--threads",
            str(self.options.threads),
            "--completion-marker",
            completion_marker,
        ]
        if run_trna:
            args.append("--trna")
        if self.options.trna_pass is not None:
            args.extend(("--trna-pass", str(self.options.trna_pass)))
        if run_rrna:
            args.append("--rrna")
        if self.options.rrna_pass:
            args.append("--rrna-pass")
        task = self.add(
            Task(
                id="05.qc.rna",
                stage="05_quality",
                command=internal(*args),
                cwd=self.root,
                deps=(identify.id,),
                inputs=(candidates, marker_dir),
                outputs=(output, summary, completion_marker),
                cpus=self.options.threads,
                description="Apply domain-aware tRNA/rRNA quality checks",
            )
        )
        return task, summary

    def _filter(
        self,
        catalog: tuple[Task, Path],
        check_task: Task,
        check_report: Path,
        gunc_task: Task | None,
        rna_task: Task | None,
        rna_summary: Path | None,
    ) -> tuple[Task, Path]:
        catalog_task, candidates = catalog
        output = self.options.outdir / "quality_control_files"
        filtered = output / "filtered_bins"
        summary = output / "quality_summary.tsv"
        args: list[str | Path] = [
            "filter-quality",
            "--bins-dir",
            candidates,
            "--checkm2",
            check_report,
            "--output-dir",
            filtered,
            "--summary",
            summary,
            "--min-completeness",
            str(self.options.min_completeness),
            "--max-contamination",
            str(self.options.max_contamination),
        ]
        deps = [catalog_task.id, check_task.id]
        inputs = [candidates, check_report]
        if self.options.min_quality_score is not None:
            args.extend(("--min-quality-score", str(self.options.min_quality_score)))
        if gunc_task:
            gunc_dir = output / "GUNC"
            args.extend(("--gunc-dir", gunc_dir))
            deps.append(gunc_task.id)
            inputs.append(gunc_dir)
        if rna_task:
            deps.append(rna_task.id)
            if (
                rna_summary
                and (
                    self.options.trna_pass is not None
                    or self.options.rrna_pass
                )
            ):
                args.extend(("--rna-summary", rna_summary))
                inputs.append(rna_summary)
        task = self.add(
            Task(
                id="05.qc.filter",
                stage="05_quality",
                command=internal(*args),
                cwd=self.root,
                deps=tuple(deps),
                inputs=tuple(inputs),
                outputs=(filtered, summary),
                description="Apply completeness, contamination, score, GUNC and RNA filters",
            )
        )
        return task, filtered

    def _dereplicate(self, filtered_task: Task, filtered: Path, quality: Path) -> tuple[Task, Path]:
        final = self.options.outdir / (
            "non_redundant_bins.untagged" if self.options.tag_contigs else "non_redundant_bins"
        )
        if self.options.dereplicator == "galah":
            clusters = self.options.outdir / "galah_clusters.tsv"
            quality_arg = (
                f"--checkm2-quality-report {q(quality)} "
                if self.options.quality_control == "checkm2"
                else f"--checkm-tab-table {q(quality)} "
            )
            task = self.add(
                Task(
                    id="06.dereplicate.galah",
                    stage="06_dereplication",
                    command=(
                        f"rm -rf {q(final)} && galah cluster --genome-fasta-directory {q(filtered)} "
                        f"--genome-fasta-extension {q(self.options.mag_suffix)} {quality_arg}"
                        f"--min-completeness {self.options.min_completeness:g} "
                        f"--max-contamination {self.options.max_contamination:g} --ani {self.options.ani:g} "
                        f"--min-aligned-fraction {self.options.min_aligned_fraction:g} --quality-formula completeness-5contamination "
                        f"--precluster-method skani --cluster-method skani "
                        f"--output-cluster-definition {q(clusters)} "
                        f"--output-representative-fasta-directory-copy {q(final)} "
                        f"--threads {self.options.threads}{self._extra('galah')}"
                    ),
                    cwd=self.root,
                    deps=(filtered_task.id,),
                    inputs=(filtered, quality),
                    outputs=(final, clusters),
                    cpus=self.options.threads,
                    description="Dereplicate MAGs with Galah",
                )
            )
            return task, final
        drep = self.options.workdir / "drep"
        genomes = f"{q(filtered)}/*.{self.options.mag_suffix}"
        task = self.add(
            Task(
                id="06.dereplicate.drep",
                stage="06_dereplication",
                command=(
                    f"rm -rf {q(drep)} {q(final)} && dRep dereplicate {q(drep)} -g {genomes} "
                    f"-p {self.options.threads} -comp {self.options.min_completeness:g} "
                    f"-con {self.options.max_contamination:g} -sa {self.options.ani / 100:g} "
                    f"-nc {self.options.min_aligned_fraction / 100:g}"
                    f"{self._extra('drep')} && mkdir -p {q(final)} && "
                    f"cp {q(drep / 'dereplicated_genomes')}/*.{self.options.mag_suffix} {q(final)}/"
                ),
                cwd=self.root,
                deps=(filtered_task.id,),
                inputs=(filtered,),
                outputs=(final,),
                cpus=self.options.threads,
                description="Dereplicate MAGs with dRep",
            )
        )
        return task, final

    def _extra(self, tool: str) -> str:
        value = self.options.extra_args.get(tool, "").strip()
        return f" {value}" if value else ""

    def _gpu_env(self) -> dict[str, str]:
        if not self.options.gpu:
            return {}
        return {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
            "METABAW_MAX_GPU_MEMORY": self.options.max_gpu_memory,
        }


@dataclass(frozen=True)
class AnnotationOptions:
    mag_dir: Path
    mag_suffix: str
    reads: tuple[ReadSample, ...]
    output: Path
    output_suffix: str
    threads: int
    read_type: str
    methods: tuple[str, ...]
    place_species: bool
    niche_rank: str
    no_niche: bool
    gtdbtk_data: Path | None


class AnnotationBuilder:
    def __init__(self, options: AnnotationOptions, root: Path):
        self.options = options
        self.root = root
        self.tasks: list[Task] = []

    def add(self, task: Task) -> Task:
        self.tasks.append(task)
        return task

    def build(self) -> list[Task]:
        taxonomy = self.options.output / "gtdbtk_result"
        place = " --place_species" if self.options.place_species else ""
        env = {"GTDBTK_DATA_PATH": str(self.options.gtdbtk_data)} if self.options.gtdbtk_data else {}
        taxonomy_task = self.add(
            Task(
                id="01.annotation.gtdbtk",
                stage="01_taxonomy",
                command=(
                    f"rm -rf {q(taxonomy)} && gtdbtk classify_wf --genome_dir {q(self.options.mag_dir)} "
                    f"--out_dir {q(taxonomy)} --extension {q(self.options.mag_suffix.lstrip('.'))} "
                    f"--cpus {self.options.threads}{place}"
                ),
                cwd=self.root,
                inputs=(self.options.mag_dir,),
                outputs=(taxonomy,),
                cpus=self.options.threads,
                env=env,
                description="Classify MAGs with GTDB-Tk",
            )
        )
        coverm_dir = self.options.output / "coverm"
        profile_tasks: list[Task] = []
        profiles: list[Path] = []
        for sample in self.options.reads:
            output = coverm_dir / f"{sample.name}{self.options.output_suffix}"
            reads = (
                f"--coupled {q(sample.read1)} {q(sample.read2)}"
                if sample.read2
                else f"--single {q(sample.read1)}"
            )
            mapper = "minimap2-sr" if self.options.read_type == "short" else "minimap2-ont"
            profile_tasks.append(
                self.add(
                    Task(
                        id=f"02.annotation.coverm.{sample.name}",
                        stage="02_abundance",
                        command=(
                            f"mkdir -p {q(coverm_dir)} && coverm genome "
                            f"--genome-fasta-directory {q(self.options.mag_dir)} "
                            f"--genome-fasta-extension {q(self.options.mag_suffix.lstrip('.'))} {reads} "
                            f"--mapper {mapper} --methods {' '.join(self.options.methods)} "
                            f"--min-read-percent-identity 95 --min-read-aligned-percent 75 "
                            f"--min-covered-fraction 10 --threads {self.options.threads} "
                            f"--output-file {q(output)}"
                        ),
                        cwd=self.root,
                        inputs=(self.options.mag_dir, *sample.reads),
                        outputs=(output,),
                        cpus=self.options.threads,
                        description=f"Profile {sample.name} with CoverM",
                    )
                )
            )
            profiles.append(output)
        merge_args: list[str | Path] = ["merge-coverm-taxonomy"]
        for sample, profile in zip(self.options.reads, profiles, strict=True):
            merge_args.extend(("--input", f"{sample.name}={profile}"))
        merge_args.extend(
            (
                "--taxonomy-dir",
                taxonomy,
                "--output-dir",
                coverm_dir,
                "--output-suffix",
                self.options.output_suffix,
            )
        )
        merge_task = self.add(
            Task(
                id="03.annotation.merge",
                stage="03_merge",
                command=internal(*merge_args),
                cwd=self.root,
                deps=(taxonomy_task.id, *(task.id for task in profile_tasks)),
                inputs=(taxonomy, *profiles),
                outputs=(coverm_dir / "manifest.json",),
                description="Merge CoverM metrics with bacterial and archaeal GTDB taxonomy",
            )
        )
        if (
            not self.options.no_niche
            and len(self.options.reads) >= 2
            and {"relative_abundance", "count"} <= set(self.options.methods)
        ):
            combined = coverm_dir / f"coverm_all_metrics{self.options.output_suffix}"
            niche_dir = self.options.output / "niche"
            args: list[str | Path] = [
                "classify-niche-literature",
                "--abundance",
                combined,
                "--taxonomy-dir",
                taxonomy,
                "--output",
                niche_dir / "niche_classification.tsv",
                "--aggregated-output",
                niche_dir / "taxon_abundance.tsv",
                "--assignments-output",
                niche_dir / "genome_to_taxon.tsv",
                "--provenance-output",
                niche_dir / "provenance.json",
                "--rank",
                self.options.niche_rank,
                "--detection-percent",
                "0.01",
                "--min-total-reads",
                "20",
                "--min-prevalence",
                "0.20",
                "--core-prevalence",
                "0.80",
                "--min-core-datasets",
                "1",
            ]
            for sample in self.options.reads:
                args.extend(("--sample-dataset", f"{sample.name}=default"))
            self.add(
                Task(
                    id="04.annotation.niche",
                    stage="04_niche",
                    command=internal(*args),
                    cwd=self.root,
                    deps=(merge_task.id,),
                    inputs=(combined, taxonomy),
                    outputs=(niche_dir / "niche_classification.tsv", niche_dir / "provenance.json"),
                    description=f"Classify ecological niches at {self.options.niche_rank} level",
                )
            )
        return topological_order(self.tasks)
