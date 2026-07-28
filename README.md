# MetaBAW

MetaBAW: Metagenome Binning Automated Workflow.

MetaBAW does not depend on Nextflow, Snakemake, or another workflow engine.
Python manages the task graph, CPU-aware concurrency, logs, retries, SQLite
state, caching, and resumable execution.

The command line has three modules:

- `metabaw bin`: read mapping, binning, refinement, quality control, and
  dereplication.
- `metabaw annotation`: GTDB-Tk taxonomy, CoverM abundance profiling,
  taxonomy-aware matrix generation, and ecological niche classification.
- `metabaw check`: check selected software, ask before installing missing
  software with Mamba/Micromamba/Conda, and report database status.

## Installation

Production runs require Linux, WSL2, or a Linux HPC node and Python 3.11 or
newer.

```bash
cd metaBAW
mamba create -n metabaw-py311 python=3.11 pip
conda activate metabaw-py311
python -m pip install -U pip
python -m pip install .

metabaw --version
metabaw -v
metabaw -h
```

If both `(base)` and `(.venv)` appear in the shell prompt, the Conda base
environment and the Python virtual environment are active at the same time.
This is not an installation error, but it can make executable resolution
ambiguous. Deactivate Conda first or use one dedicated Conda environment.

The Python package has no third-party runtime dependencies. External programs
are required only when their corresponding workflow options are selected.
MetaBAW and compatible tools remain in the main Python 3.11 environment.
Programs with incompatible Python requirements run through isolated Conda
environments:

| Program | Python | Default isolated environment | Package |
|---|---:|---|---|
| COMEBin | 3.7 | `metabaw-comebin-py37` | `comebin` |
| CheckM2 | 3.12 | `metabaw-checkm2-py312` | `checkm2` |
| MetaWRAP | 2.7 | `metabaw-metawrap-py27` | `metawrap-refinement` |
| LorBin | 3.10 | `metabaw-lorbin-py310` | official pinned source |

VAMB, SemiBin2, MetaDecoder, CheckM, GUNC, GTDB-Tk, and dRep are compatible
with the main Python 3.11 environment. MetaDecoder is installed from its
official wheel because it currently has no Bioconda recipe.

LorBin is installed from the official source commit
`ee10232282c2b71ed3ce2a34d5dbd78af3dd0b0a`, which fixes integer parsing for
its thread option. Its released Python 3.10 and PyTorch 1.11 dependency stack
is kept outside the main MetaBAW environment. MetaBAW uses Biopython 1.83
because the upstream Biopython 1.78 pin has no Conda build for Python 3.10;
the SeqIO interfaces used by LorBin remain compatible. LorBin retains its upstream
80 kb output threshold, and MetaBAW applies `--minfasta-kbs` while normalizing
LorBin bins so the shared minimum size remains effective.

| Stage | Default programs | Optional programs |
|---|---|---|
| Shell and installation support | Bash, Git | |
| Mapping | Bowtie2, Minimap2, SAMtools | minibwa |
| Abundance preparation | jgi_summarize_bam_contig_depths, strobealign | |
| Binning | Short: MetaBAT2, MetaDecoder, VAMB; long: MetaDecoder, VAMB, LorBin | COMEBin, SemiBin2 |
| Refinement | MAGScoT, GNU Parallel, Prodigal, HMMER, R | DAS Tool, DIAMOND, MetaWRAP |
| Quality control | CheckM2 | CheckM, GUNC, tRNAscan-SE, Barrnap, GTDB-Tk |
| Dereplication | Galah | dRep |
| Annotation | GTDB-Tk, CoverM, Minimap2 | |

MAGScoT also requires the R packages `optparse`, `dplyr`, `readr`, `funr`,
and `digest`. The default MAGScoT source location is
`~/.cache/metabaw/MAGScoT`. Configure and save another location with
`metabaw check --magscot-dir PATH`; `metabaw bin` reads the saved value.
DAS Tool requires the R packages `data.table`, `magrittr`, and `docopt`;
MetaBAW's DAS Tool invocation also checks Prodigal, pullseq, Ruby, and
DIAMOND.

R packages are not forced through Mamba. MetaBAW uses Mamba/Conda only for
the R runtime and external programs, installs missing CRAN packages with
`Rscript install.packages()`, and installs Bioconductor packages with
`BiocManager::install()`. If a future selected dependency needs a
Bioconductor package and `BiocManager` is absent, MetaBAW installs
`BiocManager` from CRAN first.

For Barrnap, MetaBAW performs an execution probe rather than accepting a
successful `which barrnap` lookup. This detects broken Perl environments such
as a missing `Path::Tiny`; interactive installation repairs the active
environment with Barrnap and `perl-path-tiny`.

Before a real `bin` or `annotation` run, MetaBAW checks only the software and
databases required by the selected options. In an interactive terminal it
asks whether missing items should be installed. In a non-interactive batch
job it stops with a complete missing-dependency message instead of starting a
long workflow that cannot finish. `--dry-run` does not require installed
external tools.

`metabaw check` uses essential mode by default. It checks the programs used
by the default short-read `bin` workflow and the default `annotation`
workflow, then asks whether any missing software should be installed.
It also reports NVIDIA driver visibility. For GPU-capable tools in the
selected scope, MetaBAW starts that tool's Python environment, calls
`torch.cuda.is_available()`, enumerates the visible devices, and performs a
small CUDA tensor-allocation test. This distinguishes a visible GPU from a
CPU-only or incompatible PyTorch installation.
`-e` and `--essential` explicitly select the same mode:

```bash
metabaw check
metabaw check -e
```

To check every supported external program and all isolated environments, use
`--all`. MetaBAW still asks for confirmation before installing missing items:

```bash
metabaw check --all \
  --comebin-env /data1/zhoux/metabaw_envs/comebin-py37 \
  --checkm2-env /data1/zhoux/metabaw_envs/checkm2-py312 \
  --metawrap-env /data1/zhoux/metabaw_envs/metawrap-py27 \
  --lorbin-env /data1/zhoux/metabaw_envs/lorbin-py310
```

A Conda environment name can be used instead of an absolute prefix. MetaBAW
verifies the required Python version and executable before saving each value
to `~/.config/metabaw/config.json`. Later `check` and `bin` commands read this
configuration automatically, so the environment arguments do not need to be
repeated:

```bash
metabaw bin -p reads -c contigs \
  --tools metabat2 metadecoder vamb comebin semibin2 \
  --quality-control checkm2 \
  -o result
```

For `metabaw check`, an explicit environment option selects the value to
validate and save. Later `metabaw bin` resolves `METABAW_COMEBIN_ENV`,
`METABAW_CHECKM2_ENV`, `METABAW_METAWRAP_ENV`, or `METABAW_LORBIN_ENV`
first, then saved configuration, and finally the built-in default. Set
`METABAW_CONFIG_FILE` to use a different configuration file. A single
environment can be changed and saved again:

```bash
metabaw check --scope comebin \
  --comebin-env /new/path/comebin-py37
```

CUDA is optional during a normal check. Add `--require-cuda` when a failed
driver or PyTorch allocation test must make the check return an error. In an
interactive terminal, MetaBAW first asks whether it should install a
CUDA-enabled PyTorch build and then repeats the allocation test. COMEBin uses
its official Python 3.7, PyTorch 1.10.2, and CUDA 11.1 profile; LorBin uses
its official PyTorch 1.11.0 and CUDA 11.3 profile:

```bash
metabaw check --scope comebin --require-cuda
metabaw check --all --require-cuda
```

To include database downloads:

```bash
metabaw check -e --install-databases \
  --database-dir ~/.cache/metabaw/databases
```

GTDB-Tk reference data requires approximately 100 GB. Database locations
installed by MetaBAW are recorded under
`~/.config/metabaw/databases.json`.

Database path precedence is an explicit command-line option, a valid
environment variable, and then the path saved by `metabaw check`. If an old
`CHECKM2DB`, `GUNC_DB`, or `GTDBTK_DATA_PATH` value no longer exists, MetaBAW
prints a warning and automatically uses a valid saved path instead. An
explicit command-line path is always authoritative, including when it is
invalid, so typing mistakes are reported directly.

## Binning

### Automatic strategy selection

With `--auto`, the only required workflow inputs are `-p`, `-s`, `-c`, and
`-f`. The output directory remains optional and defaults to
`metabaw_result`; every other strategy and resource option is selected
automatically.

```bash
metabaw bin \
  -p reads/ -s fastq.gz \
  -c assemblies/ -f fa \
  --auto
```

Before dependency checks and task construction, MetaBAW scans the current
server rather than relying only on installed hardware totals. It records CPU
count and affinity, one-minute CPU load, estimated available CPU capacity,
total and currently available RAM, free space on the output filesystem,
NVIDIA device models, total and currently free VRAM, driver version, and the
CUDA version advertised by `nvidia-smi`. Available CPU, RAM, sample count,
and free VRAM are used to select `--threads`, `--task`, `--max-memory`,
`--gpu`, `--max-gpu-memory`, COMEBin batch size, and retry behavior. The
per-task GPU budget never exceeds 4 GiB. Automatic per-task RAM is not capped
at the manual 100 GiB default: MetaBAW reserves 15% of currently available
RAM and divides the remainder between concurrent samples.

`--auto` also profiles every unique assembly. It records contig count,
assembly size, N50, usable contig counts, sample count, pairing fraction,
and whether samples share one assembly. The deterministic policy uses that
evidence to select single- or multi-sample binning, a cohort group size, a
complementary binner set, and a 1,500 or 2,000 bp minimum contig length.

Three or more samples normally use differential-coverage multi-sample
binning. Per-sample assembly cohorts larger than 20 are split into
deterministic groups of at most 20 samples. Short-read cohorts use MetaBAT2,
MetaDecoder, VAMB, and,
when cohort size or assembly complexity justifies it, SemiBin2. COMEBin is
added when a usable NVIDIA GPU with sufficient currently free memory is
detected. Long-read data use MetaDecoder, VAMB, LorBin, and optionally
SemiBin2.

MetaBAW writes `<output>/metabaw_auto_run.sh` before execution, including
the complete server-resource snapshot, every resolved parameter, and a
fully explicit reproducible `metabaw bin` command. The command intentionally
omits `--auto`, `--dry-run`, and `--force`, so rerunning the script uses the
recorded decisions without rescanning or bypassing cached results. The same
information is stored under `auto_strategy` in `run_manifest.json`.
The script separately labels automatically resolved effective values and
fixed workflow policy values. For GPU COMEBin runs it records both the
requested batch size and the memory-capped effective batch size.

Advanced users may still provide resource or strategy options together with
`--auto`. Automatic selection only fills options that the user did not
explicitly provide. Explicit `--threads`, `--task`, `--max-memory`,
`--gpu`/`--no-gpu`,
`--max-gpu-memory`, `--batch-size`, `--retries`, `--single`/`--multi`,
`--cohort-size`, `--tools`, and `--min-contig-length` values override only
their corresponding automatic decisions, even when the supplied value is
the normal non-auto default. The generated script lists these values under
`User-provided values that overrode automatic decisions`. A COMEBin batch
size may still be reduced by the explicit `--max-gpu-memory` safety ceiling;
both the requested and effective values are recorded.

### Per-sample binning

Files named with `_R1/_R2`, `_1/_2`, or `.1/.2` are paired automatically.
Other names are split at the first `--separate-sample-name` separator
(default `.`): the part before the separator becomes the sample name, a
trailing `1`/`2` (or `R1`/`R2`, also after processing labels such as
`clean_R1`) marks the read mate, and any other trailing content is treated
as a processing label. For example, `B425.1.fastq.gz` and
`B425.2.fastq.gz` form sample `B425`, and every result is named with the
`B425` prefix.

```bash
metabaw bin \
  -p reads/ -s fastq.gz \
  -c assemblies/ -f fa \
  --single \
  --tools metabat2 metadecoder vamb \
  --refinement magscot \
  --quality-control checkm2 \
  --dereplication-tool galah \
  -t 32 --max-memory 100 \
  -o metabaw_result
```

When `--contig` points to one file, all samples use that assembly. When it
points to a directory, assemblies are matched to normalized sample names.
Common trailing processing labels are removed before matching. For example,
reads named `A606.clean_R1.fastq.gz` and `A606.clean_R2.fastq.gz` match
`A606.contig.ok.fa` because both names normalize to the sample key `A606`.
The original input files are not renamed. Generated task IDs, directories,
BAM files, logs, bin directories, and final MAG names all use `A606` and
never `A606.clean`. The prepared assembly retains the original file as its
task input and is written to the temporary workspace before mapping so the
shared `--min-contig-length` filter is applied consistently.

### Multi-sample binning

```bash
metabaw bin \
  -p reads/ -c assemblies/ \
  --multi \
  --tools metabat2 metadecoder vamb comebin semibin2 \
  --environment human_gut \
  -t 32 -o metabaw_result
```

With per-sample assemblies, `--multi` uses cross mapping: every sample's
reads are mapped to every assembly, and each assembly is binned separately
with the full multi-sample coverage matrix. BAM files are named after both
partners: for example, `A_to_B.bam` holds sample A's reads mapped to sample
B's assembly. Bins keep the owning sample's name, for example
`A_MetaBAT2_1.fa`. Only the multi-sample analyses run; redundant per-sample
analyses are not repeated.

For two samples, MetaBAW therefore builds four alignments. BAM-based tools
bin sample A's assembly only from `A_to_A.bam` and `B_to_A.bam`, and sample
B's assembly only from `A_to_B.bam` and `B_to_B.bam`. BAM files aligned to
different assemblies are never combined in one per-assembly binner
invocation. Across the two independent binning runs, all four BAM files are
used and the final A and B bin sets remain separate. Short-read VAMB computes
the equivalent target-specific abundance matrix directly with strobealign
`--aemb`: A's reads and B's reads are each profiled against A's assembly for
A's VAMB run, and separately against B's assembly for B's VAMB run.

SemiBin2 follows its native multi-sample workflow. MetaBAW concatenates the
per-sample assemblies with `sample:contig` identifiers, maps every sample's
reads separately to that concatenated assembly, and runs one
`SemiBin2 multi_easy_bin` command with all resulting BAM files. The
sample-specific results under `samples/<sample>/output_bins` are then routed
back into each sample's refinement workflow.

This concatenation is only SemiBin2 input preparation. MetaBAW does not pool
reads or run an assembler, so it is not co-assembly. The temporary files use
the explicit internal name `semibin2_multisample_input` and remain under
`<output>/tmp/work`; they never become a sample or final MAG prefix.
When a large run is split by `--cohort-size`, each cohort receives its own
numbered input and `multi_easy_bin` task. Contigs and BAM files are never
mixed across cohorts.

Cross mapping costs one alignment per sample-assembly pair (N x N alignments
for N samples), so runtime grows quickly with sample count.

The current workflow deliberately requires one independently assembled
contig file per sample. If several samples resolve to the same contig file,
MetaBAW stops with a clear error because shared co-assembly input is not
enabled yet.

The `--environment` option selects one SemiBin2 pretrained environment model:

```text
human_gut          dog_gut             ocean
soil               cat_gut             human_oral
mouse_gut          pig_gut             built_environment
wastewater         chicken_caecum      global
```

The default is not set, which leaves environment handling to SemiBin2.
Pretrained `--environment` models apply only to single-sample SemiBin2 runs.
Multi-sample runs use self-supervised training because their coverage
feature dimensions depend on the selected samples.

### Explicit multi-sample groups

`--multi-files` accepts a two-column, tab-separated format:

```text
# READ1[,READ2] <TAB> CONTIGS
/data/A_R1.fastq.gz,/data/A_R2.fastq.gz	/data/A.fa
/data/B_R1.fastq.gz,/data/B_R2.fastq.gz	/data/B.fa
```

An explicit sample-name column is also accepted:

```text
# SAMPLE <TAB> READ1[,READ2] <TAB> CONTIGS
A	/data/A_R1.fastq.gz,/data/A_R2.fastq.gz	/data/A.fa
B	/data/B_R1.fastq.gz,/data/B_R2.fastq.gz	/data/B.fa
```

```bash
metabaw bin -p reads/ -c assemblies/ --multi \
  --multi-files selected.tsv -t 32 -o metabaw_result
```

Samples listed in the file are processed together and cross-mapped using
their per-sample assemblies. Reusing one contig file for several samples is
rejected until co-assembly input support is added. Remaining samples found
under `--path` are processed separately in single-sample mode.

### Long reads

```bash
metabaw bin \
  -p long_reads/ -s fastq.gz --type long \
  --align-tool minimap2 --long-read-preset map-ont \
  -c assemblies/ -f fa \
  -t 32 -o metabaw_result
```

Minimap2 is the default long-read mapper. minibwa is available for accurate
long reads. If `--tools` is omitted, short reads use MetaBAT2, MetaDecoder,
and VAMB, whereas long reads replace MetaBAT2 with LorBin and use MetaDecoder,
VAMB, and LorBin. Explicit `--tools` selections are not rewritten. LorBin is
accepted only with `--type long`.

### Quality and RNA filters

```bash
metabaw bin \
  -p reads/ -c assemblies/ \
  --con 50 --com 10 --quality-score 50 \
  --gunc --trna-pass 18 --rrna-pass \
  -t 32 -o metabaw_result
```

Configure CheckM2/GUNC databases beforehand with `metabaw check --all
--checkm2-db PATH --gunc-db PATH`. The bin command reads the saved paths.

- `--quality-score` uses `completeness - 5 * contamination`.
- `--trna` and `--rrna` only predict and report RNA features. They never
  remove MAGs from the result.
- `--trna-pass N` enables a hard filter requiring at least `N` distinct tRNA
  types. It also enables tRNA prediction; there is no implicit default
  threshold.
- `--rrna-pass` enables a hard filter requiring 5S, 16S, and 23S rRNAs. It
  also enables rRNA prediction.
- RNA quality control uses GTDB-Tk marker summaries to select bacterial or
  archaeal tRNAscan-SE and Barrnap modes. The bin module resolves the GTDB-Tk
  database from `GTDBTK_DATA_PATH` or the path saved by `metabaw check`.
- Barrnap writes GFF directly; MetaBAW does not request the optional rRNA
  FASTA export. Standard error is stored separately from GFF output. A tool
  failure for one MAG is recorded in the `error` column of `rna_quality.tsv`
  and marks that MAG as failed without stopping RNA checks for other MAGs.
- RNA QC parallelizes across MAGs while keeping the total requested tool
  threads within `-t/--threads`. For example, 44 MAGs with `-t 32` use up to
  32 concurrent MAG workers with one tool thread each. Four MAGs with
  `-t 32` use four workers with eight tool threads each.
- `rna/rna_qc.v2.complete` is written only after every candidate MAG has been
  processed. An interrupted or partial RNA QC directory is therefore rerun
  during resume instead of being mistaken for a complete result.
- `--tag-contigs` renames final contigs to `bin_name_1`, `bin_name_2`, and so
  on.

### GPU options

`--gpu` enables GPU options for supported VAMB, COMEBin, SemiBin2, or LorBin
configurations. Before task execution, MetaBAW checks `nvidia-smi` and
performs a real CUDA tensor allocation inside every selected Python
environment. A driver-visible GPU is therefore not accepted when COMEBin,
SemiBin2, VAMB, or LorBin has a CPU-only or incompatible PyTorch build.
When the command has an interactive terminal, it asks whether the failed
PyTorch runtime should be repaired and retests it before starting any
workflow task. Non-interactive jobs never modify an environment
automatically; run the corresponding
`metabaw check --scope ... --require-cuda` interactively first.

`--max-gpu-memory` defaults to `4G`. MetaBAW uses this per-task budget to cap
COMEBin's requested batch size conservatively: with the default 4G budget,
the default requested batch size of 1024 becomes an effective batch size of
256. Both values are printed and stored in `run_manifest.json`. This reduces
out-of-memory risk but is not a universal hard GPU-memory limit because the
external tools do not expose one shared limiter.

GPU tasks declare one memory-budgeted GPU task slot. MetaBAW reserves 15% of
the currently free memory on each visible GPU for the driver, CUDA context,
and framework overhead, then derives concurrent slots from
`--max-gpu-memory`. For example, one otherwise idle 48 GB GPU can run three
sample tasks concurrently with `--task 3 --max-gpu-memory 4G`. All processes
may share the selected device; the
external tools remain responsible for respecting their effective memory
budget. CPU-only steps can continue in parallel. Use
`CUDA_VISIBLE_DEVICES` to select devices.

### Important options

```text
--auto                  Select strategy and resources from assembly/sample
                        evidence and current free server resources; write
                        <output>/metabaw_auto_run.sh; shown by -h/--help;
                        default disabled
--tools                 Any combination of metabat2, vamb, metadecoder,
                        comebin, semibin2, and lorbin
--cohort-size           Maximum per-sample assembly cohort size; resolved and
                        recorded automatically by --auto
--separate-sample-name  Custom sample-name delimiter for read discovery
--min-contig-length     Shared minimum contig length; default 1500 bp
--minfasta-kbs          Minimum bin size; default 200 kb
--batch-size            Requested COMEBin batch size; default 1024; GPU mode
                        may lower it to fit --max-gpu-memory
--refinement            magscot, das_tool, or metawrap
--quality-control       checkm2 or checkm
--dereplication-tool    galah or drep
--ani                   Dereplication ANI threshold in percent; default 99
--min-aligned-fraction  Minimum aligned fraction in percent; default 30
-x, --mag-suffix        MAG file extension propagated to CheckM2/CheckM,
                        GUNC, GTDB-Tk, and dereplication; default fa
--task                  Maximum concurrent samples; default 1
--max-memory            Per-task virtual-memory ceiling in GiB; default 100
--tmp-files             Temporary directory; default <output>/tmp
--delete-tmp-files      Delete temporary files after a successful run
--dry-run               Print the complete task and command plan
--force                 Ignore successful cached tasks and rerun
--advanced-arg TOOL=... Append expert-only arguments to an external tool
```

MetaWRAP refinement accepts bin sets from no more than three binning tools.
If `--refinement metawrap` is combined with more than three values in
`--tools`, MetaBAW reports the selected tool names and exits before input
discovery, dependency checks, or workflow execution. Use MAGScoT or DAS Tool
when refining more than three bin sets.

Software locations and database paths are intentionally absent from
`metabaw bin`. Configure MAGScoT, CheckM2, GUNC, COMEBin, LorBin, MetaWRAP,
and the isolated CheckM2 environment through `metabaw check`; bin runs read
the saved configuration automatically.

Run `metabaw bin --full-help` for the complete interface.

## Annotation

```bash
metabaw annotation \
  -p metabaw_result/non_redundant_bins -s fa \
  -r reads/ -f fastq.gz \
  --methods relative_abundance rpkm tpm mean count \
  --niche-rank family \
  --gtdbtk-db /db/gtdb/release226 \
  -t 32 --max-memory 100 \
  -o metabaw_annotation_result
```

GTDB-Tk species placement is enabled by default. Use `--no-place-species` to
disable it.

Supported CoverM metrics are:

```text
relative_abundance  rpkm  tpm  mean  count  trimmed_mean
covered_fraction    covered_bases  reads_per_base  variance  length
```

metaBAW writes one matrix per metric. GTDB-Tk domain, phylum, class, order,
family, genus, and species columns are inserted after `Genome`.

Supported niche levels are:

```text
domain phylum class order family genus species strain
```

The default niche level is `family`. Niche classification requires at least
two samples and both `relative_abundance` and `count`. If these conditions are
not met, taxonomy and abundance processing still complete. Use `--no-niche`
to disable niche classification explicitly.

## Binning output

Published bins use the stable `sample_tool_sequence.fa` naming convention.
Trailing processing suffixes `.clean`, `_clean`, `-clean`, and the equivalent
`cleaned` forms are removed from sample and analysis labels before the task
graph is built. Individual binner examples are
`A606_MetaBAT2_1.fa`, `A606_MetaDecoder_2.fa`, and `A606_VAMB_3.fa`.
Refined bins are named by how much the refinement changed them. With MAGScoT,
a refined bin whose contig set is identical to a bin already published by an
individual binner keeps that bin's original name, such as
`A606_MetaBAT2_1.fa`. Only bins created or modified by MAGScoT receive the
refinement label, numbered as `A606_MAGScoT_1.fa`, `A606_MAGScoT_2.fa`, and so
on. DAS Tool follows the same provenance rule: an unchanged selected MAG
keeps a name such as `A606_MetaBAT2_1.fa`, while a MAG assembled or modified
by DAS Tool is named `A606_DASTool_1.fa`. DAS Tool's internal label prefix,
such as `metabat2__`, is removed during normalization. The normalized FASTA
set under `<tmp-files>/work/refinement/<sample>/*_DASTool_bins` is the exact
set collected into `quality_control_files/candidate_bins`. MetaWRAP also uses
this provenance rule inside `metawrap_50_10_bins`: unchanged MAGs retain
their original source-tool names, while only merged or modified MAGs receive
names such as `A606_MetaWRAP_1.fa`. The normalized MetaWRAP directory and
`quality_control_files/candidate_bins` contain the same FASTA names and
content. Internal source identifiers such as `cleanbin_000030` remain
available in `manifest.tsv` but never appear in published FASTA filenames.

```text
metabaw_result/
|-- bin_files/
|   |-- metabat2/
|   |-- vamb/
|   |-- metadecoder/
|   |-- comebin/
|   |-- semibin2/
|   `-- lorbin/
|-- non_redundant_bins/
|-- quality_control_files/
|   |-- checkm2/ or checkm/
|   |-- GUNC/
|   |-- rna/
|   |   |-- tRNA/
|   |   |-- rRNA/
|   |   `-- rna_qc.v2.complete
|   |-- quality_summary.tsv
|   `-- rna_quality.tsv
|-- galah_clusters.tsv
|-- tmp/
|   |-- work/
|   |-- system_tmp/
|   `-- runtime/
`-- run_manifest.json
```

Directories for optional tools are created only when those tools are enabled.

## Annotation output

```text
metabaw_annotation_result/
|-- gtdbtk_result/
|-- coverm/
|   |-- sampleA.tsv
|   |-- sampleB.tsv
|   |-- coverm_counts.tsv
|   |-- coverm_rel_abd.tsv
|   |-- coverm_rpkm_abd.tsv
|   |-- coverm_tpm_abd.tsv
|   |-- coverm_mean_abd.tsv
|   |-- coverm_all_metrics.tsv
|   `-- manifest.json
|-- niche/
|   |-- niche_classification.tsv
|   |-- taxon_abundance.tsv
|   |-- genome_to_taxon.tsv
|   `-- provenance.json
|-- tmp/
|   |-- work/
|   |-- system_tmp/
|   `-- runtime/
`-- run_manifest.json
```

All temporary and intermediate files are retained in `<output>/tmp` by
default:

```text
<output>/tmp/
|-- work/
|   `-- tool and stage intermediate files
|-- system_tmp/
`-- runtime/
    |-- state.sqlite3
    `-- logs/
```

A relative custom name such as `--tmp-files scratch` is created as
`<output>/scratch`; an absolute path is used directly. The directory is not
hidden. It is retained unless `--delete-tmp-files` is set. Deletion occurs
only after a successful workflow; failed runs retain temporary files and
logs for diagnosis.

External tasks run with `TMPDIR`, `TMP`, and `TEMP` set to
`<tmp-files>/system_tmp`, and their working directory is inside
`<tmp-files>/work`. Before execution, MetaBAW scans work outputs in the fixed
order `contigs`, `mapping`, `binning`, and `refinement`. Every declared output
must exist, every required alternative output set must be complete, and an
output directory must contain at least one file. A task with complete outputs
is skipped; a task with any missing output is rerun without forcing unrelated
complete tasks to run again.

Complete work outputs can be adopted even if
`<tmp-files>/runtime/state.sqlite3` is missing, which supports recovery after
copying an existing result directory or upgrading MetaBAW. When a state record
is present, its command, input, parameter, and dependency fingerprint must
still match. Keep the state database whenever possible because stages after
refinement use it for strict cache validation. The output and temporary paths
must remain the same, and `--force` must not be used when resuming.

A failed run can therefore be restarted with the same command. For example,
if refined MAGs were successfully published before CheckM2 failed, the next
run starts at CheckM2. An interrupted CheckM2 invocation restarts CheckM2
itself because its partial output directory is cleared before retrying. State
files produced by earlier 0.1.0 builds with precomputed intermediate-input
fingerprints are also recognized.

Bowtie2 cache validation checks all six `.bt2` or `.bt2l` index
components; a marker file alone is not sufficient. Index files are written
under `mapping/<sample>/index/` with the normalized filtered assembly name as
their prefix, for example `index/A606.1.bt2`; Bowtie2 receives the matching
`-x index/A606` prefix.

`mapping/<sample>/bam/` contains the BAM and BAI files produced by mapping.
`mapping/<sample>/bamset/` is a standardized directory view required by
tools such as COMEBin and long-read VAMB. MetaBAW creates hard links when the
directories are on the same filesystem, so the two paths share one inode and
do not consume the BAM size twice. A link count of `2` in `ls -l` confirms
this behavior.

`--task` is available in both modules and defaults to `1`. It controls the
maximum number of distinct samples processed concurrently. Tasks belonging
to one sample are serialized within that sample slot, so helper tasks from
sample A cannot consume the slots intended for samples B and C. Therefore
`-t 16 --task 3` starts work for as many as three ready samples and may use
up to 48 CPU threads. Cohort-wide tasks run after their declared sample
dependencies. The former `--max-parallel` spelling remains a hidden
compatibility alias.

`--max-memory` is available in both modules. Its unit is GiB and its default
is `100`, which means 100 GiB. On Linux, metaBAW applies the value as a
per-task virtual address-space ceiling and passes the same limit to internal
Python tasks. With `--task` greater than one, the value remains a per-task
ceiling rather than a combined cgroup limit.

## Runtime progress and logs

Normal execution prints timestamped progress events to the terminal:

```text
[2026-07-23T15:10:00+08:00] [PIPELINE] Starting bin with 28 tasks (threads=16, samples=3, memory=100 GiB); logs=result/tmp/runtime/logs
[2026-07-23T15:10:00+08:00] [RESUME] Reusing 20/28 complete tasks; 8 will run.
[2026-07-23T15:10:01+08:00] [START 3/28] 02.map.A.A - Map A reads to A (cpu=16)
[2026-07-23T15:17:14+08:00] [DONE 3/28] 02.map.A.A - completed in 7m13s
```

The task counter is the task position in the complete dependency graph.
It is not a linear completed-step counter. Parallel tasks can therefore finish
out of numerical order. The startup scan is represented by one aggregate
`RESUME` line instead of one line per task. MetaBAW does not print periodic
running heartbeats or a separate `RERUN` line. Terminal output changes only
when a task starts, completes, retries, or fails. Dependency-blocked tasks are
reported once as an aggregate `BLOCKED` line after scheduling finishes.
`FAIL` shows a short error tail and the exact task-log path; full commands and
program output remain in the task logs.

Resume decisions are output-first. A task is skipped whenever all of its
declared outputs exist and pass completeness validation, even if the previous
SQLite status says `running` or `failed`, the state file is absent, or the
task fingerprint changed after a MetaBAW update. Only missing, empty, or
invalid outputs are regenerated. Use `--force` when a deliberate parameter or
software change should replace otherwise complete existing results.

After every sample-specific binner command, MetaBAW validates the raw bin
directory before emitting `DONE`. At least one `.fa`, `.fna`, `.fasta`, or
compressed equivalent must exist, and every detected bin file must contain
one or more complete FASTA records with non-empty headers and sequences.
Missing, empty, truncated, or malformed bin output triggers one automatic
rerun even when `--retries` is zero. If that rerun also fails, the binner is
reported as failed and later binners continue according to the normal
failure-tolerant policy.

If an upstream task is rebuilt only because one of its own output files is
missing, downstream tasks are not invalidated automatically. After the
upstream task succeeds, each downstream task is independently reused when its
command/input fingerprint and outputs remain valid. For example, rebuilding a
Bowtie2 index does not remap reads when the existing BAM and BAI still match
the unchanged contigs, reads, and mapping command.

Each task has a complete log under `<tmp-files>/runtime/logs/`. A log records
the task ID, stage, description, local start and finish times, working
directory, temporary directory, command, memory ceiling, program output, and
exit code. Per-task environment overrides are also recorded. When a command
fails, the terminal displays the exit code, a recognized cause when available,
the last relevant output lines, and the full log path.

CheckM2 keeps the requested `--threads` value for gene calling and DIAMOND.
Its TensorFlow, OpenMP, MKL, OpenBLAS, BLIS, and NumExpr nested pools are
limited to one thread per pool. This prevents the model-prediction phase from
creating an additional large set of threads after the earlier multithreaded
phases have completed.

## Integrated data flow

- One `--min-contig-length` value is applied to input filtering and all
  binners.
- Selected binners are scheduled in the fixed priority order MetaBAT2,
  MetaDecoder, VAMB, COMEBin, SemiBin2, and LorBin. A failed binner is reported
  but does not stop later binners. MAGScoT continues with available successful
  assignment maps and fails clearly only when none are available.
- A global binning barrier waits for every selected binner and every sample to
  finish output validation and any automatic retry. No refinement task starts
  while any binning task is still running. Terminal tolerated failures satisfy
  the barrier so later refinement can continue with the successful bin sets.
- Thread-aware tools receive `--threads`, `-t`, `-p`, or their corresponding
  native option. Prodigal is split with GNU Parallel using `--threads` jobs.
  Up to `--task` samples are scheduled concurrently, and each running sample
  task receives `--threads` CPU threads.
- MetaBAT2, MetaDecoder, COMEBin, SemiBin2, and LorBin reuse filtered, sorted,
  and indexed BAM files. In multi-sample mode, each per-assembly BAM-based
  binner receives every BAM aligned to that assembly and no BAM aligned to a
  different assembly.
- Short-read VAMB uses strobealign `--aemb`. Each target assembly is profiled
  independently against every read sample, and the resulting per-sample
  matrices are merged with a mandatory `contigname` first column.
- Every binner result is normalized to one bin-contig-binner representation.
  DAS Tool input is converted automatically to its required contig-bin order.
- MAGScoT thresholds are shared with final completeness and contamination
  filtering.
- CheckM2 or CheckM, GUNC, RNA, and quality-score decisions are consolidated
  in `quality_summary.tsv` before Galah or dRep.
- Bacterial and archaeal GTDB-Tk summaries are merged into the same CoverM
  matrices with a consistent genome and taxonomy order.

## Development checks

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python -m compileall -q src tests
python -m pip wheel . --no-deps -w dist
```
