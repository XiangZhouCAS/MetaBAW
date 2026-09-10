# MetaBAW

MetaBAW: metagenome Binning Automated Workflow.

MetaBAW does not depend on Nextflow, Snakemake, or another workflow engine.
Python manages the task graph, CPU-aware concurrency, logs, retries, SQLite
state, caching, and resumable execution.

The command line has three modules:

- `metabaw bin`: read mapping, binning, refinement, quality control, and
  dereplication.
- `metabaw annotation`: GTDB-Tk taxonomy, CoverM abundance profiling,
  ecological niche classification, KEGG Orthology annotation with KofamScan,
  CAZy annotation with run_dbCAN, Fe/NiFe/FeFe hydrogenase annotation, and
  hydrogen-metabolism terminal-enzyme annotation.
- `metabaw check`: check selected software, ask before installing missing
  software with Mamba/Micromamba/Conda, and report database status.

## Installation

Production runs require Linux, WSL2, or a Linux HPC node and Python 3.11.
MetaBAW constrains its main environment to `>=3.11,<3.12`; tools with
incompatible dependency stacks, including Bin Chicken 0.14.1, run through
separately validated Conda environments.

```bash
cd metaBAW
mamba create -n metabaw-py311 python=3.11 pip "setuptools<82"
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

The Python package pins `setuptools<82` for legacy CheckM compatibility.
Other external programs are required only when their corresponding workflow
options are selected.
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
MetaBAW requires GTDB-Tk 2.7.0 or newer for the current R232 reference package
and reports older executables as broken before classification starts.

LorBin is installed from the official source commit
`ee10232282c2b71ed3ce2a34d5dbd78af3dd0b0a`, which fixes integer parsing for
its thread option. Its released Python 3.10 and PyTorch 1.11 dependency stack
is kept outside the main MetaBAW environment. MetaBAW uses Biopython 1.83
because the upstream Biopython 1.78 pin has no Conda build for Python 3.10;
the SeqIO interfaces used by LorBin remain compatible. LorBin retains its upstream
80 kb output threshold, and MetaBAW applies `--minfasta-kbs` while normalizing
LorBin bins so the shared minimum size remains effective. The environment
pins `mkl=2024.0.0` because newer MKL releases are binary-incompatible with
LorBin's PyTorch 1.11 build. If the managed `metabaw-lorbin-py310`
environment reports `iJIT_NotifyEvent`, MetaBAW removes and recreates that
environment instead of incrementally overwriting an inconsistent binary
stack. `metabaw check --scope lorbin` imports PyTorch,
allocates a test tensor, and imports LorBin rather than accepting the
executable path alone; a broken existing environment is therefore offered
for repair before binning starts.

| Stage | Default programs | Optional programs |
|---|---|---|
| Shell and installation support | Bash, Git | |
| Mapping | Bowtie2, Minimap2, SAMtools | minibwa |
| Abundance preparation | jgi_summarize_bam_contig_depths, strobealign | |
| Binning | Short: MetaBAT2, MetaDecoder, VAMB; long: MetaDecoder, VAMB, LorBin | COMEBin, SemiBin2 |
| Refinement | MAGScoT, GNU Parallel, Prodigal, HMMER, R | DAS Tool, DIAMOND, MetaWRAP |
| Quality control | CheckM2 | CheckM, GUNC, tRNAscan-SE, Barrnap, GTDB-Tk |
| Dereplication | Galah | dRep |
| Annotation | GTDB-Tk, CoverM, Minimap2, SAMtools, Prodigal, KofamScan, run_dbCAN, BLAST, DIAMOND | |

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

Legacy CheckM imports `pkg_resources`, which is absent from Setuptools 82 and
newer. MetaBAW probes `checkm --help` before a CheckM quality-control run and
offers to install `setuptools<82` in the active environment when this runtime
failure is detected. The repair can also be requested directly:

```bash
metabaw check --scope checkm
```

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

To check every supported external program, isolated environment, database, and
reference package, use `--all`. Invalid or unconfigured database paths are
requested interactively; enter an existing path and press Enter to validate and
save it. Press Enter on an empty response to leave that database unset and make
the check fail with a complete summary. MetaBAW still asks for confirmation
before installing missing software. Standard programs are listed together and
then installed as independent transactions. Missing isolated environments are
also listed together and confirmed once. A failed package or environment is
reported immediately, but every remaining selected installation is still
attempted before the final full recheck. Runtime defects that can only be
detected after a program is first installed, such as `gtdbtk:runtime` or
`checkm:runtime`, are repaired automatically in additional installation passes
within the same `metabaw check` invocation and do not require rerunning the
command:

```bash
metabaw check --all \
  --comebin-env /data1/zhoux/metabaw_envs/comebin-py37 \
  --checkm2-env /data1/zhoux/metabaw_envs/checkm2-py312 \
  --metawrap-env /data1/zhoux/metabaw_envs/metawrap-py27 \
  --lorbin-env /data1/zhoux/metabaw_envs/lorbin-py310 \
  --binchicken-env /data1/zhoux/metabaw_envs/binchicken-py311
```

Bin Chicken 0.14.1 is installed in its own Python 3.11 environment because it
cannot coexist reliably with GTDB-Tk 2.7.2 in the main MetaBAW environment.
Version validation reads each package record from the Conda prefix that owns
the executable. This prevents an installation loop in which Bin Chicken is
upgraded while GTDB-Tk is downgraded, followed by the reverse operation.
Saved references to the obsolete `metabaw-binchicken-py310` environment are
automatically migrated to `metabaw-binchicken-py311` during the next check.

The complete database/reference check covers legacy CheckM, CheckM2, GUNC,
GTDB-Tk, KOfam/KEGG, dbCAN/CAZy, HydDB/FeFe hydrogenases, the MAGScoT HMM
assets, and the Bin Chicken SingleM metapackage. CheckM2 and GUNC must resolve
to non-empty `.dmnd` files.
GTDB-Tk must resolve to the unpacked reference-data root rather than its parent
directory or archive.
An invalid `GTDBTK_DATA_PATH` is classified as a database configuration
problem, not a broken GTDB-Tk installation. Checks that include GTDB-Tk ask
for a replacement reference-data path and never offer to reinstall GTDB-Tk
solely because its current database path is missing or invalid. A valid saved
path also overrides a stale environment value during the runtime probe.

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

For `metabaw check`, an explicit environment option selects both the isolated
tool scope and the value to validate and save. For example,
`metabaw check --lorbin-env metabaw-lorbin-py310` checks only that LorBin
environment; `--scope lorbin` is no longer required. Add explicit
`-e/--essential` when the default dependency set should be checked in the
same invocation. Later `metabaw bin` resolves `METABAW_COMEBIN_ENV`,
`METABAW_CHECKM2_ENV`, `METABAW_METAWRAP_ENV`, `METABAW_LORBIN_ENV`, or
`METABAW_BINCHICKEN_ENV`
first, then saved configuration, and finally the built-in default. Set
`METABAW_CONFIG_FILE` to use a different configuration file. A single
environment can be changed and saved again:

```bash
metabaw check --scope comebin \
  --comebin-env /new/path/comebin-py37
metabaw check --lorbin-env metabaw-lorbin-py310 --require-cuda
```

CUDA is optional during a normal check. When the NVIDIA driver is available
but a selected PyTorch runtime is CPU-only or broken, an interactive check asks
whether it should install a CUDA-enabled build and then repeats the allocation
test. Declining the repair keeps CPU execution available. Add `--require-cuda`
when a failed driver or PyTorch allocation test must make the check return an
error. The main Python 3.11 environment uses the official pinned PyTorch 2.5.1
and CUDA 11.8 combination without mixing in conda-forge `libtorch` builds.
COMEBin keeps its isolated Python 3.7 environment and its package-required
PyTorch 1.10.2 build with CUDA Toolkit 11.3;
LorBin retains its working PyTorch 1.11.0 and CUDA 11.3 profile:

```bash
metabaw check --scope comebin --require-cuda
metabaw check --all --require-cuda
```

If an existing environment still prevents Conda from solving one of these
pinned profiles, MetaBAW retries with the corresponding official PyTorch CUDA
wheel and repeats the real tensor-allocation test.

To include database downloads:

```bash
metabaw check -e --install-databases \
  --database-dir ~/.cache/metabaw/databases
```

GTDB-Tk reference data requires approximately 100 GB. Database locations
installed by MetaBAW are recorded under
`~/.config/metabaw/databases.json`.

Existing annotation databases can be validated and saved without downloading
them again:

```bash
metabaw check --scope annotation \
  --gtdbtk-db /db/gtdb/release232 \
  --kegg-db /db/kofam \
  --dbcan-db /db/dbcan \
  --hydrogenase-db /db/hydrogenase
```

Legacy CheckM data can be configured once with `--checkm-db`; MetaBAW exports
the saved location through `CHECKM_DATA_PATH` whenever CheckM is needed by
CheckM quality control, MetaWRAP, or dRep:

```bash
metabaw check --all --checkm-db /db/checkm_data
```

The recognized CheckM v1 layout uses `genome_tree/` (not a top-level
`phylo/` directory), together with `hmms/`, `pfam/`, `distributions/`,
`selected_marker_sets.tsv`, and `taxon_marker_sets.tsv`. Invalid interactive
entries report the exact missing assets.

`--kegg-db` is the KOfam database root. It must contain a non-empty `ko_list`
and `profiles/`; additional files such as `K.descript.txt`, `K.ko.map.txt`,
and `ko.layer.txt` are accepted but are not required by KofamScan.
`--dbcan-db` must contain `CAZy.dmnd`, `dbCAN.hmm`, either
`dbCAN-sub.hmm` or `dbCAN_sub.hmm`, and `fam-substrate-mapping.tsv`.
`--hydrogenase-db` must contain the raw HydDB protein FASTA
`hyddb.all.fa`, the curated FeFe DIAMOND database `FeFe.dmnd`, the hydrogen
metabolism terminal-enzyme database `Terminal.dmnd`, and the two-column
ID-to-class table `hyd_id-name.script.txt`.
When `--install-databases` is selected, MetaBAW downloads the official KOfam
profiles and KO list and uses `run_dbcan database --aws_s3 --no-cgc` for the
CAZyme-only dbCAN assets.

Database path precedence is an explicit command-line option, a valid
environment variable, and then the path saved by `metabaw check`. If an old
`CHECKM_DATA_PATH`, `CHECKM2DB`, `GUNC_DB`, `GTDBTK_DATA_PATH`, `KOFAM_DB`,
`DBCAN_DB`, or `HYDROGENASE_DB` value
no longer exists, MetaBAW prints a warning and automatically uses a valid
saved path instead. An explicit command-line path is always authoritative,
including when it is invalid, so typing mistakes are reported directly.

## Binning

### Per-sample binning

Read names are split at the first `--separate-sample-name` separator
(default `.`): the part before the separator becomes the sample name. A
trailing `1`/`2` (or `R1`/`R2`, also after processing labels such as
`clean_R1`) in the remaining text marks the read mate. For example,
`SRR24653725.clean.rehost.1.fastq.gz` and
`SRR24653725.clean.rehost.2.fastq.gz` form sample `SRR24653725`. If the
separator is absent, `_R1/_R2`, `_1/_2`, and `.1/.2` pairing patterns are
used automatically.

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

With `--assembly-strategy default`, MetaBAW deliberately requires one
independently assembled contig file per sample. If several samples resolve to
the same contig file, MetaBAW stops with a clear error rather than treating an
unknown shared assembly as a per-sample input. The optional Bin Chicken
strategy creates its own explicitly tracked shared assemblies instead.

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
their per-sample assemblies. Remaining samples found under `--path` are
processed separately in single-sample mode.

### Bin Chicken-selected coassembly

`--assembly-strategy default` is the default and preserves the existing
independent-assembly workflow. `--assembly-strategy bin-chicken` adds a
targeted, short-read coassembly branch:

```bash
metabaw bin \
  -p reads/ -s fastq.gz \
  -c assemblies/ -f fa \
  --assembly-strategy bin-chicken \
  --tools metabat2 metadecoder vamb semibin2 \
  -t 32 --task 3 -o metabaw_coassembly
```

MetaBAW invokes `binchicken coassemble` only to select coassembly groups and
differential-abundance recovery samples. It **does not install Aviary**, does
not supply `--run-aviary`, and never consumes Aviary commands. Bin Chicken's
own planner may use its packaged implementation internally; MetaBAW itself
continues to schedule all assembly and binning tasks with its Python executor.
MetaBAW then uses MEGAHIT to create each selected shared assembly, maps every
Bin Chicken-selected recovery sample to that assembly, and runs the selected
MetaBAW binners, refinement, quality control, and dereplication steps.
Each coassembly is renamed from Bin Chicken's generic internal ID to a label
that lists its assembly members. For example, samples `A606` and `HBD-6066`
produce analysis label `coassembly_A606_HBD-6066`, refined MAG names such as
`coassembly_A606_HBD-6066_MAGScoT_1.fa`, and MEGAHIT contig identifiers
`coassembly_A606_HBD-6066_1`, `coassembly_A606_HBD-6066_2`, and so on. The
original Bin Chicken coassembly ID remains in the run manifest for provenance.

This strategy currently requires paired short reads because Bin Chicken's
`coassemble` command does not support long reads. MetaBAW saves the original
plan at `coassembly/binchicken/coassemble/target/elusive_clusters.tsv` and
writes `coassembly/bin_provenance.tsv` after dereplication. The provenance
table labels each final MAG as either `bin_chicken_coassembly` (with the
assembly and recovery samples) or `individual_assembly`.

`--assembly-strategy bin-chicken` cannot be combined with `--multi-files` or
`--cohort-size`: those options supply user-defined groups, while Bin Chicken
must select the groups itself. MEGAHIT remains in the main MetaBAW environment.
Bin Chicken is installed in the isolated `metabaw-binchicken-py311`
environment with Python 3.11 and `binchicken=0.14.1`; this prevents its Conda
dependencies from downgrading GTDB-Tk 2.7.2. Both are checked and offered for
installation when this strategy is selected or `metabaw check --all` is used.
Bin Chicken also needs
a SingleM metapackage for planning; configure its reusable location once:

```bash
metabaw check --scope binchicken \
  --binchicken-singlem-metapackage /path/to/singlem_metapackage
```

This check and `metabaw check --all` never install Aviary. Because `--all`
validates every supported reference asset, an unset or invalid SingleM
metapackage is requested interactively and causes a failed check if skipped.
Existing input contigs are used
for samples not selected for coassembly; selected groups receive new MEGAHIT
coassemblies under `tmp/work/coassembly/`.

Conda/Mamba installation transactions use a 60-second connection timeout, a
300-second idle-read timeout, and five HTTP retries. MetaBAW retries a transaction
up to two additional times after a transient network failure. Missing packages
are installed as separate restartable transactions. A failed package does not
stop later package or isolated-environment attempts in the same
`metabaw check`; successful transactions are retained and the final report
lists only dependencies that remain unavailable. Transactions inherit
the user's existing Conda/Mamba plugin, solver, channel, and package-cache
configuration so an automated single-package install behaves like the same
manual `mamba install PACKAGE` command. User-defined `CONDA_REMOTE_*`
environment variables take priority over the timeout defaults. If an
interrupted download later produces an archive extraction or invalid UTF-8
error, MetaBAW removes cached package archives once, then downloads the
affected package again.

VAMB is installed from PyPI with `python -m pip install vamb`, as
recommended by the VAMB project. It is therefore not included in MetaBAW's
Conda/Mamba installation transactions. The supported MetaBAW Python 3.11
runtime satisfies VAMB 5.0.3's Python requirement (`>=3.10,<3.14`).
SemiBin2, GUNC, GTDB-Tk, and galah are installed as separate transactions with
`mamba install --channel bioconda --channel conda-forge PACKAGE`, using the
package specifications `semibin`, `gunc`, `gtdbtk=2.7.2`, and `galah`,
respectively.

When `metabaw check` validates a GTDB-Tk database, it saves the path in
MetaBAW's reusable database configuration and sets `GTDBTK_DATA_PATH` for the
running process. In a Conda installation, it also persists the variable in the
Conda environment that contains the running MetaBAW interpreter. Reactivate
that environment once before invoking `gtdbtk` directly from the shell; MetaBAW
pipeline tasks receive the validated path immediately and do not require
reactivation.

### Long reads

```bash
metabaw bin \
  -p long_reads/ -s fastq.gz --type long \
  --align-tool minimap2 --long-read-preset map-ont \
  -c assemblies/ -f fa \
  -t 32 -o metabaw_result
```

Minimap2 is the default long-read mapper. minibwa is available for accurate
long reads. Minibwa is not treated as a Bioconda package: `metabaw check`
clones the official `lh3/minibwa` repository, builds the pinned source with
Make, and copies the resulting executable into the active MetaBAW environment.
The build requires Git, Make, a C compiler, and zlib; missing build dependencies
are installed from Conda Forge. Install or validate only this mapper with
`metabaw check --scope minibwa`. If `--tools` is omitted, short reads use MetaBAT2, MetaDecoder,
and VAMB, whereas long reads replace MetaBAT2 with LorBin and use MetaDecoder,
VAMB, and LorBin. Explicit `--tools` selections are not rewritten. LorBin is
accepted only with `--type long`.

### Quality and RNA filters

```bash
metabaw bin \
  -p reads/ -c assemblies/ \
  --con 10 --com 50 --quality-score 50 \
  --gunc --trna-pass 18 --rrna-pass \
  -t 32 -o metabaw_result
```

Configure CheckM2/GUNC databases beforehand with `metabaw check --all
--checkm2-db PATH --gunc-db PATH`. The bin command reads the saved paths.

- `--quality-score` uses `completeness - 5 * contamination`.
- `--con` sets the maximum contamination percentage; `--com` sets the
  minimum completeness percentage.
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
  on. The only final MAG directory remains `non_redundant_bins`; untagged
  representatives are retained only in `tmp/work/dereplication/`.

### GPU options

GPU use is automatic by default for supported VAMB, COMEBin, SemiBin2, or
LorBin configurations. Before task execution, MetaBAW checks `nvidia-smi`
and performs a real CUDA tensor allocation inside every selected Python
environment. CUDA eligibility is evaluated independently for each binner. A
binner with insufficient configured memory or a CPU-only/incompatible PyTorch
runtime falls back to CPU without disabling CUDA for other eligible binners.
`--gpu` is not needed and is not an available parameter. Use `--no-gpu` only
to skip CUDA detection and force CPU execution for every binner.
`--no-gpu` has priority over `--max-gpu-memory`: if both are supplied, MetaBAW
prints a warning and ignores the explicit GPU-memory budget.

`--max-gpu-memory` defaults to `4G`. MetaBAW applies conservative minimum
per-process budgets before launching GPU training: VAMB `4G`, SemiBin2 `4G`,
and LorBin `4G`. COMEBin is calculated from the requested batch size as
`max(1 GiB, batch_size / 128 GiB)`: batch sizes 128, 256, 512, 1024, and 2048
therefore require estimated minima of 1, 2, 4, 8, and 16 GiB, respectively.
If the resolved budget is below a selected binner's minimum, the log names
that binner, includes the COMEBin batch size where applicable, reports the
required minimum, and forces only that binner to CPU. Consequently, the
default `4G` allows COMEBin batch size 512 on GPU but runs the default batch
size 1024 on CPU. MetaBAW does not silently reduce a user-requested batch size:
the batch, resulting CPU fallback, and estimated minimum are recorded in
`run_manifest.json`. These values are MetaBAW scheduling estimates, not
universal hard memory requirements published by the upstream programs.

Because the supported external binners do not expose a reliable hard VRAM
cap, MetaBAW serializes GPU training tasks across the visible device set. This
prevents several sample tasks from overcommitting the same device even when
`--task` is greater than one. `--max-gpu-memory` remains a conservative
eligibility check for choosing GPU or CPU execution; it is not treated as a
virtual GPU partition. CPU-only steps can continue in parallel. Use
`CUDA_VISIBLE_DEVICES` to select devices.

### Important options

```text
--tools                 Any combination of metabat2, vamb, metadecoder,
                        comebin, semibin2, and lorbin
-t, --threads           Total CPU thread budget shared across concurrent
                        sample tasks; default 1
--task                  Maximum concurrently running samples; default 1
--cohort-size           Maximum per-sample assembly cohort size; omit to use
                        all selected samples in one cohort
--separate-sample-name  Custom sample-name delimiter for read discovery
--min-contig-length     Shared minimum contig length; default 1500 bp
--minfasta-kbs          Minimum bin size; default 200 kb
--no-gpu                Disable CUDA detection and GPU execution; default is
                        automatic CUDA detection
--max-gpu-memory        Per-task CUDA memory budget; ignored with --no-gpu;
                        default 4G
--batch-size            COMEBin batch size; default 1024; used to calculate
                        the conservative GPU-memory minimum
--environment           SemiBin2 pretrained environment model; it is passed
                        to SemiBin2 only when semibin2 is selected, otherwise
                        MetaBAW warns that it is ignored
--refinement            magscot, das_tool, or metawrap
--quality-control       checkm2 or checkm
--dereplication-tool    galah or drep
--ani                   Dereplication ANI threshold in percent; default 99
--min-aligned-fraction  Minimum aligned fraction in percent; default 30
-x, --mag-suffix        MAG file extension propagated to CheckM2/CheckM,
                        GUNC, GTDB-Tk, and dereplication; default fa
--task                  Maximum concurrent samples; default 1
--max-memory            Total workflow process-tree PSS ceiling in GiB;
                        default 100
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

Configure the databases once through `metabaw check`; the saved KOfam and
dbCAN paths are reused automatically by annotation runs:

```bash
metabaw check --scope annotation \
  --gtdbtk-db /db/gtdb/release232 \
  --kegg-db /db/kofam \
  --dbcan-db /db/dbcan \
  --hydrogenase-db /db/hydrogenase
```

```bash
metabaw annotation \
  -p metabaw_result/non_redundant_bins -s fa \
  -r reads/ -f fastq.gz \
  --methods relative_abundance rpkm tpm mean count \
  --niche-rank family \
  --niche-method cv \
  -t 32 --task 2 --max-memory 160 \
  -o metabaw_annotation_result
```

KEGG, CAZy, and hydrogenase annotation are enabled by default. MetaBAW predicts proteins
once per MAG with Prodigal, assigns collision-free protein identifiers, and combines all
predicted proteins into `tmp/work/annotation/proteins/all_mags.faa`. This combined protein
set is passed once to KofamScan, `run_dbcan CAZyme_annotation --mode protein`, HydDB, and
the terminal-enzyme database. Use `--no-kegg` or `--no-cazy`
to disable either function, or `--no-hyd` to disable both hydrogenase and
hydrogen-metabolism terminal-enzyme annotation. `--task` sets the maximum
number of read samples or MAGs processed concurrently, and the total
`-t/--threads` budget is divided evenly across those slots. For example, with
eight samples, `--task 2 -t 32` runs at most two samples at once and gives each
running sample task 16 threads; as a slot finishes, the next sample starts.
Per-sample tasks receive the allocated per-sample thread count. Each combined functional
annotation task runs once and receives the full `-t/--threads` budget because no other
functional database search is scheduled concurrently with it.

Hydrogenase annotation first builds a temporary BLAST protein index from
`hyddb.all.fa` under `tmp/work/annotation/hydrogenase`; the configured shared
database is never modified. The combined protein set is searched once with BLASTP at
E-value `1e-50`. MetaBAW keeps the best mapped hit per query, requires query
coverage of at least `0.9` and identity of at least `0.5`, and classifies the
hit from column two of `hyd_id-name.script.txt`. Fe and NiFe hits are retained
directly. FeFe candidates are extracted and confirmed against `FeFe.dmnd` by
DIAMOND BLASTP with E-value `1e-50` and one target per query. Native combined-search
outputs are written below `hydrogenase/combined/`; the final MAG-resolved result is
`hydrogenase/hydrogenase_annotations.tsv`.

The same protein FASTA is searched against `Terminal.dmnd` with DIAMOND.
All terminal-enzyme hits require query coverage of at least `0.80`. Identity
thresholds are marker-specific: PsaA `0.80`; HbsT `0.75`; PsbA, IsoA, AtpA,
YgfK, and ARO `0.70`; CoxL, MmoX, AmoA, NxrA, NuoF, RbcL, and hydrogenases
`0.60`; RHO `0.40`; and all remaining curated markers `0.50`. MetaBAW keeps
the best passing hit per gene. Native combined-search outputs are written below
`terminal_enzymes/combined/`, and the MAG-resolved result is
`terminal_enzymes/terminal_enzyme_annotations.tsv`.

The dbCAN path validator accepts both the current `dbCAN.hmm` and
`dbCAN-sub.hmm` names and the legacy `dbCAN.txt` and `dbCAN_sub.hmm` names.
Before annotation, MetaBAW creates zero-copy aliases under the temporary work
directory so either database layout can be used without modifying the shared
database.

Native KofamScan and run_dbCAN combined-search results are retained for provenance.
`tmp/work/annotation/gene_maps/all_mags.tsv` maps every collision-free query identifier
back to its original MAG, contig, and Prodigal protein identifier.
The four final functional tables (`kegg/kegg_annotations.tsv`,
`cazy/cazy_annotations.tsv`, `hydrogenase/hydrogenase_annotations.tsv`, and
`terminal_enzymes/terminal_enzyme_annotations.tsv`) use the same first five
columns: `MAG`, `MAG_contig_raw_id`, `MAG_contig_id`,
`MAG_contig_gene_id`, and `Gene`. `MAG_contig_raw_id` is the contig identifier
present in the input MAG. `MAG_contig_id` is the deterministic identifier
formed from the MAG name and contig order, and `MAG_contig_gene_id` adds the
Prodigal gene number. If contigs were already renamed by `metabaw bin`, the raw
and normalized contig columns are identical. `Gene` contains the KO, CAZy
family/result, hydrogenase class, or terminal-enzyme name, depending on the
table. Tool-specific evidence columns follow these five common columns.
Missing combined-search outputs invalidate the corresponding functional task during resume.

GTDB-Tk species placement is enabled by default. Use `--no-place-species` to
disable it. `--place-species` and `--no-place-species` are mutually exclusive;
supplying both is rejected before the workflow starts.
The annotation runner probes `classify_wf --help` before adding
`--place_species`, so unsupported options are not passed to older releases.

An existing GTDB-Tk output can be reused with
`--gtdbtk_res /path/to/gtdbtk_result`. MetaBAW recursively reads
`gtdbtk.*.summary.tsv`, requires the `user_genome` and `classification`
columns, and confirms that every input MAG has one non-empty, non-conflicting
classification. GTDB-Tk's legitimate `Unclassified Bacteria` and
`Unclassified Archaea` records are accepted; downstream rank columns retain
their existing `unclassified` behavior. Output-root symlinks and their target
summary files are read only once. A complete result skips the GTDB-Tk task and
its software/database preflight; downstream CoverM merging and niche
classification read taxonomy directly from the supplied path. An incomplete
result stops before the workflow starts and reports that GTDB-Tk must be
rerun. `--gtdbtk_res` and `--gtdbtk-db` are mutually exclusive.
Generated GTDB-Tk output is reusable only after its per-MAG summaries pass the
same validation and MetaBAW writes `gtdbtk_result/.metabaw.complete`; an
interrupted non-empty directory is therefore rerun rather than adopted.

Supported CoverM metrics are:

```text
relative_abundance  rpkm  tpm  mean  count  trimmed_mean
covered_fraction    covered_bases  reads_per_base  variance  length
```

CoverM estimators that do not calculate covered bases (`count`,
`reads_per_base`, `rpkm`, `tpm`, and `length`) are calculated together with
`--min-covered-fraction 0`. Coverage-aware estimators retain the intended 10%
minimum covered-fraction filter. MetaBAW merges the two per-sample tables before
taxonomy and niche outputs are generated. This split avoids CoverM's rejection
of threshold-incompatible estimator combinations while preserving the filter
for metrics that support it.
Before CoverM starts, MetaBAW validates the real `samtools --version` output
and places a temporary PATH shim in the work directory. The shim removes
linker warnings from the version probe while forwarding every alignment
operation to the original samtools executable unchanged.

GTDB-Tk is exempt from the secondary Linux `RLIMIT_AS` guard used alongside
`--max-memory`. Its consolidated skani database is memory-mapped and may need
a virtual address range larger than the configured resident-memory budget.
GTDB-Tk and all of its descendants are still included in the workflow-wide
resident-memory total.
Generated GTDB-Tk classification reserves 160 GiB of total memory, above the
140 GiB minimum documented for GTDB release R232. MetaBAW forces
`--pplacer_cpus 1` and enables GTDB-Tk's disk-backed `--scratch_dir`; this
prevents the general `-t/--threads` allocation from multiplying pplacer's large
reference-tree memory. Annotation therefore defaults to `--max-memory 160`; a
lower value is accepted only when a complete matching `--gtdbtk_res` skips the
GTDB-Tk task. The other GTDB-Tk phases may still use the allocated CPU threads.

metaBAW writes one matrix per metric. GTDB-Tk domain, phylum, class, order,
family, genus, and species columns are inserted after `Genome`. A MAG without
a GTDB-Tk species assignment is reported as `unclassified`; its MAG name is
not copied into the species column.

Supported niche levels are:

```text
domain phylum class order family genus species strain
```

The default niche level is `family`. `--niche-method cv` is the default and
classifies eligible taxa from the corrected coefficient of variation
`CV - sqrt(K/N)` relative to the eligible-community mean. Use
`--niche-method prevalence` to classify taxa from core prevalence instead.

Niche classification requires at least two samples and both
`relative_abundance` and `count`. MetaBAW validates these requirements before
building the annotation workflow and reports a direct error instead of
silently omitting the final niche result. Use `--no-niche` when taxonomy and
abundance outputs are required without ecological niche classification.
All four niche files are checked during resume, so a missing classification,
taxon-abundance, genome-assignment, or provenance file causes only the niche
step to run again.

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
on. MAGScoT intentionally runs in score-only mode when only one binner result
is available. MetaBAW detects that successful mode, preserves the original
binner assignments and names, and continues to quality control. DAS Tool
follows the same provenance rule: an unchanged selected MAG
keeps a name such as `A606_MetaBAT2_1.fa`, while a MAG assembled or modified
by DAS Tool is named `A606_DASTool_1.fa`. DAS Tool's internal label prefix,
such as `metabat2__`, is removed during normalization. The normalized FASTA
set under `<tmp-files>/work/refinement/<sample>/*_DASTool_bins` is the exact
set collected into `quality_control_files/candidate_bins`. MetaWRAP also uses
this provenance rule inside `metawrap_50_10_bins`: unchanged MAGs retain
their original source-tool names, while only merged or modified MAGs receive
names such as `A606_MetaWRAP_1.fa`. The normalized MetaWRAP directory and
`quality_control_files/candidate_bins` contain the same FASTA names and
content. When `--refinement metawrap` and `--quality-control checkm` are used
together, MetaBAW maps MetaWRAP's final `*.stats` CheckM results through this
same manifest and writes `quality_control_files/checkm/quality_report.tsv`;
it does not run another CheckM workflow after refinement. Internal source
identifiers such as `cleanbin_000030` remain
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
|   |-- manifest.json
|   `-- merge_schema.json
|-- niche/
|   |-- niche_classification.tsv
|   |-- taxon_abundance.tsv
|   |-- genome_to_taxon.tsv
|   `-- provenance.json
|-- kegg/
|   |-- kofamscan_mapper.tsv
|   `-- kegg_annotations.tsv
|-- cazy/
|   |-- run_dbcan/
|   `-- cazy_annotations.tsv
|-- hydrogenase/
|   |-- combined/
|   `-- hydrogenase_annotations.tsv
|-- terminal_enzymes/
|   |-- combined/
|   `-- terminal_enzyme_annotations.tsv
|-- tmp/
|   |-- work/
|   |-- system_tmp/
|   `-- runtime/
`-- run_manifest.json
```

`merge_schema.json` records the merged CoverM table schema. MetaBAW uses this
visible marker to rebuild incompatible legacy merged tables during resume
without rerunning completed per-sample CoverM profiles.

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

Outputs left by a task whose recorded state is `failed` are never adopted as
a completed checkpoint, even when they superficially resemble a valid output.
The failed task is rerun while earlier successful tasks remain reusable.

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
`-t/--threads` is a workflow-wide CPU budget, not a per-sample value. With
three available samples, `-t 128 --task 3` assigns 42 threads to each running
sample task and never raises the executor CPU ceiling above 128. The two
integer-division remainder threads remain available for scheduler and small
helper tasks. If only one sample can run, it may receive the full budget.
Cohort-wide single tasks, such as the final SemiBin2 multi-sample training
task, may use the full 128 threads after their sample dependencies finish.

LorBin receives `--num_process` from its allocated per-sample share of
`--threads`. Because LorBin creates a
process pool and also invokes threaded numerical libraries, MetaBAW limits
OpenMP, MKL, OpenBLAS, BLIS, NumExpr, and Accelerate to one nested thread per
LorBin worker. This preserves the requested LorBin worker count and
multi-sample `--task` concurrency without multiplying each worker into
additional unmanaged threads. A LorBin checkpoint also requires
`<work>/binning/<sample>/lorbin/.metabaw.complete`, which is written only
after LorBin exits successfully; interrupted partial bin directories are
therefore rerun rather than adopted.

`--max-memory` is available in both modules. Its unit is GiB; the default is
`100` for `bin` and `160` for `annotation`, whose generated GTDB-Tk task uses
one pplacer CPU, disk scratch, and a 160 GiB reserve. All concurrently running MetaBAW task process
trees share this one physical-memory budget. On Linux, MetaBAW reads the
process hierarchy and `Pss` from `/proc/<pid>/smaps_rollup`; PSS apportions
shared resident pages among the processes mapping them, so GTDB-Tk workers do
not count the same memory-mapped reference database multiple times. `VmRSS` is
used only as a fallback when PSS is unavailable. `--task 6` does not create six
independent allowances. The configured value is retained as a secondary
per-process virtual-address guard where applicable. If the summed PSS exceeds
the limit, or if a task reports a CPU-memory allocation failure, MetaBAW prints
the observed usage in a `[MEMORY LIMIT]` workflow event,
disables retries, terminates every process group started by the workflow, and
blocks all pending tasks. CUDA out-of-memory errors remain separate because
they are governed by `--max-gpu-memory`.

Before any workflow task starts, MetaBAW prints `[MEMORY ESTIMATE]` with the
estimated minimum total memory, configured limit, and headroom or shortfall.
The estimate is a conservative floor derived from declared high-memory task
requirements; data-dependent additional usage is still watched at runtime. If
a known minimum already exceeds `--max-memory`, MetaBAW writes
`tmp/runtime/logs/memory_preflight.log`, prints `[MEMORY CONFIG ERROR]` with an
actionable correction, and exits without starting a workflow process. If a
data-dependent runtime peak crosses the limit instead, the final `[ERROR]`
summary repeats the observed PSS and configured ceiling rather than reporting
only blocked task IDs.

## Runtime progress and logs

Normal execution prints elapsed runtime in `HH:MM:SS` followed by concise
progress events:

```text
MetaBAW is Running.
Command: metabaw bin -p reads -s fastq.gz -c contigs -f fa -o result
[00:00:00] [PIPELINE] Running bin with 28 tasks (total_threads=16, threads_per_sample=5, concurrent_samples=3, total_memory_limit=100 GiB); logs=result/tmp/runtime/logs
[00:00:00] [RESUME] Reusing 20/28 complete tasks; 8 will run.
[00:00:01] [STEP 1/2] 02.map - running 3 tasks across 3 samples [A,B,C] (cpu=16/sample)
[00:07:14] [STEP 2/2] 03.bin.metabat2 - running 3 tasks across 3 samples [A,B,C] (cpu=16/sample)
[00:19:31] [COMPLETE] Workflow completed successfully; status=skipped=20, success=8; results=result; temporary files=result/tmp
ALL DONE.
```

Every executed command starts with the fixed banner and the exact reconstructed
command line. `ALL DONE.` is printed only after successful completion and is
always the final output line; failed or interrupted runs never print it.
Terminal progress is grouped by logical step. Each step prints one `STEP`
line containing all participating samples. There are no per-step `START` or
`DONE` lines. Logical steps run sequentially, while the sample tasks inside
the active step still use the concurrency selected with `--task`. The next
`STEP` line therefore indicates that the previous step completed. Individual
commands remain in per-task log files.
Failures print an immediate `ERROR` for the affected sample and one final
group-level `FAIL`. The progress counter is the grouped-step position among
the steps that are not reused from cache. The startup scan is represented by one aggregate
`RESUME` line instead of one line per task. MetaBAW does not print periodic
running heartbeats or a separate `RERUN` line. Terminal output changes only
when a task starts, completes, retries, or fails. Dependency-blocked tasks are
reported once as an aggregate `BLOCKED` line after scheduling finishes.
`FAIL` shows a short error tail and the exact task-log path; full commands and
program output remain in the task logs. A successful workflow ends with one
final `[COMPLETE]` result line containing its status counts, result path, and
temporary file disposition, followed by `ALL DONE.`.

Resume decisions are output-first. A task is skipped whenever all of its
declared outputs exist and pass completeness validation, even if the previous
SQLite status says `running` or `failed`, the state file is absent, or the
task fingerprint changed after a MetaBAW update. Only missing, empty, or
invalid outputs are regenerated. Use `--force` when a deliberate parameter or
software change should replace otherwise complete existing results.

After every sample-specific binner command, MetaBAW validates the raw bin
directory before advancing to the next step. At least one `.fa`, `.fna`, `.fasta`, or
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
  binners. MetaBAT2 is clamped to its native 1500 bp minimum when a smaller
  shared value is requested.
- Selected binners are scheduled in the fixed priority order MetaBAT2,
  MetaDecoder, VAMB, COMEBin, SemiBin2, and LorBin. A failed binner is reported
  but does not stop later binners. MAGScoT, DAS Tool, and MetaWRAP continue
  with the available successful bin sets and fail clearly only when none are
  available. MetaWRAP still accepts no more than three initially selected
  binners.
- A global binning barrier waits for every selected binner and every sample to
  finish output validation and any automatic retry. No refinement task starts
  while any binning task is still running. Terminal tolerated failures satisfy
  the barrier so later refinement can continue with the successful bin sets.
- Thread-aware tools receive `--threads`, `-t`, `-p`, or their corresponding
  native option. Prodigal is split with GNU Parallel using `--threads` jobs.
  Up to `--task` samples are scheduled concurrently. The total
  `-t/--threads` budget is divided across those sample slots, and the executor
  enforces the original total as a hard scheduling ceiling.
- Mamba or micromamba can still create and repair isolated environments. When
  Conda is also available, parallel workflow tasks use `conda run` so multiple
  samples do not contend for mamba's package-cache process lock.
- MetaBAT2, MetaDecoder, COMEBin, SemiBin2, and LorBin reuse filtered, sorted,
  and indexed BAM files. In multi-sample mode, each per-assembly BAM-based
  binner receives every BAM aligned to that assembly and no BAM aligned to a
  different assembly.
- Short-read VAMB uses strobealign `--aemb`. Each target assembly is profiled
  independently against every read sample, and the resulting per-sample
  matrices are merged with a mandatory `contigname` first column.
- Every binner result is normalized to one bin-contig-binner representation.
  DAS Tool input is converted automatically to its required contig-bin order.
  SemiBin2 multi-sample outputs are normalized across versions that use
  `output_bins`, `output_recluster_bins`, or `output_prerecluster_bins`; every
  cohort member must produce a valid FASTA bin directory before the run is
  accepted.
- MAGScoT thresholds are shared with final completeness and contamination
  filtering.
- CheckM2 or CheckM, GUNC, RNA, and quality-score decisions are consolidated
  in `quality_summary.tsv` before Galah or dRep.
- When dRep is selected, MetaBAW converts the existing CheckM2 or CheckM
  quality table to dRep `--genomeInfo` format. dRep therefore retains
  completeness/contamination-based representative selection without rerunning
  legacy CheckM or requiring a separate CheckM database.
- Bacterial and archaeal GTDB-Tk summaries are merged into the same CoverM
  matrices with a consistent genome and taxonomy order.

## Development checks

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python -m compileall -q src tests
python -m pip wheel . --no-deps -w dist
```
