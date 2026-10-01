# MetaBAW 0.1.0

MetaBAW (Metagenome Binning Automated Workflow) is an integrated workflow for
building and annotating metagenome-assembled genome (MAG) datasets from
environmental metagenomes. It supports paired-end short reads, single-file long
reads, existing contigs, per-sample assembly, grouped coverage analysis, and
user-defined coassembly.

The workflow has two main modules:

- `metabaw binning`: assembly, read mapping, binning, bin refinement, MAG
  quality control, and dereplication.
- `metabaw annotation`: taxonomy, MAG abundance, ecological niche
  classification, KEGG annotation, CAZy annotation, and hydrogen-metabolism
  annotation.

MetaBAW uses explicit sample manifests instead of inferring sample names from
file names. This makes mixed projects and reproducible reruns easier to audit.

## Requirements and installation

Use Linux, WSL2, or an HPC Linux environment for production runs. MetaBAW
requires Python 3.11. External bioinformatics programs and reference databases
are not bundled in the wheel.

```bash
conda create -n metabaw python=3.11 pip "setuptools<82"
conda activate metabaw
python -m pip install ./metabaw-0.1.0-py3-none-any.whl

metabaw --version
metabaw --help
```

To replace an existing installation of this edition:

```bash
python -m pip install --force-reinstall --no-deps \
  ./metabaw-0.1.0-py3-none-any.whl
```

Check dependencies before the first analysis:

```bash
metabaw check --essential
metabaw check --scope binning
metabaw check --scope annotation
metabaw check --all
```

Use `metabaw check -h` to list database-path and environment options. The check
command validates required software and databases and can interactively install
supported missing components.

## Command help

```bash
metabaw binning -h           # core binning options
metabaw binning --full-help  # all binning options
metabaw annotation -h        # all annotation options
metabaw check -h             # dependency and database options
```

`metabaw bin` is an alias of `metabaw binning`.

## Input files

All manifests are UTF-8 text files. Fields must be separated by a real tab
character, not spaces or the literal characters `\t`. Empty lines and lines
beginning with `#` are ignored. Relative paths are resolved against the
directory containing the manifest.

### Read manifest

`--input_reads_files` is required by both modules.

```text
#sample	reads
S1	/data/reads/S1_R1.fastq.gz,/data/reads/S1_R2.fastq.gz
S2	/data/reads/S2_R1.fastq.gz,/data/reads/S2_R2.fastq.gz
L1	/data/reads/L1.fastq.gz
```

- Two comma-separated files are interpreted as paired-end short reads in R1,
  R2 order.
- One file is interpreted as long reads.
- Supported extensions are `.fastq`, `.fq`, `.fastq.gz`, and `.fq.gz`.
- Paired files may use different compression formats.
- Sample names must be unique. Letters, numbers, dots, underscores, and hyphens
  are accepted; `__mbw_` is reserved for internal identifiers.

### Optional contig manifest

`--input_contig_files` is optional for binning.

```text
#sample	contigs
S1	/data/contigs/S1.fa
S2	/data/contigs/S2.fa.gz
L1	/data/contigs/L1.fasta
```

The sample name must exist in the read manifest. Supported FASTA extensions are
`.fa`, `.fna`, `.fasta`, and their gzip-compressed variants. A partial contig
manifest is valid: samples without supplied contigs are assembled automatically.

### Genome list for annotation

`--input_genome_files` contains one MAG FASTA path per line and has no sample
name column.

```text
# one MAG per line
/data/mags/MAG_001.fa
/data/mags/MAG_002.fna.gz
```

MAG basenames must be unique after the complete FASTA suffix is removed.
MetaBAW stages only listed MAGs and does not modify the original files.

### Coassembly manifest

A coassembly manifest has two tab-separated columns: sample and group.

```text
#sample	group
S1	group1
S2	group1
S3	group2
S4	group2
```

Each sample may occur once, and each group must contain at least two samples.
All samples in one group must have the same read type: all paired short reads or
all single-file long reads.

### Grouped-coverage list

`--multi-files` accepts one existing sample name per line:

```text
S1
S2
S3
```

This mode cross-maps the listed reads to each listed sample assembly to obtain
multi-sample coverage. It does not coassemble the reads. Samples not listed in
the file remain independent.

## Binning workflow

### Per-sample assembly and binning

The minimal command needs only a read manifest:

```bash
metabaw binning \
  --input_reads_files reads.tsv \
  -t 32 --task 2 \
  -o metabaw_result
```

Missing short-read assemblies are generated with MEGAHIT. Missing long-read
assemblies are generated with `flye --meta`. Flye input modes are mutually
exclusive:

```text
--pacbio-raw  --pacbio-corr  --pacbio-hifi
--nano-raw    --nano-corr    --nano-hq
```

The default long-read mode is `--nano-raw`. These options select the Flye input
mode and the matching Minimap2 preset; read paths still come from the read
manifest.

Generated per-sample contigs are written as:

```text
assemblies/<sample>/<sample>_contig_ok.fa
```

Contig identifiers use `<sample>_<number>`.

Existing contigs can be supplied explicitly:

```bash
metabaw binning \
  --input_reads_files reads.tsv \
  --input_contig_files contigs.tsv \
  --tools metabat2 metadecoder vamb comebin semibin2 lorbin \
  --refinement magscot \
  --quality-control checkm2 \
  --gunc --trna --rrna \
  --dereplication-tool galah \
  --con 10 --com 50 \
  -t 64 --task 4 \
  -o metabaw_result
```

Default mapping and binner selection depends on the read type:

| Read type | Default mapper | Default binners | Inapplicable requested binner |
|---|---|---|---|
| Paired short reads | Bowtie2 | MetaBAT2, MetaDecoder, VAMB | LorBin |
| Single-file long reads | Minimap2 | MetaDecoder, VAMB, LorBin | MetaBAT2 |

The complete binner set is MetaBAT2, MetaDecoder, VAMB, COMEBin, SemiBin2, and
LorBin. Explicitly requested but inapplicable binners are recorded in
`binner_summary.tsv`; they are not treated as execution failures.

### User-defined coassembly

```bash
metabaw binning \
  --input_reads_files reads.tsv \
  --assembly-strategy coassembly \
  --coassembly-file coassembly.tsv \
  --tools metabat2 metadecoder vamb comebin semibin2 \
  -t 64 --task 2 \
  -o metabaw_coassembly_result
```

Paired short-read groups are coassembled with MEGAHIT. Long-read groups are
coassembled with Flye. All group members are mapped separately to the group
assembly to provide differential coverage. Samples not present in the
coassembly manifest follow the normal per-sample workflow.

The coassembly FASTA is named:

```text
coassembly/assemblies/coassembly_<group>/<group>.contigs.ok.fa
```

Its sequence identifiers use `<group>_<number>`. Group bins use:

```text
<group>_<sample1>-<sample2>-..._<binner-or-refiner>_<number>.fa
```

`--assembly-strategy coassembly` requires `--coassembly-file` and cannot be
combined with `--multi-files`.

### Refinement and quality control

Important binning options include:

| Option | Meaning | Default |
|---|---|---|
| `--refinement` | `magscot`, `das_tool`, or `metawrap` | `magscot` |
| `--quality-control` | `checkm2` or `checkm` | `checkm2` |
| `--con` | Maximum contamination (%) | `10` |
| `--com` | Minimum completeness (%) | `50` |
| `--quality-score` | Minimum completeness - 5 x contamination | not set |
| `--dereplication-tool` | `galah` or `drep` | `galah` |
| `--ani` | Dereplication ANI (%) | `99` |
| `--min-aligned-fraction` | Minimum aligned fraction (%) | `30` |
| `--gunc` | Run GUNC chimerism/contamination checks | off |
| `--trna` / `--rrna` | Predict and report RNA genes | off |
| `--trna-pass` / `--rrna-pass` | Apply RNA-based hard filters | off |

`--failure-policy strict` is the default and stops new work after a required
binner chain fails. `--failure-policy best-effort` continues with other
successful binners.

### Binning output

Key output locations are:

```text
metabaw_result/
|-- assemblies/ or coassembly/assemblies/
|-- bin_files/<binner>/
|-- quality_control_files/
|-- non_redundant_bins/
|-- binner_summary.tsv
|-- workflow_details.md
|-- start_info.txt
|-- run_manifest.json
|-- workflow_summary.json
`-- tmp/
```

Independent bins use `<sample>_<binner-or-refiner>_<number>.fa`. Final
dereplicated MAGs are written under `non_redundant_bins/`.

## Annotation workflow

Configure databases before the first annotation run:

```bash
metabaw check --scope annotation \
  --gtdbtk-db /db/gtdb/release232 \
  --kegg-db /db/kofam \
  --dbcan-db /db/dbcan \
  --hydrogenase-db /db/hydrogenase
```

Run annotation with the same read manifest used by binning and a MAG list:

```bash
metabaw annotation \
  --input_reads_files reads.tsv \
  --input_genome_files genomes.txt \
  --methods relative_abundance rpkm tpm mean \
  --niche-rank family \
  --niche_classify_method cv \
  --niche_abundance relative_abundance \
  -t 64 --task 4 \
  -o metabaw_annotation_result
```

The annotation module performs:

1. Taxonomic classification with GTDB-Tk.
2. MAG abundance profiling with CoverM.
3. Optional ecological niche classification.
4. Protein prediction with Prodigal.
5. KEGG Orthology annotation with KofamScan.
6. CAZyme annotation with run_dbCAN.
7. Hydrogenase annotation with HydDB searches.
8. Hydrogen-metabolism terminal-enzyme annotation with DIAMOND.

KEGG, CAZy, and hydrogen-metabolism annotation are enabled by default. Disable
them with `--no-kegg`, `--no-cazy`, or `--no-hyd`, respectively. `--no-hyd`
disables both hydrogenase and terminal-enzyme annotation.

### Taxonomy and abundance

GTDB-Tk species placement is enabled by default and can be disabled with
`--no-place-species`. A complete existing GTDB-Tk result can be reused with:

```bash
--gtdbtk_res /path/to/gtdbtk_result
```

MetaBAW validates that the supplied result contains one consistent
classification for every input MAG before skipping GTDB-Tk. `--gtdbtk_res` and
`--gtdbtk-db` are mutually exclusive.

Supported abundance methods are:

```text
relative_abundance  rpkm  tpm  mean
```

Each read sample is profiled against all listed MAGs. Taxonomic ranks are added
to the merged abundance tables.

### Ecological niche classification

Niche classification runs only when the analysis contains at least four MAGs
and four read samples. Otherwise it is skipped without blocking other
annotation steps. Use `--no-niche` to disable it explicitly.

Two criteria are supported:

- `--niche_classify_method cv`: corrected coefficient of variation (default).
- `--niche_classify_method occupancy`: sample-occupancy classification.

The default taxonomic rank is `family`, and the default abundance input is
`relative_abundance`. The selected niche abundance must also be included in
`--methods`. The decision, input counts, criterion, and abundance scale are
recorded in `niche_status.log` and niche provenance.

### Functional annotation outputs

Proteins are predicted once per MAG and mapped back to their source MAG and
contig. The principal functional tables are:

```text
kegg/kegg_annotations.tsv
cazy/cazy_annotations.tsv
hydrogenase/hydrogenase_annotations.tsv
terminal_enzymes/terminal_enzyme_annotations.tsv
```

The tables share stable MAG, contig, and gene identifier columns, followed by
method-specific evidence. Native combined-search output is retained for
provenance.

### Annotation output

```text
metabaw_annotation_result/
|-- gtdbtk_result/
|-- coverm/
|   |-- <sample>.tsv
|   |-- coverm_rel_abd.tsv
|   |-- coverm_rpkm_abd.tsv
|   |-- coverm_tpm_abd.tsv
|   |-- coverm_mean_abd.tsv
|   `-- coverm_all_metrics.tsv
|-- niche/
|   |-- niche_classification.tsv
|   |-- taxon_abundance.tsv
|   |-- genome_to_taxon.tsv
|   `-- provenance.json
|-- kegg/
|-- cazy/
|-- hydrogenase/
|-- terminal_enzymes/
|-- workflow_details.md
|-- start_info.txt
|-- niche_status.log
|-- run_manifest.json
|-- workflow_summary.json
`-- tmp/
```

## Resource management

### CPU scheduling

`-t/--threads` is the total workflow CPU budget. `--task` is the maximum number
of samples or MAGs processed concurrently. MetaBAW divides the total thread
budget across active sample slots; it does not multiply `-t` by `--task`.

For example, `-t 64 --task 4` permits four concurrent sample tasks with up to
16 threads each. A cohort-wide task may use the full CPU budget after its
dependencies finish.

### System memory

`--max-memory` is the combined workflow memory limit in GiB. Its default is the
detected machine memory, constrained by a smaller readable Linux cgroup limit.
The limit is shared by all active task process trees.

On Linux, MetaBAW monitors proportional set size (PSS) from `/proc`. RSS is
used only as a conservative upper bound when PSS is unavailable. Confirmed
limits stop active workflow process groups and block pending tasks.

Generated GTDB-Tk classification has a 160 GiB preflight floor, uses one
pplacer CPU, and enables disk scratch. This floor is a configuration check, not
a predicted peak. A complete `--gtdbtk_res` skips this requirement.

### GPU execution

MetaBAW probes CUDA automatically. Use `--no-gpu` to force CPU execution.
`--max-gpu-memory` defaults to `100%` of the detected total memory on the
selected visible GPU; explicit sizes such as `16G` or percentages such as
`50%` override the default.

VAMB, SemiBin2, COMEBin, and LorBin use GPU execution only when both their CUDA
runtime and conservative memory minimum pass validation. COMEBin's minimum is
derived from `--batch-size`; its GPU default is 1024. When COMEBin runs on CPU
and no batch size was explicitly supplied, MetaBAW uses the safer batch size
128.

## Dependencies and databases

MetaBAW integrates external programs according to the selected workflow. Major
components include:

- Assembly and mapping: MEGAHIT, Flye, Bowtie2, Minimap2, MinIBWA, and Samtools.
- Binning: MetaBAT2, MetaDecoder, VAMB, COMEBin, SemiBin2, and LorBin.
- Refinement: MAGScoT, DAS Tool, and MetaWRAP.
- MAG quality: CheckM2 or CheckM, GUNC, tRNAscan-SE, and Barrnap.
- Dereplication: Galah or dRep.
- Annotation: GTDB-Tk, CoverM, Prodigal, KofamScan, run_dbCAN, BLAST+, and
  DIAMOND.

Supported database groups include GTDB-Tk, CheckM2, legacy CheckM, GUNC, KOfam,
dbCAN, and the MetaBAW hydrogenase/terminal-enzyme database. Validate paths with
`metabaw check -h`.

Tools with incompatible Python requirements use isolated Conda environments:

| Tool | Default environment |
|---|---|
| COMEBin | `metabaw-comebin-py37` |
| CheckM2 | `metabaw-checkm2-py312` |
| MetaWRAP | `metabaw-metawrap-py27` |
| LorBin | `metabaw-lorbin-py310` |

`metabaw check --require-cuda` validates tensor allocation in relevant
environments and offers CUDA-enabled PyTorch repair when needed.

## Reports, logs, and reproducibility

Each run creates a compact set of audit files:

- `workflow_details.md`: human-readable numbered workflow steps, software, and
  key parameters selected for the run.
- `start_info.txt`: command, sample configuration, dependency checks, CUDA
  status, resource budgets, warnings, and failure policy. Repeated starts append
  timestamped records.
- `run_manifest.json`: machine-readable inputs, checksums, resolved workflow
  plan, task commands, dependencies, database paths, software versions, and
  source fingerprint.
- `workflow_summary.json`: final workflow status and task counts.
- `binner_summary.tsv`: per-target binner eligibility, completion, failure, and
  output information for binning runs.
- `memory_diagnostics.jsonl`: created only when memory accounting is degraded,
  a process has no readable address space, or a memory breach is suspected.
- `<tmp>/runtime/logs/`: one detailed log per workflow task.

`workflow_details.md` describes the planned methods. Use the final summary and
task logs to determine completion status.

## Temporary files and resume behavior

The default temporary directory is `<output>/tmp`. A relative custom path is
created under the output directory; an absolute path is used directly.

```text
<tmp>/
|-- work/
|-- system_tmp/
`-- runtime/
    |-- state.sqlite3
    `-- logs/
```

Temporary files are retained by default. `--delete-tmp-files` removes them only
after a successful workflow. Failed runs retain intermediates and logs.

Restart an interrupted workflow with the same command, output directory, and
temporary directory. Complete outputs are reused; incomplete tasks are rerun.
Use a new output and temporary directory whenever input manifests, grouping,
input content, or analysis parameters change.

## Examples

Current manifest and command examples are provided under `examples/`:

```text
examples/reads.tsv.example
examples/contigs.tsv.example
examples/genomes.txt.example
examples/multi.txt.example
examples/run.sh
examples/run.coassembly.sh
examples/run.annotation.sh
```

## Development verification

The test suite and wheel-integrity checker are not installed into the runtime
environment. From the source directory:

```bash
PYTHONPATH=src python -m pytest -q
python -m build --wheel --outdir .
python verify_wheel.py ./metabaw-0.1.0-py3-none-any.whl
```

## License

MetaBAW is distributed under the MIT License. See `LICENSE`.
