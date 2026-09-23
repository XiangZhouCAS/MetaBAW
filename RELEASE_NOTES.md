# Named-input edition — 0.1.0

Updated on 2026-09-23. Original edition preserved in the adjacent metaBAW directory.

## Implemented

- Made `--input_contig_files` fully optional. Missing per-sample contigs are
  generated before mapping/binning: paired short reads use MEGAHIT and
  single-file long reads use `flye --meta`. Partial contig manifests are valid;
  only their missing samples are assembled. Multi-files coverage waits for all
  generated assemblies, including SemiBin2 multi-sample preparation.
- Added mutually exclusive Flye modes `--pacbio-raw`, `--pacbio-corr`,
  `--pacbio-hifi`, `--nano-raw`, `--nano-corr`, and `--nano-hq` (default
  `--nano-raw`). Read paths remain in the named manifest. The same selection
  derives the long-read Minimap2 preset; removed `--long-read-preset`.
  Flye mode switches are now shown only by `bin --full-help`, not `bin -h`.
- Both workflows now default `--max-memory` to the detected physical memory
  total, constrained by a smaller readable Linux cgroup limit. Explicit values
  still override detection; systems where detection is unavailable fall back
  to 160 GiB.
- User-defined coassembly now accepts homogeneous paired-short groups with
  MEGAHIT or homogeneous single-file-long groups with Flye; mixed-technology
  groups are rejected before dependency checks. Generated individual contigs are
  named `SAMPLE_contig_ok.fa` with `SAMPLE_N` sequence IDs. Coassembly contigs are
  named `GROUP.contigs.ok.fa` with `GROUP_N` IDs. Group-level bins include the
  group, all contributing sample names, binner/refiner and sequence number.
- Fixed the reported skani false abort at 0.18 GiB under a 160 GiB budget:
  ESRCH from smaps_rollup plus successfully read, empty smaps identifies a PID
  without a user address space even while stat is still non-zombie. Exclude it
  only for the current sample, continue monitoring its descendants, and resample
  it on every poll. Record the exclusion and raw process state in diagnostics.
  Permission errors and actual memory breaches still retain their safeguards.
  Incomplete memory sums are now labeled partial accounted amounts, with null
  total upper bounds in diagnostics and the run manifest, rather than misleading
  users with a tiny purported total upper bound.
- Corrected process-tree memory accounting: prefer smaps_rollup PSS, then summed
  smaps PSS; exclude exited/zombie/recycled PIDs using process start-time checks.
  RSS is an explicitly labeled upper bound, never silently reported as PSS.
  Confirm runtime PSS breaches on two consecutive samples and hold new work during
  confirmation. Unverifiable budgets stop with a distinct accounting error, without
  disabling protection. Per-process evidence is saved in memory_diagnostics.jsonl.
- Annotation adds full-help-only --gtdbtk-threads (default 16, bounded by -t) and
  --gtdbtk-tmpdir (default /tmp). Other annotation tools retain their original CPU
  allocation. GTDB-Tk IPC/subprocess files now use private real temporary directories;
  known NFS/CIFS/SSHFS mounts and overly long socket paths are rejected. Large
  pplacer scratch remains in the workflow workspace. No classification criteria change.
- Added --niche_classify_method {cv,occupancy} and --niche_abundance {relative_abundance,rpkm,tpm,mean}.
- Public --methods now accepts/defaults to these four abundance measures only; count is internal niche support.
- Selected abundance is used for CV; non-percentage abundance is normalized within samples only for occupancy detection.
- Criterion, abundance scale and internal count usage are recorded; low-support taxa are reported without failing annotation.
- Named reads/contigs TSVs with explicit sample identity; contigs are optional.
- Annotation now uses the same named reads TSV plus a path-only genome list.
- Annotation directory/suffix/global-type flags removed; mixed reads choose per-sample CoverM mappers.
- Only listed MAGs are validated and staged, including compressed FASTA and files from different directories.
- Duplicate MAG basenames are rejected, provenance is recorded, and taxonomy/abundance/gene prediction depend on staging.
- Niche now requires at least 4 input MAGs AND 4 named read samples. Lower counts (including exactly 3)
  skip niche without blocking other annotation, and explicit --no-niche always disables it.
- Niche eligibility, counts and reasons are printed and persisted in niche_status.log and run_manifest.json.
- Per-row paired-short/single-file-long inference and mixed-type task planning.
- Per-target binner eligibility: paired excludes LorBin; long excludes MetaBAT2.
- One-name-per-line multi-files groups, using existing or automatically generated
  per-sample contigs and group-wide coverage.
- Restored explicit user-defined coassembly with `--assembly-strategy coassembly`
  and a required two-column `--coassembly-file`; homogeneous groups use MEGAHIT
  or Flye and unlisted samples use supplied contigs or automatic assembly.
- Removed obsolete automatic-group planning code, its isolated environment and
  reference-package checks, and the corresponding legacy check options/tests.
  User-defined coassembly and MEGAHIT/Flye dependency detection remain supported.
- Bare commands show only compact usage plus missing required inputs. Binning
  points to `--full-help`; annotation points to `-h`, which now displays every
  annotation option and replaces the removed annotation `--full-help` flag.
  Binning full help continues to omit its redundant long usage block.
- CUDA repair now treats a successful conda transaction as provisional. If the
  repaired runtime still reports `PyTorch CUDA none` or fails allocation,
  MetaBAW overlay-installs the official CUDA wheel set with pip
  `--ignore-installed` and retests before reporting the final status. This
  bypasses the uninstall phase when an incomplete old torch package is missing
  files such as `torch/include/ATen/ATen.h`.
- Binning and annotation route verbose startup diagnostics to `start_info.txt`:
  sample settings, software environments, CUDA checks, CPU fallback, resource budgets,
  startup warnings, and failure policy. Repeated starts append timestamped records.
  Input/group summaries, interactive prompts, fatal errors and runtime progress/warnings
  remain visible on the console. The startup log path is printed before execution.
- `workflow_details.md` now uses compact numbered steps, phase names and
  `Software: name (key parameters)` lines, merging identical settings across samples.
  Different thresholds, mapping presets and CPU/GPU configurations remain explicit.
  Full task commands, input/output paths, dependencies, resource/environment settings,
  manifest snapshots/checksums, resolved configuration and available software versions
  remain in `run_manifest.json`, alongside a MetaBAW source fingerprint.
  The concise report distinguishes planned methods from completed work.
- Unlisted samples remain independent; shared FASTA paths do not imply grouping.
- Per-read-type mapping presets, combined target-specific BAM sets, mixed-coverage VAMB input.
- SemiBin2 technology cohorts; singleton cohorts use all group BAMs with one assembly.
- Old binning input discovery, global type, multi and cohort flags removed; assembly
  strategy now has the explicit `existing-contigs` and `coassembly` choices.
- Resolved input provenance and inapplicable-binner reasons included in output summaries.
- Updated documentation and standalone examples. Old-parameter test script copies removed
  from this edition; their original copies remain in the original edition.

## Verification

- Regression tests cover conditional coassembly inputs, bare-command short help,
  pre-execution workflow reports for both modules, software metadata detection,
  2x4 niche criterion/abundance combinations, internal counts and low-support outputs.
- 220 tests passed on local Windows Python 3.13.4, including startup routing,
  both CLI modules with a mocked executor, compact method reports, and retained
  user-defined coassembly coverage; 25 standalone case-script tests previously passed.
- New assembly regression cases cover all six Flye modes and derived mapping
  presets, reads-only mixed MEGAHIT/Flye planning, long-read group coassembly,
  partial contig manifests, generated-contig/bin naming, and SemiBin2 dependency order.
- New regression cases simulate disappearing/recycled PIDs, shared pages, smaps
  fallback, PSS permission failures, transient spikes, confirmed breaches and
  uncertain RSS bounds, plus GTDB-Tk CPU caps and private temporary directory cleanup.
- Seven additional tests reproduce the supplied skani PID/start-time/PSS evidence,
  verify resampling and descendant accounting, reject stale RSS, retain valid
  fallback PSS, distinguish empty maps from permission/malformed-data failures,
  and check incomplete-bound diagnostics. Executor tests also retain shutdown
  protection when both PSS and RSS are genuinely unreadable.
- All source/test files also passed Python 3.11 syntax parsing (not execution).
- Wheel ZIP CRC, every RECORD hash/size, all 14 packaged Python sources, and metadata verified.
- Binning/annotation help and version commands are checked directly from the wheel during packaging.
- Current wheel SHA256 is recorded in SHA256SUMS.
- Original wheel SHA256 unchanged: c225669ab1c0e7f99048152cdff66f2f1a73889da88e8695054f3a08123054b0.

Production metadata remains Python >=3.11,<3.12. No Python 3.11 runtime or full Linux/GPU/external-binner
execution was available in this local verification. These tests verify Python logic, file handling,
and generated task dependencies, not biological performance or completed server-side training.

The release directory contains one wheel. The wheel was moved out of staging. Removal of generated
build, egg-info, bytecode caches and empty staging/old-example directories was blocked by the tool's
deletion policy; those directories remain and are not needed for wheel installation.

Use a new output and temporary directory for this edition and whenever manifests/group membership
or analysis settings change. The retained executor can reuse complete outputs after parameter changes.

## Transfer verification (Linux)

Place SHA256SUMS next to the wheel, then run:

```bash
sha256sum -c SHA256SUMS
python -m zipfile -t metabaw-0.1.0-py3-none-any.whl
```

Do not install a zero-byte file or a wheel that fails either integrity check.
