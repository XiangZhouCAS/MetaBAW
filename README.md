# MetaBAW 0.1.0 — named-input edition

这是独立的“具名清单输入版”，原版保留在旁边的 `metaBAW` 目录。
本版 binning 和 annotation 共用具名 reads 清单，支持混合双端短读长和单文件长读长。
contigs 清单可选：缺失 contigs 的双端短读样本用 MEGAHIT，单文件长读样本用 `flye --meta` 自动组装。
`--multi-files` 只计算全组覆盖度，不重新组装 reads；`--assembly-strategy coassembly`
则按用户分组调用 MEGAHIT 或 Flye 共组装同类型 reads。

## 安装

生产运行使用 Linux/WSL2/HPC 和 Python 3.11。两版包名和版本相同，建议使用不同环境。

```bash
conda create -n metabaw-named python=3.11 pip "setuptools<82"
conda activate metabaw-named
python -m pip install ./metabaw-0.1.0-py3-none-any.whl
metabaw --version
metabaw binning -h
metabaw binning --full-help
```

同一环境更新本版时：

```bash
python -m pip install --force-reinstall --no-deps ./metabaw-0.1.0-py3-none-any.whl
```

`--force-reinstall` 是 pip 参数。MetaBAW binning/annotation 均无 `--dry-run`、`--force`。

CUDA 修复会在 conda 安装后重新执行 PyTorch CUDA 分配测试。如果 conda 返回成功但
环境仍是 CPU 版 PyTorch（`PyTorch CUDA none`），MetaBAW 会自动从官方 cu118 源
覆盖安装匹配的 torch、torchvision 和 torchaudio，再次验证后才判定修复结果。
覆盖安装使用 pip 的 `--ignore-installed`，不会先卸载可能已经不完整的旧版 PyTorch，
因此也能修复缺失文件或遗留 `~orch`/`~ensorpipe` 临时目录的环境。

## 输入格式

两份清单都是 UTF-8 文本，每行两列，列间使用真正的 Tab，而非空格或字面量 `\t`。
空行和 `#` 开头的注释行可忽略；普通表头不可用。相对路径按清单所在目录解析。
样本名在各清单内唯一。`contigs.tsv` 可以省略或只列出部分 reads 样本；未列出的样本自动组装。
contigs 清单不能包含 reads 清单中不存在的样本。
同一 contigs 文件可明确分配给不同样本，不会因此自动开启多样本分箱；reads 路径不可重复。

`reads.tsv` 示例（文件路径需换成实际存在的文件）：

```text
#sample	reads
S1	/data/reads/S1_1.fastq.gz,/data/reads/S1_2.fastq.gz
S2	/data/reads/S2_1.fastq,/data/reads/S2_2.fastq
L1	/data/reads/L1.fastq.gz
```

- 两个逗号分隔的文件：双端短读长，按写入顺序传递 R1/R2。
- 一个文件：按本版约定视为长读长，不是从序列内容检测平台。
- 支持 `.fastq`、`.fastq.gz`、`.fq`、`.fq.gz`，两端压缩格式可不同。
- 普通短读长单端文件和 interleaved 双端文件不可按这个约定当作长读长；双端需拆为两个文件。

`contigs.tsv`：

```text
#sample	contigs
S1	/data/contigs/S1.fa
S2	/data/contigs/S2.fa
L1	/data/contigs/L1.fa
```

支持 `.fa`、`.fna`、`.fasta` 及其 gzip 格式。具名清单不依据文件名猜测样本关联。
样本名可用字母、数字、下划线、点、短横线，以字母或数字开头，不能以点或下划线结尾；
`__mbw_` 为内部保留字符串。合法名字原样保留，`A_clean` 不会变成 `A`。

只提供 reads 时：

```bash
metabaw binning --input_reads_files reads.tsv --nano-raw -o result_reads_only
```

每个缺失 contigs 的样本先独立组装。输出为
`OUTPUT/assemblies/<样本名>/<样本名>_contig_ok.fa`，序列 ID 为 `<样本名>_1`、`<样本名>_2`……。
长读类型选项互斥，可用 `--pacbio-raw`、`--pacbio-corr`、`--pacbio-hifi`、
`--nano-raw`、`--nano-corr` 或 `--nano-hq`；默认 `--nano-raw`。FASTQ 路径仍来自
`--input_reads_files`，这些选项只选择 Flye 输入模式，并同步决定 Minimap2 使用
`map-pb`、`map-hifi` 或 `map-ont`。这些 Flye 选项仅在 `bin --full-help` 中显示，
不再出现在 `bin -h`；原 `--long-read-preset` 已删除。

## 独立分箱

```bash
metabaw binning \
  --input_reads_files reads.tsv \
  --input_contig_files contigs.tsv \
  --tools metabat2 metadecoder vamb comebin semibin2 lorbin \
  --refinement magscot --quality-control checkm2 \
  --gunc --trna --rrna \
  --dereplication-tool galah --ani 99 --min-aligned-fraction 30 \
  --con 10 --com 50 --min-contig-length 1500 --minfasta-kbs 200 \
  --no-gpu -t 32 --task 1 --max-memory 200 -o result_named
```

不设 `--multi-files` 时，每份 contigs 只使用所属样本的 reads。
各目标 contigs 的 binner 按所属样本的读长类型选择：

| reads 类型 | 默认比对 | 默认 binner | 显式 --tools 排除规则 |
|---|---|---|---|
| 双文件短读长 | Bowtie2 | metabat2, metadecoder, vamb | 排除 lorbin |
| 单文件长读长 | Minimap2 | metadecoder, vamb, lorbin | 排除 metabat2 |

其余请求的工具保留。排除规则打印警告并写入 binner summary，不等同于工具运行失败。
某样本排除后无可用工具时，在启动任务前报错。
`--align-tool minimap2` 可处理混合类型；短读长用 `sr`，长读长预设由上述 Flye 数据类型自动确定。
显式 Bowtie2 不可处理长读长，显式 minibwa 不可处理双端短读长。

## 指定多样本组

`multi.txt` 每行只有一个名字，至少两个不同样本，必须存在于两份输入清单：

```text
S1
S2
L1
```

```bash
metabaw binning \
  --input_reads_files reads.tsv \
  --input_contig_files contigs.tsv \
  --multi-files multi.txt \
  --tools metabat2 metadecoder vamb comebin semibin2 lorbin \
  --no-gpu -t 32 --task 1 --max-memory 200 -o result_named_multi
```

组内每份 contigs 接收全组 reads 的覆盖度，以上三样本对应 3 × 3 次目标比对。
缺少 contigs 时先按样本分别用 MEGAHIT/Flye 组装，再执行上述交叉回贴；这不是把全组 reads 合并共组装。
未列入组的样本独立运行。同一次运行只定义一个组；比较不同组需分别运行、使用不同输出目录。
应依据研究设计选择适合联合覆盖度分析的样本，不要把无关生态系统样本随意合组。

混合组内每个 reads 样本使用适合其类型的比对方式，BAM 只汇集到对应目标 contigs。
VAMB 对全短读长覆盖度使用 AEMB；覆盖度包含长读长时使用含全组样本的 BAM 目录。
SemiBin2 按目标样本读长类型分组训练：同类型多份 contigs 使用 multi_easy_bin；
同类型仅一份 contigs 时使用 single_easy_bin 接收全组 BAM，均保留全组覆盖度、自监督训练。
用于 SemiBin2 的 FASTA 拼接只是输入组织，不是原始 reads 重组装。
独立样本仍可用 `--environment` 指定 SemiBin2 环境模型。

## 指定共组装组

共组装分组文件为两列制表符文本：第一列是 `reads.tsv` 中的样本名，
第二列是组名。空行和 `#` 注释可忽略；每个样本只能出现一次，每组至少两个样本。

```text
#sample name	group
sampleA	group1
sampleB	group1
sampleC	group1
sampleD	group2
sampleE	group2
sampleF	group2
```

```bash
metabaw binning \
  --input_reads_files reads.tsv \
  --assembly-strategy coassembly \
  --coassembly-file coassembly.tsv \
  --tools metabat2 metadecoder vamb comebin semibin2 \
  --no-gpu -t 32 --task 1 --max-memory 200 -o result_coassembly
```

MetaBAW 为每组创建一个内部分析。同组必须全部为双端短读长或全部为单文件长读长，不能混合；
短读组由 MEGAHIT 共组装，长读组由 `flye --meta` 按所选长读类型共组装。
全部成员 reads 分别回贴到共组装 contigs 计算差异覆盖度，每组只产生一套 bins。
共组装 FASTA 为 `OUTPUT/coassembly/assemblies/coassembly_<组名>/<组名>.contigs.ok.fa`，
其中序列 ID 为 `<组名>_1`、`<组名>_2`……。
公开 bin 名为 `<组名>_<组内样本名以-连接>_<binner>_<序号>.fa`，例如
`group1_sampleA-sampleB_metabat2_1.fa`。这保留全部覆盖度来源，不把同一组级 bin 重复归属给单一样本。
最终 MAG 来源表保存于 `OUTPUT/coassembly/bin_provenance.tsv`。
无需提供 `--input_contig_files`；未分组且没有 contigs 的样本会按自身读长类型自动独立组装。
也可提供包含所有样本的 contigs 清单；组内样本的已有 contigs 不用于共组装。
`--coassembly-file` 只在 `--assembly-strategy coassembly` 下有效；共组装模式不能同时设置
`--multi-files`，避免两套分组定义冲突。

## 参数、结果与恢复

本版 binning 移除 `-p/--path`、`-c/--contig`、`-s/--suffix`、`-f/--contig-suffix`、
`--separate-sample-name`、`--type`、`--single`、`--multi`、`--cohort-size`。
不接受参数缩写，`--multi` 不会误认为 `--multi-files`。
`metabaw bin` 仍是 binning 的兼容别名，但只接受新版接口。

- 不加参数的命令仅显示简短 usage 和缺失的必填参数；binning 提示使用
  `--full-help`，annotation 提示使用 `-h`。
- binning 的 `-h` 只显示主要选项，其余见 `--full-help`；annotation 的 `-h/--help`
  直接显示全部参数，不提供 `--full-help`。
- `metabaw bin --full-help` 省略重复的长 usage 参数串，直接显示完整参数说明。
- `--failure-policy strict` 默认：适用且已选的 binner 失败会阻止后续任务、非零退出。
  显式 best-effort 才允许使用其他成功工具继续。
- `--con` 最大污染度，默认 10%；`--com` 最低完整度，默认 50%。
- `--refinement` 可选 magscot（默认）、das_tool、metawrap。MetaWRAP 每目标最多三个适用 binner。
- `--quality-control` 可选 checkm2（默认）、checkm；`--gunc` 启用嵌合体检查。
- `--trna`、`--rrna` 只预测报告；`--trna-pass N`、`--rrna-pass` 才启用对应硬过滤。
- `--quality-score` 用 completeness − 5 × contamination；去冗余可选 galah（默认）或 drep。
- `-t` 为总线程调度预算，`--task` 为最大并行样本数。`--max-memory` 为全流程 PSS 总上限
  （PSS 不可用时 RSS 仅作上界参考，不冒充 PSS），并非每个样本各一份内存上限。
- 默认检查 CUDA；`--no-gpu` 强制 CPU。COMEBin CPU 未显式指定 batch 时用 128，
  保留每 20 分钟训练存活/进度日志。

公共 bin 名为 `样本名_binner_序号.fa`，例如 `A_clean_metabat2_1.fa`。
多样本组也以目标 contigs 所属样本名命名，全部 reads 来源写入清单。
共组装 bin 使用 `分组名_全部成员样本名_binner_序号.fa`，成员名以短横线连接。
未改变的精炼 MAG 保留来源 binner 名称，改变/合并的 MAG 使用对应 refiner 名称。

- `bin_files/<binner>/`：各 binner 结果。
- `quality_control_files/`：质量与可选 RNA/GUNC 报告。
- `non_redundant_bins/`：最终 MAG。
- `start_info.txt`：binning 和 annotation 的详细启动信息，包括逐样本配置、软件环境、
  CUDA/GPU 检查、CPU 回退、资源预算、启动警告及失败策略。同一结果目录重跑时按时间追加。
  终端保留输入/分组概况、该文件路径、执行进度、交互提示和致命错误；运行期警告仍正常显示。
- `memory_diagnostics.jsonl`：内存计量异常、疑似超限或首次发现无用户地址空间的进程时按需生成，
  追加记录 PID、PPID、进程名、内核状态、PSS、RSS、测量来源及排除原因；区分真实超限与无法可靠计量。
- `run_manifest.json`：完整命令、具名输入、逐样本类型/工具、组成员、任务依赖和资源；
  同时保留输入清单原文与 SHA256、解析后的参数、数据库路径、软件版本及检测来源、
  MetaBAW 源码 SHA256，便于区分同为 0.1.0 的不同安装包。未识别的版本标记为 `not detected`。
- `workflow_details.md`：binning 和 annotation 在输入、软件与数据库预检通过后、首个任务执行前自动生成。
  按 `# Step1`、步骤名称、`Software: 软件名 (关键参数)` 简洁列出实际选用的流程。
  相同软件/参数的多样本记录合并，保留阈值和不同配置；生态位跳过、分类结果复用会注明。
  它记录本次计划；完整复现信息见 `run_manifest.json`，完成情况以日志和最终汇总为准。
- `resolved_parameters.tsv`：最终解析参数。
- `binner_summary.tsv`：每目标/每请求工具的状态、类型、排除或失败原因、日志和输出。
- `workflow_summary.json`：最终运行状态。
- `tmp/runtime/logs/`：逐任务日志。

同一输入中断后，用原命令和原输出目录重跑即可复用完整输出。现有执行器优先依据输出完整性恢复；
**更改样本清单、分组、输入内容或参数时必须使用新的输出及临时目录**，以免复用旧结果。
本版初次运行也不要复用旧版结果目录。默认临时目录为 `<output>/tmp`。

## 依赖检查

```bash
metabaw check -h
metabaw check --scope binning
metabaw check --scope comebin
metabaw check --scope lorbin
```

binning 启动前按全体样本实际需要的工具集合检查依赖。数据库路径通过 check 配置，
如 `--checkm2-db PATH`、`--gunc-db PATH`、`--gtdbtk-db PATH`；其他选项见 check 帮助。
安装包不包含外部分箱工具或数据库。本包默认隔离环境：

| 工具 | 默认环境 |
|---|---|
| COMEBin | metabaw-comebin-py37 |
| CheckM2 | metabaw-checkm2-py312 |
| MetaWRAP | metabaw-metawrap-py27 |
| LorBin | metabaw-lorbin-py310 |

## Annotation 接口说明

annotation 现在也使用清单输入，移除 `-p/--path`、`-r/--reads`、`-s/--suffix`、
`-f/--read-suffix`、`--type`、`--separate-sample-name`，不接受参数缩写。
check 接口保持不变。

`--input_reads_files reads.tsv` 与 binning **完全共用同一个解析函数和格式**：

```text
#sample	reads
S1	/my/path/to/genome/reads1.1.fastq.gz,/my/path/to/genome/reads1.2.fastq.gz
S2	/my/path/to/genome/reads2.fastq
```

两列由 Tab 分隔，双端两文件以逗号分隔；单文件视为长读长。两者可出现在同一清单中。
CoverM 按样本使用 `--coupled` + `minimap2-sr` 或 `--single` + `minimap2-ont`。
丰度文件及汇总表使用第一列的具名样本名，不使用 reads 文件名推断。

`--input_genome_files genomes.txt` 每行仅一个 genome FASTA 路径，**没有样本名列**：

```text
/my/path/to/genome/genome1.fa
/another/path/to/genome2.fna.gz
```

两个清单均支持 UTF-8/BOM、注释和空行，相对路径按各自清单所在目录解析。
genome 数量不要求与 reads 样本数一致，每个 reads 样本对全体列出的 MAG 计算丰度。
MAG 支持 `.fa`、`.fna`、`.fasta` 及其 gzip 文件。去掉完整 FASTA 后缀后生成 MAG 名称，
不同目录中同名 MAG（忽略大小写）会提前报错，而不是覆盖；文件名需为唯一、可安全用于输出的
字母/数字/点/下划线/短横线组合，不能含重复 FASTA 后缀。目录名可含空格。

只整理清单中列出的 MAG 到 `tmp/work/annotation/genomes/`，统一为未压缩 `.fa`；
原始 FASTA 不修改，需预留这份未压缩副本的磁盘空间。目录中同时生成来源 `manifest.tsv`。
GTDB-Tk、CoverM 和 Prodigal 均依赖这个准备步骤；复用 GTDB-Tk 结果时，其他步骤仍正常准备 MAG。
输入来源、MAG 名称及逐样本比对方式还会记录在结果根目录的 `run_manifest.json`。

niche 仅在输入 genome 数量和具名 reads 样本数都大于 3（均至少 4）时启用。
任一数量不超过 3（包括正好 3）时自动跳过，不再因为样本不足而报错，其他注释继续。
每对双端文件只算一个 reads 样本；genome 数量按清单中列出的 MAG 文件数计算。
终端和结果根目录 `niche_status.log` 会记录数量与跳过原因；`run_manifest.json` 中保存同样的决策。
显式 `--no-niche` 始终禁用，即使两个数量均达标。
改变任一清单、输入内容或分析参数后，请使用新的输出和临时目录，避免复用旧结果。

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
  --input_genome_files genomes.txt \
  --input_reads_files reads.tsv \
  --methods relative_abundance rpkm tpm mean \
  --niche-rank family \
  --niche_classify_method cv --niche_abundance relative_abundance \
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
Prodigal gene number. If contigs were already renamed by `metabaw binning`, the raw
and normalized contig columns are identical. `Gene` contains the KO, CAZy
family/result, hydrogenase class, or terminal-enzyme name, depending on the
table. Tool-specific evidence columns follow these five common columns.
For `kegg/kegg_annotations.tsv`, `KO_gene_name` contains the gene name or
comma-separated aliases before the first semicolon in `K.descript.txt`, and
`KO_description` retains the complete description, including enzyme names and
EC identifiers. Every reported KO must have a description; an incomplete
description table produces an explicit error listing the missing KO IDs.
Missing combined-search outputs invalidate the corresponding functional task during resume.

GTDB-Tk species placement is enabled by default. Use `--no-place-species` to
disable it. `--place-species` and `--no-place-species` are mutually exclusive;
supplying both is rejected before the workflow starts. Both options are listed
in `metabaw annotation -h`.
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
relative_abundance  rpkm  tpm  mean
```

The CLI accepts only these four abundance methods, including comma-separated
lists. `count` and other estimators are no longer accepted by `--methods`.
CoverM estimators that do not calculate covered bases (`rpkm`, `tpm`,
and internal niche-support read counts) are calculated together with
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
Generated GTDB-Tk classification uses a 160 GiB preflight memory floor; this
is not an allocation, a predicted peak, or a guarantee of sufficient memory. MetaBAW forces
`--pplacer_cpus 1` and enables GTDB-Tk's disk-backed `--scratch_dir`; this
limits pplacer concurrency. The default `--max-memory` is the detected machine
or cgroup memory total. A detected or explicitly configured value below 160 GiB
is accepted only when a complete matching `--gtdbtk_res` skips the GTDB-Tk task.
All GTDB-Tk classification phases additionally use
`min(-t, --gtdbtk-threads)` CPUs; `--gtdbtk-threads` defaults to 16, independently
of the CPU budget available to other annotation steps. Both GTDB-Tk options are
listed in `annotation -h`.

On Linux, subprocess/IPC temporary files use a unique real directory under
`--gtdbtk-tmpdir` (default `/tmp`), not a symlink to the workflow's NFS directory.
Known NFS/CIFS/SSHFS mounts are rejected; choose a writable short local path.
Private temporary files are cleaned on normal return or Python exceptions;
forced process termination can leave the printed private directory for inspection.
Large pplacer scratch files remain in the workflow workspace. Species-placement,
taxonomy, abundance and niche criteria are unchanged.

metaBAW writes one matrix per metric. GTDB-Tk domain, phylum, class, order,
family, genus, and species columns are inserted after `Genome`. A MAG without
a GTDB-Tk species assignment is reported as `unclassified`; its MAG name is
not copied into the species column.

Supported niche levels are:

```text
domain phylum class order family genus species strain
```

The default niche level is `family`. `--niche_classify_method cv` is the default and
classifies eligible taxa from the corrected coefficient of variation
`CV - sqrt(K/N)` relative to the eligible-community mean. Use
`--niche_classify_method occupancy` to classify taxa from occupancy instead.
The old `--niche-method prevalence` spelling remains a hidden compatibility
alias; new commands and recorded criteria use `occupancy`.

`--niche_abundance` selects `relative_abundance` (default), `rpkm`, `tpm`, or
`mean`, independently of the criterion. When niche is enabled, the selected
metric must be included in `--methods`; otherwise preflight reports the missing
metric instead of silently substituting another one. Both new niche options
are shown in `annotation -h`.

CV is calculated from the selected taxon's native abundance values across
samples, with the existing `sqrt(K/N)` correction and minimum total support
of 20 mapped reads. Relative abundance keeps its CoverM percentage scale.
For occupancy detection, RPKM, TPM and mean are converted to percentages of
the summed selected MAG abundance within each sample; this makes the fixed
0.01% detection threshold meaningful across units. These normalized
percentages are used only for detection, not substituted into the CV input.
The default occupancy rule requires >=20% sample occupancy for classification,
and calls taxa with >=80% occupancy generalists (others passing support are
specialists). The direct annotation workflow treats its input samples as one
dataset. Zero/insufficient-support taxa are reported as unclassified rather
than aborting the entire annotation.

Read counts are calculated internally only when niche is enabled. The support
profiles and combined matrix are stored under `tmp/work/niche_support/`, not
published as `coverm_counts.tsv` or count columns in `coverm_all_metrics.tsv`.
The niche diagnostic table retains explicitly labelled internal read counts.
`run_manifest.json`, `resolved_parameters.tsv`, `niche_status.log` and niche
provenance record the selected criterion/metric and calculation rules.
Non-relative-abundance CV is a MetaBAW extension of the existing correction,
not an exact reproduction of the original study's abundance scale.
Changing abundance methods, niche criterion/abundance or the niche enable
decision requires a new annotation output directory. If existing abundance or
niche outputs use different recorded settings, preflight refuses to reuse them.

Niche classification is enabled only when both the input genome count and named
read-sample count exceed three (at least four of each). A paired-end read pair
counts as one sample. If either count is three or fewer, MetaBAW skips niche
classification, logs the counts and reason in the console and `niche_status.log`,
and continues the other annotation stages. Explicit `--no-niche` always disables it.
All four niche files are checked during resume, so a missing classification,
taxon-abundance, genome-assignment, or provenance file causes only the niche
step to run again.


## Annotation output

```text
metabaw_annotation_result/
|-- gtdbtk_result/
|-- coverm/
|   |-- sampleA.tsv
|   |-- sampleB.tsv
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
must remain the same when resuming.

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

`--max-memory` is available in both modules. Its unit is GiB; both modules default
to the physical memory total detected from Linux `/proc/meminfo`, POSIX sysconf,
or the Windows memory API. A smaller readable Linux cgroup limit takes precedence;
if detection is unavailable, the conservative fallback is 160 GiB. Annotation's
generated GTDB-Tk task still uses one pplacer CPU, disk scratch, and a 160 GiB
preflight floor. All concurrently running MetaBAW task process trees share this
one memory budget. On Linux, MetaBAW reads the
process hierarchy and `Pss` from `/proc/<pid>/smaps_rollup`, falling back to
summed `Pss` fields from `smaps`. PID start times are checked around measurement;
exited, zombie and recycled PIDs are excluded. A still-visible PID whose
`smaps_rollup` returns ESRCH and whose readable `smaps` is completely empty is
recorded as `no_address_space`, not as an unmeasurable live process. This exclusion
applies only to that snapshot: descendants remain monitored and the PID is
resampled on the next poll. Permission errors, missing files alone, or malformed
nonempty memory maps do not qualify. PSS apportions shared resident
pages across the processes mapping them. `VmRSS` is retained only as a conservative
upper bound, never relabeled as PSS. `--task 6` does not create six
independent allowances. The configured value is retained as a secondary
per-process virtual-address guard where applicable. Two consecutive above-limit
PSS samples confirm a breach; new tasks are held during confirmation. A task's
explicit CPU-memory allocation failure still triggers immediate protection.
MetaBAW prints the confirmed measurement in a `[MEMORY LIMIT]` workflow event,
disables retries, terminates every process group started by the workflow, and
blocks all pending tasks. CUDA out-of-memory errors remain separate because
they are governed by `--max-gpu-memory`.

When PSS is unavailable but a complete conservative bound remains below the limit,
execution can continue. If the budget cannot be verified on two consecutive
samples, MetaBAW stops safely with `[MEMORY ACCOUNTING ERROR]`, explicitly NOT
claiming a confirmed PSS breach. `memory_diagnostics.jsonl` in the result directory
records per-process measurements/sources, excluded PIDs and confirmation events.
If both PSS and RSS are missing for any live process, the available sum is labeled
`accounted_bytes`, not a total upper bound; `upper_bound_bytes` is null and
`upper_bound_complete` is false. The run manifest likewise marks an incomplete
peak upper bound as null. This does not disable the memory guard or count unknown
memory as zero.
These measurements are observations, not predictions of required memory.

Before any workflow task starts, MetaBAW records `[MEMORY ESTIMATE]` in `start_info.txt` with the
estimated minimum total memory, configured limit, and headroom or shortfall.
The estimate is a conservative floor derived from declared high-memory task
requirements; data-dependent additional usage is still watched at runtime. If
a known minimum already exceeds `--max-memory`, MetaBAW writes
`tmp/runtime/logs/memory_preflight.log`, prints `[MEMORY CONFIG ERROR]` with an
actionable correction, and exits without starting a workflow process. If a
data-dependent runtime peak crosses the limit instead, the final `[ERROR]`
summary repeats the observed PSS and configured ceiling rather than reporting
only blocked task IDs.


## 开发验证

```bash
PYTHONPATH=src python -m unittest discover -s tests
python -m build --wheel --no-isolation --outdir .package_build
```

测试覆盖输入解析、混合工具路由、全组覆盖度依赖、单成员类型子组、具名 FASTA/BAM 输出、
错误参数拒绝及既有功能回归。本地测试不代表已完成外部分箱工具或 GPU 的真实数据运行。
