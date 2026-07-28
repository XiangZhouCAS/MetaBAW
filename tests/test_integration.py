from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest

from metabaw.direct import AnnotationBuilder, AnnotationOptions, BinOptions, DirectBinBuilder
from metabaw.discovery import Analysis, ReadSample
from metabaw.executor import Executor
from metabaw.model import Task
from metabaw.state import StateStore


def simulated_task(task: Task) -> Task:
    paths = list(task.outputs)
    if task.output_alternatives:
        paths.extend(task.output_alternatives[0])
    directories = {
        path
        for path in paths
        if path.suffix == ""
        or any(path != other and path in other.parents for other in paths)
    }
    script = (
        "from pathlib import Path; "
        f"directories={[str(path) for path in directories]!r}; "
        f"files={[str(path) for path in paths if path not in directories]!r}; "
        f"fasta_directories={[str(path) for path in task.fasta_output_dirs]!r}; "
        "[(Path(path).mkdir(parents=True, exist_ok=True), "
        "(Path(path) / '.simulated').write_text('complete\\n')) "
        "for path in directories]; "
        "[(Path(path).mkdir(parents=True, exist_ok=True), "
        "(Path(path) / 'bin.1.fa').write_text('>contig\\nACGT\\n')) "
        "for path in fasta_directories]; "
        "[(Path(path).parent.mkdir(parents=True, exist_ok=True), "
        "Path(path).write_text('simulated\\n')) for path in files]"
    )
    return replace(task, command=(sys.executable, "-c", script))


class SimulatedWorkflowIntegrationTests(unittest.TestCase):
    def test_complete_six_binner_workflow_dag_executes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            contigs = root / "A606.contig.ok.fa"
            read1 = root / "A606.clean_R1.fastq.gz"
            read2 = root / "A606.clean_R2.fastq.gz"
            contigs.write_text(">contig\n" + "A" * 2000 + "\n", encoding="utf-8")
            read1.write_bytes(b"")
            read2.write_bytes(b"")
            sample = ReadSample("A606.clean", read1, read2, contigs)
            analysis = Analysis("A606.clean", (sample,), (contigs,), False)
            self.assertEqual("A606", sample.name)
            self.assertEqual("A606", analysis.name)
            options = BinOptions(
                outdir=root / "result",
                workdir=root / "result" / "tmp" / "work",
                threads=4,
                read_type="short",
                align_tool="bowtie2",
                long_read_preset="map-ont",
                binners=(
                    "metabat2",
                    "metadecoder",
                    "vamb",
                    "comebin",
                    "semibin2",
                    "lorbin",
                ),
                min_contig_length=1500,
                min_fasta_kbs=200,
                batch_size=1024,
                refiner="magscot",
                quality_control="checkm2",
                min_completeness=50,
                max_contamination=10,
                min_quality_score=None,
                run_gunc=True,
                run_trna=True,
                trna_pass=18,
                run_rrna=True,
                rrna_pass=True,
                dereplicator="galah",
                environment="global",
                tag_contigs=True,
                gpu=False,
                max_gpu_memory="4G",
                magscot_dir=root / "MAGScoT",
                checkm2_db=root / "checkm2.dmnd",
                gunc_db=root / "gunc.dmnd",
                gtdbtk_data=root / "gtdbtk",
                extra_args={},
            )
            tasks = [
                simulated_task(task)
                for task in DirectBinBuilder([analysis], options, options.workdir).build()
            ]
            self.assertTrue(all("clean" not in task.id.lower() for task in tasks))
            generated_paths = [
                path
                for task in tasks
                for path in (
                    *task.outputs,
                    *(path for alternatives in task.output_alternatives for path in alternatives),
                )
            ]
            self.assertTrue(
                all("clean" not in str(path.relative_to(root)).lower() for path in generated_paths)
            )
            state = StateStore(root / "result" / "tmp" / "runtime" / "state.sqlite3")
            try:
                statuses = Executor(
                    state,
                    root / "result" / "tmp" / "runtime" / "logs",
                    max_cpus=4,
                    max_parallel=4,
                    fail_fast=False,
                ).run(tasks)
            finally:
                state.close()
            self.assertTrue(statuses)
            self.assertEqual({"success"}, set(statuses.values()))
            self.assertTrue((root / "result" / "non_redundant_bins").exists())

    def test_complete_annotation_workflow_dag_executes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            mags = root / "mags"
            mags.mkdir()
            (mags / "bin.fna").write_text(">contig\nAAAA\n", encoding="utf-8")
            reads = []
            for name in ("A", "B"):
                path = root / f"{name}.fastq.gz"
                path.write_bytes(b"")
                reads.append(ReadSample(name, path))
            options = AnnotationOptions(
                mag_dir=mags,
                mag_suffix="fna",
                reads=tuple(reads),
                output=root / "annotation",
                output_suffix=".tsv",
                threads=4,
                read_type="short",
                methods=("relative_abundance", "count"),
                place_species=True,
                niche_rank="family",
                no_niche=False,
                gtdbtk_data=root / "gtdbtk",
            )
            work = root / "annotation" / "tmp" / "work"
            tasks = [
                simulated_task(task)
                for task in AnnotationBuilder(options, work).build()
            ]
            state = StateStore(root / "annotation" / "tmp" / "runtime" / "state.sqlite3")
            try:
                statuses = Executor(
                    state,
                    root / "annotation" / "tmp" / "runtime" / "logs",
                    max_cpus=4,
                    max_parallel=4,
                    fail_fast=False,
                ).run(tasks)
            finally:
                state.close()
            self.assertEqual({"success"}, set(statuses.values()))
            self.assertTrue(
                (root / "annotation" / "niche" / "niche_classification.tsv").is_file()
            )


if __name__ == "__main__":
    unittest.main()
