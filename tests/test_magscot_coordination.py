from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from metabaw.direct import BinOptions, DirectBinBuilder
from metabaw.discovery import Analysis, ReadSample
from metabaw.internal import finalize_magscot_mapping
from metabaw.model import Task


class MAGScoTCoordinationTests(unittest.TestCase):
    def test_single_available_binner_is_preserved_after_score_only_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            combined = root / "all_binners.tsv"
            combined.write_text(
                "bin.1\tcontig_1\tmetabat2\n"
                "bin.1\tcontig_2\tmetabat2\n",
                encoding="utf-8",
            )
            mapping = root / "sample.refined.contig_to_bin.out"

            finalize_magscot_mapping(combined, mapping)

            with mapping.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.reader(handle, delimiter="\t"))
            self.assertEqual(
                rows,
                [
                    ["binnew", "contig"],
                    ["bin.1", "contig_1"],
                    ["bin.1", "contig_2"],
                ],
            )

    def test_multiple_binners_without_refined_output_raise_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            combined = root / "all_binners.tsv"
            combined.write_text(
                "bin.1\tcontig_1\tmetabat2\n"
                "bin.2\tcontig_2\tvamb\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "no bins met the configured refinement thresholds across 2 binners",
            ):
                finalize_magscot_mapping(combined, root / "missing.tsv")

    def test_existing_refined_mapping_is_validated_and_retained(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            combined = root / "all_binners.tsv"
            combined.write_text(
                "bin.1\tcontig_1\tmetabat2\n"
                "bin.2\tcontig_2\tvamb\n",
                encoding="utf-8",
            )
            mapping = root / "sample.refined.contig_to_bin.out"
            expected = "binnew\tcontig\nrefined.1\tcontig_1\n"
            mapping.write_text(expected, encoding="utf-8")

            finalize_magscot_mapping(combined, mapping)

            self.assertEqual(mapping.read_text(encoding="utf-8"), expected)

    def test_refined_bins_wait_for_successful_magscot_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contigs = root / "contigs.fa"
            read1 = root / "reads.1.fastq.gz"
            read2 = root / "reads.2.fastq.gz"
            sample = ReadSample("A606", read1, read2, contigs)
            analysis = Analysis("A606", (sample,), (contigs,), False)
            options = BinOptions(
                outdir=root / "results",
                workdir=root / "results" / "tmp" / "work",
                threads=8,
                read_type="short",
                align_tool="bowtie2",
                flye_read_type="--nano-raw",
                binners=("metabat2",),
                min_contig_length=1500,
                min_fasta_kbs=200,
                batch_size=1024,
                refiner="magscot",
                quality_control="checkm2",
                min_completeness=50.0,
                max_contamination=10.0,
                min_quality_score=None,
                run_gunc=False,
                run_trna=False,
                trna_pass=None,
                run_rrna=False,
                rrna_pass=False,
                dereplicator="galah",
                environment=None,
                tag_contigs=False,
                gpu=False,
                max_gpu_memory="4G",
                assembly_strategy="default",
                magscot_dir=root / "MAGScoT",
                checkm2_db=None,
                gunc_db=None,
                gtdbtk_data=None,
                extra_args={},
            )
            builder = DirectBinBuilder([], options, root)
            prepare = Task("01.prepare.A606", "01_prepare", "true", root)
            map_task = Task("03.publish.metabat2.A606", "03_binning", "true", root)
            mapping = root / "metabat2.tsv"
            public_bins = root / "results" / "bin_files" / "metabat2" / "A606"

            result, _bins = builder._refine(
                analysis,
                prepare,
                contigs,
                [("metabat2", map_task, mapping, public_bins)],
            )
            refine = next(
                task for task in builder.tasks if task.id == "04.refine.magscot.A606"
            )

            self.assertIn("finalize-magscot", refine.display_command())
            self.assertIn("rm -f", refine.display_command())
            self.assertEqual(result.deps, (refine.id,))
            self.assertFalse(result.allow_failed_deps)


if __name__ == "__main__":
    unittest.main()
