from pathlib import Path
import csv
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from metabaw.internal import (
    _run_logged,
    _rank_taxon,
    bins_to_map,
    classify_niche_literature,
    collect_fasta,
    concatenate_fastas,
    fasta_filter,
    filter_quality,
    iter_fasta,
    materialize_bins,
    normalize_refined_fasta,
    publish_fasta,
    rna_qc,
)


class FastaTests(unittest.TestCase):
    def test_rna_qc_rejects_an_output_directory_containing_candidate_bins(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            candidates = root / "quality_control_files" / "candidate_bins"
            candidates.mkdir(parents=True)
            (candidates / "bin.fa").write_text(">c\nAAAA\n", encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError,
                "would delete the input bins directory",
            ):
                rna_qc(
                    candidates,
                    root / "markers",
                    root / "quality_control_files",
                    root / "quality_control_files" / "rna_quality.tsv",
                    "tRNAscan-SE",
                    "barrnap",
                    4,
                    False,
                    None,
                    True,
                    True,
                )

    def test_rna_qc_preserves_candidate_and_checkm2_directories(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            quality = root / "quality_control_files"
            candidates = quality / "candidate_bins"
            protein_files = quality / "checkm2" / "protein_files"
            markers = root / "markers"
            candidates.mkdir(parents=True)
            protein_files.mkdir(parents=True)
            markers.mkdir()
            candidate = candidates / "bin.fa"
            protein = protein_files / "bin.faa"
            candidate.write_text(">c\nAAAA\n", encoding="utf-8")
            protein.write_text(">p\nAAAA\n", encoding="utf-8")
            commands: list[list[str]] = []

            def fake_run(command: list[str], output: Path) -> None:
                commands.append(command)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(
                    "c\tbarrnap\trRNA\t1\t4\t.\t+\t.\t"
                    "Name=5S_rRNA\n"
                    "c\tbarrnap\trRNA\t1\t4\t.\t+\t.\t"
                    "Name=16S_rRNA\n"
                    "c\tbarrnap\trRNA\t1\t4\t.\t+\t.\t"
                    "Name=23S_rRNA\n",
                    encoding="utf-8",
                )

            summary = quality / "rna_quality.tsv"
            completion = quality / "rna" / "rna_qc.complete"
            with patch("metabaw.internal._run_logged", side_effect=fake_run):
                rna_qc(
                    candidates,
                    markers,
                    quality / "rna",
                    summary,
                    "tRNAscan-SE",
                    "barrnap",
                    4,
                    False,
                    None,
                    True,
                    True,
                    completion,
                )

            self.assertTrue(candidate.is_file())
            self.assertTrue(protein.is_file())
            self.assertTrue((quality / "rna" / "rRNA" / "bin.gff").is_file())
            self.assertTrue(completion.is_file())
            self.assertIn("\ttrue\t\n", summary.read_text(encoding="utf-8"))
            self.assertNotIn("--outseq", commands[0])
            self.assertEqual(str(candidate), commands[0][-1])
            self.assertEqual(
                "4",
                commands[0][commands[0].index("--threads") + 1],
            )

    def test_run_logged_reports_real_stderr_separately(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            output = root / "result.gff"
            with self.assertRaisesRegex(RuntimeError, "specific barrnap failure"):
                _run_logged(
                    [
                        sys.executable,
                        "-c",
                        (
                            "import sys; "
                            "print('partial stdout'); "
                            "print('specific barrnap failure', file=sys.stderr); "
                            "raise SystemExit(2)"
                        ),
                    ],
                    output,
                )

            self.assertEqual(
                "partial stdout\n",
                output.read_text(encoding="utf-8"),
            )
            self.assertIn(
                "specific barrnap failure",
                (root / "result.gff.stderr.log").read_text(encoding="utf-8"),
            )

    def test_rna_prediction_only_reports_absence_without_filtering(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            candidates = root / "candidate_bins"
            markers = root / "markers"
            candidates.mkdir()
            markers.mkdir()
            (candidates / "bin.fa").write_text(">c\nAAAA\n", encoding="utf-8")

            def fake_run(_command: list[str], output: Path) -> None:
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text("", encoding="utf-8")

            summary = root / "rna_quality.tsv"
            with patch("metabaw.internal._run_logged", side_effect=fake_run):
                rna_qc(
                    candidates,
                    markers,
                    root / "rna",
                    summary,
                    "tRNAscan-SE",
                    "barrnap",
                    2,
                    False,
                    None,
                    True,
                    False,
                )

            with summary.open("r", encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual("true", row["pass"])
            self.assertEqual("false", row["rrna_5S"])
            self.assertEqual("false", row["rrna_16S"])
            self.assertEqual("false", row["rrna_23S"])

    def test_rrna_pass_enables_hard_filtering(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            candidates = root / "candidate_bins"
            markers = root / "markers"
            candidates.mkdir()
            markers.mkdir()
            (candidates / "bin.fa").write_text(">c\nAAAA\n", encoding="utf-8")

            def fake_run(_command: list[str], output: Path) -> None:
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text("", encoding="utf-8")

            summary = root / "rna_quality.tsv"
            with patch("metabaw.internal._run_logged", side_effect=fake_run):
                rna_qc(
                    candidates,
                    markers,
                    root / "rna",
                    summary,
                    "tRNAscan-SE",
                    "barrnap",
                    2,
                    False,
                    None,
                    False,
                    True,
                )

            with summary.open("r", encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual("false", row["pass"])

    def test_rna_qc_fails_when_every_requested_tool_invocation_fails(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            candidates = root / "candidate_bins"
            markers = root / "markers"
            candidates.mkdir()
            markers.mkdir()
            (candidates / "bin.fa").write_text(">c\nAAAA\n", encoding="utf-8")
            summary = root / "rna_quality.tsv"
            completion = root / "rna" / "rna_qc.complete"

            with patch(
                "metabaw.internal._run_logged",
                side_effect=RuntimeError("tool unavailable"),
            ), self.assertRaisesRegex(
                RuntimeError,
                "Every requested RNA tool invocation failed",
            ):
                rna_qc(
                    candidates,
                    markers,
                    root / "rna",
                    summary,
                    "tRNAscan-SE",
                    "barrnap",
                    2,
                    False,
                    None,
                    True,
                    False,
                    completion,
                )

            self.assertTrue(summary.is_file())
            self.assertFalse(completion.exists())

    def test_rna_qc_records_one_bin_failure_and_continues(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            candidates = root / "candidate_bins"
            markers = root / "markers"
            candidates.mkdir()
            markers.mkdir()
            good = candidates / "good.fa"
            bad = candidates / "bad.fa"
            good.write_text(">good\nAAAA\n", encoding="utf-8")
            bad.write_text(">bad\nAAAA\n", encoding="utf-8")

            def fake_run(command: list[str], output: Path) -> None:
                if command[-1] == str(bad):
                    raise RuntimeError("simulated Barrnap failure")
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(
                    "c\tbarrnap\trRNA\t1\t4\t.\t+\t.\tName=5S_rRNA\n"
                    "c\tbarrnap\trRNA\t1\t4\t.\t+\t.\tName=16S_rRNA\n"
                    "c\tbarrnap\trRNA\t1\t4\t.\t+\t.\tName=23S_rRNA\n",
                    encoding="utf-8",
                )

            summary = root / "rna_quality.tsv"
            completion = root / "rna" / "rna_qc.complete"
            with patch("metabaw.internal._run_logged", side_effect=fake_run):
                rna_qc(
                    candidates,
                    markers,
                    root / "rna",
                    summary,
                    "tRNAscan-SE",
                    "barrnap",
                    4,
                    False,
                    None,
                    True,
                    True,
                    completion,
                )

            with summary.open("r", encoding="utf-8", newline="") as handle:
                rows = {
                    row["genome"]: row
                    for row in csv.DictReader(handle, delimiter="\t")
                }
            self.assertEqual("true", rows["good"]["pass"])
            self.assertEqual("", rows["good"]["error"])
            self.assertEqual("false", rows["bad"]["pass"])
            self.assertIn("simulated Barrnap failure", rows["bad"]["error"])
            self.assertTrue(completion.is_file())

    def test_rna_qc_parallelizes_bins_within_the_thread_budget(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            candidates = root / "candidate_bins"
            markers = root / "markers"
            candidates.mkdir()
            markers.mkdir()
            for number in range(4):
                (candidates / f"bin{number}.fa").write_text(
                    f">c{number}\nAAAA\n",
                    encoding="utf-8",
                )

            barrier = threading.Barrier(4, timeout=2)
            commands: list[list[str]] = []
            command_lock = threading.Lock()

            def fake_run(command: list[str], output: Path) -> None:
                with command_lock:
                    commands.append(command)
                barrier.wait()
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(
                    "c\tbarrnap\trRNA\t1\t4\t.\t+\t.\tName=5S_rRNA\n"
                    "c\tbarrnap\trRNA\t1\t4\t.\t+\t.\tName=16S_rRNA\n"
                    "c\tbarrnap\trRNA\t1\t4\t.\t+\t.\tName=23S_rRNA\n",
                    encoding="utf-8",
                )

            summary = root / "rna_quality.tsv"
            with patch("metabaw.internal._run_logged", side_effect=fake_run):
                rna_qc(
                    candidates,
                    markers,
                    root / "rna",
                    summary,
                    "tRNAscan-SE",
                    "barrnap",
                    4,
                    False,
                    None,
                    True,
                    True,
                    root / "rna" / "rna_qc.complete",
                )

            self.assertEqual(4, len(commands))
            self.assertTrue(
                all(
                    command[command.index("--threads") + 1] == "1"
                    for command in commands
                )
            )
            with summary.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(4, len(rows))
            self.assertTrue(all(row["pass"] == "true" for row in rows))

    def test_concatenate_fastas_supports_semibin_sample_separator(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            first = root / "site-A.fa"
            second = root / "site-B.fa"
            first.write_text(">contig_1\nAAAA\n", encoding="utf-8")
            second.write_text(">contig_2\nCCCC\n", encoding="utf-8")
            output = root / "combined.fa"

            concatenate_fastas(
                [f"site-A={first}", f"site-B={second}"],
                output,
                1,
                ":",
            )

            self.assertEqual(
                ["site-A:contig_1", "site-B:contig_2"],
                [identifier for identifier, _header, _sequence in iter_fasta(output)],
            )

    def test_all_requested_taxonomic_levels(self) -> None:
        classification = "d__Bacteria;p__P;c__C;o__O;f__F;g__G;s__G species"
        expected = {
            "phylum": "p__P",
            "class": "c__C",
            "order": "o__O",
            "family": "f__F",
            "genus": "g__G",
            "species": "s__G species",
            "strain": "MAG001",
        }
        for rank, value in expected.items():
            with self.subTest(rank=rank):
                self.assertEqual(_rank_taxon("MAG001", classification, rank), value)

    def test_filter_and_materialize(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "contigs.fna"
            source.write_text(
                ">contig1 description\nAAAAAA\n>contig2\nCCCC\n>contig3\nGGGGGGG\n",
                encoding="utf-8",
            )
            filtered = root / "filtered.fna"
            fasta_filter(source, filtered, 5)
            self.assertEqual([record[0] for record in iter_fasta(filtered)], ["contig1", "contig3"])

            mapping = root / "mapping.tsv"
            mapping.write_text("binA\tcontig1\nbinB\tcontig3\n", encoding="utf-8")
            bins = root / "bins"
            materialize_bins(filtered, mapping, bins, "asm")
            self.assertTrue((bins / "manifest.tsv").is_file())
            self.assertEqual(
                ["asm_1.fa", "asm_2.fa"],
                sorted(path.name for path in bins.glob("*.fa")),
            )

    def test_materialized_names_do_not_retain_internal_bin_identifiers(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            assembly = root / "A606.clean.fna"
            assembly.write_text(">c1\nAAAA\n>c2\nCCCC\n", encoding="utf-8")
            mapping = root / "mapping.tsv"
            mapping.write_text(
                "A606.clean_cleanbin_000030\tc1\n"
                "A606.clean_cleanbin_000005\tc2\n",
                encoding="utf-8",
            )
            output = root / "bins"
            materialize_bins(assembly, mapping, output, "A606_MAGScoT")
            self.assertEqual(
                ["A606_MAGScoT_1.fa", "A606_MAGScoT_2.fa"],
                sorted(path.name for path in output.glob("*.fa")),
            )
            self.assertNotIn(
                "cleanbin",
                "\n".join(path.name for path in output.iterdir()),
            )

    def test_publish_fasta_uses_sample_tool_sequence_fa_names(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source"
            source.mkdir()
            (source / "bin.10.fna").write_text(">a\nAAAA\n", encoding="utf-8")
            (source / "bin.20.fa").write_text(">b\nCCCC\n", encoding="utf-8")
            output = root / "published"
            publish_fasta(source, output, "A606_DASTool")
            self.assertEqual(
                ["A606_DASTool_1.fa", "A606_DASTool_2.fa"],
                sorted(path.name for path in output.glob("*.fa")),
            )

    def test_dastool_normalization_retains_source_tools_and_matches_catalog(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            metabat2 = root / "metabat2"
            vamb = root / "vamb"
            refined = root / "S_DASTool_bins"
            for directory in (metabat2, vamb, refined):
                directory.mkdir()
            (metabat2 / "S_MetaBAT2_1.fa").write_text(
                ">c1\nAAAA\n",
                encoding="utf-8",
            )
            (vamb / "S_VAMB_1.fa").write_text(
                ">c2\nCCCC\n",
                encoding="utf-8",
            )
            (refined / "metabat2__S.MetaBAT2.1.fa").write_text(
                ">c1\nAAAA\n",
                encoding="utf-8",
            )
            (refined / "vamb__S.VAMB.1.fa").write_text(
                ">c2\nCCCC\n>c3\nGGGG\n",
                encoding="utf-8",
            )

            normalize_refined_fasta(
                refined,
                refined,
                "S_DASTool",
                [metabat2, vamb],
            )
            expected = ["S_DASTool_1.fa", "S_MetaBAT2_1.fa"]
            self.assertEqual(
                expected,
                sorted(path.name for path in refined.glob("*.fa")),
            )
            self.assertFalse(
                any("__" in path.name for path in refined.glob("*.fa"))
            )
            manifest = (refined / "manifest.tsv").read_text(encoding="utf-8")
            self.assertIn("S_MetaBAT2_1.fa\tunchanged", manifest)
            self.assertIn("S_DASTool_1.fa\trefined", manifest)

            candidates = root / "quality_control_files" / "candidate_bins"
            collect_fasta([refined], candidates)
            self.assertEqual(
                expected,
                sorted(path.name for path in candidates.glob("*.fa")),
            )
            self.assertTrue((candidates / "manifest.tsv").is_file())
            for name in expected:
                self.assertEqual(
                    (refined / name).read_bytes(),
                    (candidates / name).read_bytes(),
                )

    def test_metawrap_normalization_retains_source_tools_and_matches_catalog(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            metabat2 = root / "metabat2"
            metadecoder = root / "metadecoder"
            refined = root / "metawrap_50_10_bins"
            for directory in (metabat2, metadecoder, refined):
                directory.mkdir()
            (metabat2 / "S_MetaBAT2_1.fa").write_text(
                ">c1\nAAAA\n",
                encoding="utf-8",
            )
            (metadecoder / "S_MetaDecoder_1.fa").write_text(
                ">c2\nCCCC\n",
                encoding="utf-8",
            )
            (refined / "bin.1.fa").write_text(
                ">c1\nAAAA\n",
                encoding="utf-8",
            )
            (refined / "bin.2.fa").write_text(
                ">c2\nCCCC\n>c3\nGGGG\n",
                encoding="utf-8",
            )

            normalize_refined_fasta(
                refined,
                refined,
                "S_MetaWRAP",
                [metabat2, metadecoder],
            )
            expected = ["S_MetaBAT2_1.fa", "S_MetaWRAP_1.fa"]
            self.assertEqual(
                expected,
                sorted(path.name for path in refined.glob("*.fa")),
            )
            self.assertFalse((refined / "bin.1.fa").exists())
            self.assertFalse((refined / "bin.2.fa").exists())

            candidates = root / "quality_control_files" / "candidate_bins"
            collect_fasta([refined], candidates)
            self.assertEqual(
                expected,
                sorted(path.name for path in candidates.glob("*.fa")),
            )
            self.assertTrue((candidates / "manifest.tsv").is_file())
            for name in expected:
                self.assertEqual(
                    (refined / name).read_bytes(),
                    (candidates / name).read_bytes(),
                )

    def test_duplicate_contigs_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "bad.fna"
            source.write_text(">same\nAAAA\n>same duplicate\nCCCC\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                fasta_filter(source, root / "out.fna", 1)

    def test_bins_to_map_filters_bins_below_the_requested_size(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            bins = root / "bins"
            bins.mkdir()
            (bins / "small.fa").write_text(">small\nAAAA\n", encoding="utf-8")
            (bins / "large.fa").write_text(">large\n" + "A" * 20 + "\n", encoding="utf-8")
            mapping = root / "mapping.tsv"
            bins_to_map(bins, mapping, "lorbin", minimum_bin_bp=10)
            text = mapping.read_text(encoding="utf-8")
            self.assertIn("lorbin__large\tlarge\tlorbin", text)
            self.assertNotIn("small", text)

    def test_gunc_low_reference_support_is_unscored_not_failed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            bins = root / "bins"
            bins.mkdir()
            (bins / "binA.fna").write_text(">c1\nAAAA\n", encoding="utf-8")
            checkm2 = root / "quality_report.tsv"
            checkm2.write_text("Name\tCompleteness\tContamination\nbinA\t90\t2\n", encoding="utf-8")
            gunc = root / "gunc"
            gunc.mkdir()
            (gunc / "GUNC.test.maxCSS_level.tsv").write_text(
                "genome\tpass.GUNC\treference_representation_score\nbinA\tTrue\t0.10\n",
                encoding="utf-8",
            )
            output = root / "selected"
            summary = root / "summary.tsv"
            filter_quality(bins, checkm2, output, summary, 50, 10, gunc, 0.30, False)
            self.assertTrue((output / "binA.fa").exists())
            self.assertIn("unscored\ttrue\tpass", summary.read_text(encoding="utf-8"))

    def test_chen_tovar_family_classification(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            abundance = root / "abundance.tsv"
            header = ["Genome"]
            for sample in ("S1", "S2", "S3", "S4"):
                header.extend((f"{sample}|{sample} Relative Abundance (%)", f"{sample}|{sample} Count"))
            rows = [
                ["g1", 10, 100, 10, 100, 10, 100, 10, 100],
                ["g2", 10, 100, 0, 0, 0, 0, 0, 0],
                ["g3", 10, 100, 10, 100, 10, 100, 0, 0],
            ]
            abundance.write_text(
                "\t".join(map(str, header)) + "\n" + "\n".join("\t".join(map(str, row)) for row in rows) + "\n",
                encoding="utf-8",
            )
            taxonomy_dir = root / "gtdbtk"
            taxonomy_dir.mkdir()
            (taxonomy_dir / "gtdbtk.bac120.summary.tsv").write_text(
                "user_genome\tclassification\n"
                "g1\td__Bacteria;p__P;c__C;o__O;f__A;g__A;s__A a\n"
                "g2\td__Bacteria;p__P;c__C;o__O;f__B;g__B;s__B b\n"
                "g3\td__Bacteria;p__P;c__C;o__O;f__C;g__C;s__C c\n",
                encoding="utf-8",
            )
            output = root / "niche.tsv"
            classify_niche_literature(
                abundance,
                taxonomy_dir,
                output,
                root / "taxon_abundance.tsv",
                root / "assignments.tsv",
                root / "provenance.json",
                ["S1=D1", "S2=D1", "S3=D1", "S4=D1"],
                "family",
                0.01,
                20,
                0.20,
                0.80,
                2,
            )
            classifications = {}
            lines = output.read_text(encoding="utf-8").splitlines()
            columns = lines[0].split("\t")
            for line in lines[1:]:
                row = dict(zip(columns, line.split("\t")))
                classifications[row["taxon"]] = row["niche"]
            self.assertEqual(classifications["f__A"], "generalist")
            self.assertEqual(classifications["f__B"], "specialist")
            self.assertEqual(classifications["f__C"], "intermediate")
            self.assertIn("10.1038/s41396-021-00988-w", (root / "provenance.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
