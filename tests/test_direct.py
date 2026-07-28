from pathlib import Path
import argparse
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from metabaw import __version__
from metabaw.cli import (
    MetaBAWHelpFormatter,
    _apply_comebin_gpu_budget,
    _check_software_interactively,
    _gpu_task_slots,
    _offer_cuda_repair,
    _run_direct,
    _temporary_path,
    build_parser,
    command_annotation,
    command_bin,
    command_check,
)
from metabaw.dependencies import (
    CudaDevice,
    CudaRuntimeStatus,
    HostCudaStatus,
    ISOLATED_TOOLS,
    IsolatedEnvironmentStatus,
    SoftwareRequirement,
    configured_isolated_environment,
)
from metabaw.direct import AnnotationBuilder, AnnotationOptions, BinOptions, DirectBinBuilder
from metabaw.discovery import (
    Analysis,
    ReadSample,
    attach_contigs,
    build_analyses,
    discover_contigs,
    discover_reads,
    normalize_sample_name,
    public_sample_name,
    read_multi_files,
)
from metabaw.internal import merge_aemb, merge_coverm_taxonomy
from metabaw.strategy import ServerResourceProfile


class DirectWorkflowTests(unittest.TestCase):
    def test_gpu_memory_budget_allows_multiple_sample_slots_on_one_large_gpu(
        self,
    ) -> None:
        status = HostCudaStatus(
            executable="/usr/bin/nvidia-smi",
            devices=(
                CudaDevice(
                    index="0",
                    name="NVIDIA RTX A6000",
                    memory_total_mib=49140,
                    driver_version="570.133.07",
                ),
            ),
            advertised_cuda_version="12.8",
        )
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(3, _gpu_task_slots(status, 1, "4G", 3))
            self.assertEqual(1, _gpu_task_slots(status, 1, "40G", 3))

    def test_gpu_slots_are_limited_by_currently_free_vram(self) -> None:
        status = HostCudaStatus(
            executable="/usr/bin/nvidia-smi",
            devices=(
                CudaDevice(
                    index="0",
                    name="NVIDIA RTX A6000",
                    memory_total_mib=49140,
                    driver_version="570.133.07",
                    memory_free_mib=7000,
                ),
            ),
            advertised_cuda_version="12.8",
        )
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(1, _gpu_task_slots(status, 1, "4G", 3))

    def test_cuda_repair_retests_the_isolated_environment(self) -> None:
        failed = CudaRuntimeStatus(
            label="COMEBin",
            python_version="3.7",
            torch_version="1.10.2",
            torch_cuda_version=None,
            cuda_visible_devices="0",
            device_count=0,
            devices=(),
            allocation_test=False,
            error="torch.cuda.is_available() returned False",
        )
        repaired = CudaRuntimeStatus(
            label="COMEBin",
            python_version="3.7",
            torch_version="1.10.2",
            torch_cuda_version="11.1",
            cuda_visible_devices="0",
            device_count=1,
            devices=("NVIDIA A100",),
            allocation_test=True,
        )
        with patch(
            "metabaw.cli.confirm_install",
            return_value=True,
        ), patch(
            "metabaw.cli.install_isolated_cuda_runtime",
        ) as installer, patch(
            "metabaw.cli._isolated_cuda_status",
            return_value=repaired,
        ), redirect_stdout(io.StringIO()):
            result = _offer_cuda_repair(
                failed,
                "comebin",
                "comebin-py37",
                "mamba",
            )
        self.assertTrue(result.available)
        installer.assert_called_once_with(
            "comebin",
            "comebin-py37",
            "mamba",
        )

    def test_default_gpu_budget_reduces_comebin_batch_size(self) -> None:
        args = argparse.Namespace(
            gpu=True,
            tools=["comebin"],
            batch_size=1024,
            max_gpu_memory="4G",
        )
        with redirect_stdout(io.StringIO()):
            _apply_comebin_gpu_budget(args, total_memory_mib=16384)
        self.assertEqual(1024, args.requested_batch_size)
        self.assertEqual(256, args.batch_size)

    def test_public_cli_contains_requested_modules(self) -> None:
        parser = build_parser()
        subparsers = next(
            action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
        )
        self.assertEqual({"bin", "annotation", "check"}, set(subparsers.choices))
        self.assertEqual("0.1.0", __version__)
        self.assertEqual(
            "MetaBAW: metagenome Binning Automated Workflow",
            parser.description,
        )

    def test_short_and_long_version_options_are_equivalent(self) -> None:
        parser = build_parser()
        outputs = []
        for option in ("-v", "--version"):
            terminal = io.StringIO()
            with redirect_stdout(terminal), self.assertRaises(SystemExit) as exit_status:
                parser.parse_args([option])
            self.assertEqual(0, exit_status.exception.code)
            outputs.append(terminal.getvalue())
        self.assertEqual(["metaBAW 0.1.0\n", "metaBAW 0.1.0\n"], outputs)

    def test_help_hides_underscore_compatibility_aliases(self) -> None:
        parser = build_parser()
        subparsers = next(
            action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
        )
        help_text = subparsers.choices["bin"].format_help()
        self.assertIn("--align-tool", help_text)
        self.assertIn("--dereplication-tool", help_text)
        self.assertNotIn("--align_tool", help_text)
        self.assertNotIn("--dereplication_tool", help_text)

        compatibility = parser.parse_args(
            [
                "bin",
                "-p",
                "reads",
                "-c",
                "contigs",
                "--align_tool",
                "minimap2",
                "--dereplication_tool",
                "drep",
            ]
        )
        self.assertEqual("minimap2", compatibility.align_tool)
        self.assertEqual("drep", compatibility.dereplication_tool)

    def test_every_option_is_marked_required_or_has_a_default(self) -> None:
        parser = build_parser()
        subparsers = next(
            action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
        )
        formatter = MetaBAWHelpFormatter("metabaw")
        for command in ("bin", "annotation", "check"):
            for action in subparsers.choices[command]._actions:
                if (
                    not action.option_strings
                    or action.help == argparse.SUPPRESS
                    or action.default == argparse.SUPPRESS
                ):
                    continue
                help_text = formatter._get_help_string(action).lower()
                if action.required:
                    self.assertIn("required", help_text, action.dest)
                else:
                    self.assertIn("default:", help_text, action.dest)

    def test_required_inputs_are_listed_before_optional_parameters(self) -> None:
        parser = build_parser()
        subparsers = next(
            action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
        )
        expected = {
            "bin": ["path", "contig"],
            "annotation": ["path", "reads"],
        }
        for command, required_destinations in expected.items():
            visible = [
                action
                for action in subparsers.choices[command]._actions
                if action.option_strings
                and action.dest not in {"help", "full_help"}
                and action.help != argparse.SUPPRESS
            ]
            self.assertEqual(
                required_destinations,
                [action.dest for action in visible[: len(required_destinations)]],
            )
            self.assertTrue(all(action.required for action in visible[: len(required_destinations)]))
            self.assertTrue(
                all(not action.required for action in visible[len(required_destinations) :])
            )

    def test_semibin2_environment_choices_are_complete(self) -> None:
        parser = build_parser()
        subparsers = next(
            action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
        )
        environment = next(
            action
            for action in subparsers.choices["bin"]._actions
            if action.dest == "environment"
        )
        self.assertEqual(
            {
                "human_gut",
                "dog_gut",
                "ocean",
                "soil",
                "cat_gut",
                "human_oral",
                "mouse_gut",
                "pig_gut",
                "built_environment",
                "wastewater",
                "chicken_caecum",
                "global",
            },
            set(environment.choices or ()),
        )

    def test_max_memory_defaults_to_100_gib_in_both_modules(self) -> None:
        parser = build_parser()
        bin_args = parser.parse_args(["bin", "-p", "reads", "-c", "contigs"])
        annotation_args = parser.parse_args(
            ["annotation", "-p", "mags", "-r", "reads"]
        )
        self.assertEqual(100, bin_args.max_memory)
        self.assertEqual(100, annotation_args.max_memory)

    def test_rna_prediction_flags_do_not_enable_rna_filters(self) -> None:
        parser = build_parser()
        prediction = parser.parse_args(
            [
                "bin",
                "-p",
                "reads",
                "-c",
                "contigs",
                "--trna",
                "--rrna",
            ]
        )
        self.assertTrue(prediction.trna)
        self.assertTrue(prediction.rrna)
        self.assertIsNone(prediction.trna_pass)
        self.assertFalse(prediction.rrna_pass)

        filtering = parser.parse_args(
            [
                "bin",
                "-p",
                "reads",
                "-c",
                "contigs",
                "--trna-pass",
                "18",
                "--rrna-pass",
            ]
        )
        self.assertEqual(18, filtering.trna_pass)
        self.assertTrue(filtering.rrna_pass)

    def test_task_controls_concurrent_samples_and_hides_old_option(self) -> None:
        parser = build_parser()
        subparsers = next(
            action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
        )
        for command, required in (
            ("bin", ["-p", "reads", "-c", "contigs"]),
            ("annotation", ["-p", "mags", "-r", "reads"]),
        ):
            default_args = parser.parse_args([command, *required])
            selected_args = parser.parse_args([command, *required, "--task", "3"])
            compatible_args = parser.parse_args(
                [command, *required, "--max-parallel", "2"]
            )
            self.assertEqual(1, default_args.task)
            self.assertEqual(3, selected_args.task)
            self.assertEqual(2, compatible_args.task)
            help_text = subparsers.choices[command].format_help()
            self.assertIn("--task", help_text)
            self.assertNotIn("--max-parallel", help_text)

    def test_auto_is_visible_in_short_bin_help(self) -> None:
        parser = build_parser()
        for option in ("-h", "--help"):
            terminal = io.StringIO()
            with redirect_stdout(terminal), self.assertRaises(SystemExit) as raised:
                parser.parse_args(["bin", option])
            self.assertEqual(0, raised.exception.code)
            help_text = terminal.getvalue()
            self.assertIn("--auto", help_text)
            self.assertIn("metabaw_auto_run.sh", help_text)

    def test_auto_requires_explicit_read_and_contig_suffixes(self) -> None:
        args = build_parser().parse_args(
            ["bin", "-p", "reads", "-c", "contigs", "--auto", "--dry-run"]
        )
        args._provided_options = {"-p", "-c", "--auto", "--dry-run"}
        with self.assertRaisesRegex(
            ValueError,
            r"-s/--suffix.*-f/--contig-suffix",
        ):
            command_bin(args)

    def test_metawrap_rejects_more_than_three_binners_before_discovery(
        self,
    ) -> None:
        parser = build_parser()
        for automatic in (False, True):
            with self.subTest(auto=automatic):
                command = [
                    "bin",
                    "-p",
                    "missing_reads",
                    "-s",
                    "fastq.gz",
                    "-c",
                    "missing_contigs",
                    "-f",
                    "fa",
                    "--tools",
                    "metabat2",
                    "metadecoder",
                    "vamb",
                    "semibin2",
                    "--refinement",
                    "metawrap",
                ]
                if automatic:
                    command.append("--auto")
                args = parser.parse_args(command)
                with patch("metabaw.cli.discover_reads") as discover:
                    with self.assertRaisesRegex(
                        ValueError,
                        r"at most 3 binning tools.*4 were selected: "
                        r"metabat2, metadecoder, vamb, semibin2",
                    ):
                        command_bin(args)
                    discover.assert_not_called()

    def test_check_has_isolated_scopes_and_environment_defaults(self) -> None:
        parser = build_parser()
        for scope in ("comebin", "checkm2", "metawrap", "lorbin"):
            args = parser.parse_args(["check", "--scope", scope])
            self.assertEqual(scope, args.scope)
        args = parser.parse_args(["check"])
        self.assertTrue(args.essential)
        self.assertFalse(args.all)
        self.assertIsNone(args.scope)
        self.assertEqual("metabaw-comebin-py37", args.comebin_env)
        self.assertEqual("metabaw-checkm2-py312", args.checkm2_env)
        self.assertEqual("metabaw-metawrap-py27", args.metawrap_env)
        self.assertEqual("metabaw-lorbin-py310", args.lorbin_env)

    def test_check_all_and_essential_are_mutually_exclusive(self) -> None:
        parser = build_parser()
        explicit_essential = parser.parse_args(["check", "--essential"])
        selected_all = parser.parse_args(["check", "--all"])
        self.assertTrue(explicit_essential.essential)
        self.assertFalse(explicit_essential.all)
        self.assertTrue(selected_all.all)
        with self.assertRaises(SystemExit):
            parser.parse_args(["check", "--all", "--essential"])

    def test_check_asks_before_installing_missing_software(self) -> None:
        requirement = SoftwareRequirement("example-tool", "example-package", "test")
        with patch(
            "metabaw.cli.missing_software",
            side_effect=[[requirement], []],
        ), patch(
            "metabaw.cli.missing_magscot_r_packages",
            return_value=[],
        ), patch(
            "metabaw.cli.confirm_install",
            return_value=True,
        ) as confirmation, patch(
            "metabaw.cli.install_software",
            return_value=[],
        ) as installer:
            self.assertTrue(_check_software_interactively([requirement]))
        confirmation.assert_called_once()
        installer.assert_called_once_with([requirement])

    def test_check_saves_an_explicit_environment_prefix_for_later_commands(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.json"
            prefix = root / "environments" / "comebin-py37"
            status = IsolatedEnvironmentStatus(
                spec=ISOLATED_TOOLS["comebin"],
                environment=str(prefix),
                frontend="mamba",
                python_version="3.7",
                executable=str(prefix / "bin" / "run_comebin.sh"),
            )
            with patch.dict(
                os.environ,
                {"METABAW_CONFIG_FILE": str(config_path)},
                clear=False,
            ), patch(
                "metabaw.cli.install_software",
                return_value=[],
            ), patch(
                "metabaw.cli._ensure_isolated_tool",
                return_value=status,
            ):
                parser = build_parser()
                args = parser.parse_args(
                    [
                        "check",
                        "--scope",
                        "comebin",
                        "--comebin-env",
                        str(prefix),
                    ]
                )
                self.assertEqual(0, command_check(args))
                later = build_parser().parse_args(
                    ["bin", "-p", "reads", "-c", "contigs"]
                )
                configured = configured_isolated_environment("comebin")
            self.assertFalse(hasattr(later, "comebin_env"))
            self.assertEqual(str(prefix.resolve()), configured)
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(
                str(prefix.resolve()),
                payload["isolated_environments"]["comebin"],
            )

    def test_temporary_workspaces_default_under_output_and_allow_custom_names(self) -> None:
        parser = build_parser()
        bin_args = parser.parse_args(["bin", "-p", "reads", "-c", "contigs"])
        annotation_args = parser.parse_args(
            ["annotation", "-p", "mags", "-r", "reads"]
        )
        self.assertIsNone(bin_args.tmp_files)
        self.assertIsNone(annotation_args.tmp_files)
        output = Path("/analysis/result").resolve()
        self.assertEqual(output / "tmp", _temporary_path(None, output))
        self.assertEqual(output / "scratch", _temporary_path("scratch", output))
        self.assertFalse(bin_args.delete_tmp_files)
        self.assertFalse(annotation_args.delete_tmp_files)

    def test_runtime_metadata_and_logs_are_created_inside_tmp_files(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            output = root / "result"
            temp_files = root / "visible_tmp"
            args = argparse.Namespace(
                dry_run=False,
                threads=1,
                task=1,
                max_memory=100,
                retries=0,
                force=False,
                delete_tmp_files=False,
            )
            terminal = io.StringIO()
            with redirect_stdout(terminal):
                self.assertEqual(
                    0,
                    _run_direct(args, [], output, temp_files, {"module": "test"}),
                )
            self.assertTrue((output / "run_manifest.json").is_file())
            self.assertTrue((temp_files / "runtime" / "state.sqlite3").is_file())
            self.assertTrue((temp_files / "runtime" / "logs").is_dir())
            self.assertTrue((temp_files / "system_tmp").is_dir())
            self.assertFalse((output / ".metabaw").exists())
            self.assertIn(f"temporary files={temp_files}", terminal.getvalue())

    def test_delete_tmp_files_removes_only_successful_run_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            output = root / "result"
            temp_files = output / "tmp"
            args = argparse.Namespace(
                dry_run=False,
                threads=1,
                task=1,
                max_memory=100,
                retries=0,
                force=False,
                delete_tmp_files=True,
            )
            terminal = io.StringIO()
            with redirect_stdout(terminal):
                self.assertEqual(
                    0,
                    _run_direct(args, [], output, temp_files, {"module": "test"}),
                )
            self.assertTrue((output / "run_manifest.json").is_file())
            self.assertFalse(temp_files.exists())
            self.assertIn("temporary files deleted", terminal.getvalue())

    def test_discovery_pairs_and_multi_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("A_R1.fastq.gz", "A_R2.fastq.gz", "B_R1.fastq.gz", "B_R2.fastq.gz"):
                (root / name).write_bytes(b"")
            for name in ("A.fa", "B.fa"):
                (root / name).write_text(">c\nAAAA\n", encoding="utf-8")
            samples = discover_reads(root, "fastq.gz", "short")
            samples = attach_contigs(samples, discover_contigs(root, "fa"), "fa")
            self.assertEqual(["A", "B"], [sample.name for sample in samples])
            self.assertTrue(all(sample.read2 for sample in samples))

            table = root / "multi.tsv"
            table.write_text(
                f"A.clean\t{root / 'A_R1.fastq.gz'},{root / 'A_R2.fastq.gz'}\t{root / 'A.fa'}\n",
                encoding="utf-8",
            )
            selected = read_multi_files(table)
            self.assertEqual(["A"], [sample.name for sample in selected])
            analyses = build_analyses(samples, "multi", selected)
            self.assertEqual(["A", "B"], [analysis.name for analysis in analyses])
            self.assertFalse(any(analysis.cross_mapped for analysis in analyses))

    def test_plain_multi_cross_maps_per_sample_assemblies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("A_R1.fastq.gz", "A_R2.fastq.gz", "B_R1.fastq.gz", "B_R2.fastq.gz"):
                (root / name).write_bytes(b"")
            for name in ("A.fa", "B.fa"):
                (root / name).write_text(">c\nAAAA\n", encoding="utf-8")
            samples = discover_reads(root, "fastq.gz", "short")
            samples = attach_contigs(samples, discover_contigs(root, "fa"), "fa")

            analyses = build_analyses(samples, "multi", None)
            self.assertEqual(["A", "B"], [analysis.name for analysis in analyses])
            self.assertTrue(all(analysis.cross_mapped for analysis in analyses))
            self.assertFalse(any(analysis.combined for analysis in analyses))
            for analysis in analyses:
                self.assertEqual(("A", "B"), tuple(sample.name for sample in analysis.samples))
            contig_names = {
                analysis.name: [path.name for path in analysis.contigs] for analysis in analyses
            }
            self.assertEqual({"A": ["A.fa"], "B": ["B.fa"]}, contig_names)

            shared = attach_contigs(samples, [root / "A.fa"], "fa")
            with self.assertRaisesRegex(
                ValueError,
                "Shared co-assembly input is not enabled",
            ):
                build_analyses(shared, "multi", None)

    def test_bin_short_help_shows_only_core_options(self) -> None:
        parser = build_parser()
        terminal = io.StringIO()
        with redirect_stdout(terminal), self.assertRaises(SystemExit) as context:
            parser.parse_args(["bin", "-h"])
        self.assertEqual(0, context.exception.code)
        text = terminal.getvalue()
        self.assertNotIn("usage:", text)
        self.assertIn("core options:", text)
        self.assertIn("--path", text)
        self.assertIn("--threads", text)
        self.assertIn("--full-help", text)
        self.assertNotIn("--tools", text)
        self.assertNotIn("--max-memory", text)
        self.assertNotIn("--advanced-arg", text)
        self.assertNotIn("--comebin-env", text)
        self.assertNotIn("--batch-size", text)

    def test_annotation_short_help_shows_only_core_options(self) -> None:
        parser = build_parser()
        terminal = io.StringIO()
        with redirect_stdout(terminal), self.assertRaises(SystemExit) as context:
            parser.parse_args(["annotation", "-h"])
        self.assertEqual(0, context.exception.code)
        text = terminal.getvalue()
        self.assertNotIn("usage:", text)
        self.assertIn("core options:", text)
        self.assertIn("--reads", text)
        self.assertIn("--threads", text)
        self.assertNotIn("--niche-rank", text)
        self.assertNotIn("--gtdbtk-db", text)
        self.assertNotIn("--output-file-suffix", text)
        self.assertNotIn("--tmp-files", text)

    def test_full_help_prints_every_option_when_not_a_tty(self) -> None:
        parser = build_parser()
        terminal = io.StringIO()
        with redirect_stdout(terminal), self.assertRaises(SystemExit) as context:
            parser.parse_args(["bin", "--full-help"])
        self.assertEqual(0, context.exception.code)
        text = terminal.getvalue()
        self.assertIn("--advanced-arg", text)
        self.assertIn("--batch-size", text)
        for option in (
            "--magscot-dir",
            "--checkm2-db",
            "--gunc-db",
            "--comebin-env",
            "--lorbin-env",
            "--metawrap-env",
            "--checkm2-env",
        ):
            self.assertNotIn(option, text)

    def test_full_help_uses_less_pager_on_a_tty(self) -> None:
        parser = build_parser()
        process = unittest.mock.Mock()
        with patch("sys.stdout.isatty", return_value=True), patch(
            "metabaw.cli.subprocess.Popen", return_value=process
        ) as popen, patch.dict(os.environ, {}, clear=False), self.assertRaises(
            SystemExit
        ) as context:
            os.environ.pop("PAGER", None)
            parser.parse_args(["bin", "--full-help"])
        self.assertEqual(0, context.exception.code)
        self.assertTrue(popen.called)
        command = popen.call_args.args[0]
        self.assertEqual("less", command[0])
        self.assertIn("-R", command)
        self.assertTrue(process.communicate.called)

    def test_clean_reads_match_contig_ok_assemblies_by_sample_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reads_dir = root / "reads"
            contigs_dir = root / "contigs"
            reads_dir.mkdir()
            contigs_dir.mkdir()
            for sample in ("A606", "B607"):
                (reads_dir / f"{sample}.clean_R1.fastq.gz").write_bytes(b"")
                (reads_dir / f"{sample}.clean_R2.fastq.gz").write_bytes(b"")
                (contigs_dir / f"{sample}.contig.ok.fa").write_text(
                    ">contig_1\nAAAA\n",
                    encoding="utf-8",
                )
            reads = discover_reads(reads_dir, "fastq.gz", "short")
            contigs = discover_contigs(contigs_dir, "fa")
            attached = attach_contigs(reads, contigs, "fa")
            self.assertEqual(
                {
                    "A606": (contigs_dir / "A606.contig.ok.fa").resolve(),
                    "B607": (contigs_dir / "B607.contig.ok.fa").resolve(),
                },
                {sample.name: sample.contigs for sample in attached},
            )
            self.assertEqual("A606", public_sample_name("A606.clean"))
            self.assertEqual("B607", public_sample_name("B607_cleaned"))

    def test_clean_name_normalization_rejects_colliding_read_sets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in (
                "A606_R1.fastq.gz",
                "A606_R2.fastq.gz",
                "A606.clean_R1.fastq.gz",
                "A606.clean_R2.fastq.gz",
            ):
                (root / name).write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "Ambiguous read files"):
                discover_reads(root, "fastq.gz", "short")

    def test_clean_suffix_variants_are_removed_without_changing_real_names(self) -> None:
        self.assertEqual("A606", normalize_sample_name("A606.clean"))
        self.assertEqual("A606", normalize_sample_name("A606_cleaned"))
        self.assertEqual("A606", normalize_sample_name("A606-clean"))
        self.assertEqual("A606", normalize_sample_name("A606.clean.cleaned"))
        self.assertEqual("cleanroom", normalize_sample_name("cleanroom"))
        self.assertEqual("A606.cleanroom", normalize_sample_name("A606.cleanroom"))

    def test_hyphens_are_preserved_in_sample_and_public_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reads_dir = root / "reads"
            contigs_dir = root / "contigs"
            reads_dir.mkdir()
            contigs_dir.mkdir()
            (reads_dir / "site-A-01.clean_R1.fastq.gz").write_bytes(b"")
            (reads_dir / "site-A-01.clean_R2.fastq.gz").write_bytes(b"")
            contigs = contigs_dir / "site-A-01.contig.ok.fa"
            contigs.write_text(">contig_1\nAAAA\n", encoding="utf-8")

            samples = discover_reads(reads_dir, "fastq.gz", "short")
            attached = attach_contigs(samples, discover_contigs(contigs_dir, "fa"), "fa")

            self.assertEqual(["site-A-01"], [sample.name for sample in attached])
            self.assertEqual(contigs.resolve(), attached[0].contigs)
            self.assertEqual("site-A-01", normalize_sample_name("site-A-01-clean"))
            self.assertEqual("site-A-01", public_sample_name("site-A-01.contig.ok"))

    def test_all_binners_build_one_consistent_dag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contigs = root / "contigs.fa"
            read1 = root / "S_R1.fastq.gz"
            read2 = root / "S_R2.fastq.gz"
            contigs.write_text(">c\n" + "A" * 2000 + "\n", encoding="utf-8")
            read1.write_bytes(b"")
            read2.write_bytes(b"")
            sample = ReadSample("S", read1, read2, contigs)
            analysis = Analysis("S", (sample,), (contigs,), False)
            options = BinOptions(
                outdir=root / "result",
                workdir=root / "work",
                threads=4,
                read_type="short",
                align_tool="bowtie2",
                long_read_preset="map-ont",
                binners=(
                    "metabat2",
                    "vamb",
                    "metadecoder",
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
                environment="human_gut",
                tag_contigs=True,
                gpu=True,
                max_gpu_memory="4G",
                magscot_dir=root / "MAGScoT",
                checkm2_db=None,
                gunc_db=None,
                gtdbtk_data=None,
                extra_args={},
            )
            tasks = DirectBinBuilder([analysis], options, root).build()
            ids = {task.id for task in tasks}
            barrier = next(
                task for task in tasks if task.id == "03.binning.complete"
            )
            self.assertEqual(
                {
                    task.id
                    for task in tasks
                    if task.stage == "03_binning" and task.id != barrier.id
                },
                set(barrier.deps),
            )
            self.assertEqual(
                (root / "work" / "binning" / "binning.complete",),
                barrier.outputs,
            )
            self.assertTrue(
                all(
                    barrier.id in task.wait_for
                    for task in tasks
                    if task.stage == "04_refinement"
                )
            )
            for name in options.binners:
                self.assertIn(f"03.bin.{name}.S", ids)
                self.assertIn(f"03.publish.{name}.S", ids)
                binner_task = next(
                    task for task in tasks if task.id == f"03.bin.{name}.S"
                )
                self.assertEqual("S", binner_task.sample)
                self.assertEqual(1, binner_task.automatic_retries)
                self.assertTrue(binner_task.fasta_output_dirs)
            aemb = next(task for task in tasks if task.id == "03.aemb.merge.S")
            self.assertIn("merge-aemb", aemb.display_command())
            self.assertIn("05.qc.rna_domain", ids)
            index = next(task for task in tasks if task.id == "02.index.S")
            self.assertEqual(2, len(index.output_alternatives))
            self.assertTrue(
                any(str(path).endswith("S.1.bt2") for path in index.output_alternatives[0])
            )
            mapping = next(task for task in tasks if task.id == "02.map.S.S")
            self.assertIn(str(root / "work" / "mapping" / "S" / "index" / "S"), mapping.display_command())
            self.assertNotIn("/index/contigs", mapping.display_command())
            run_order = [
                task.id.split(".")[2]
                for task in tasks
                if task.id.startswith("03.bin.")
            ]
            self.assertEqual(
                [
                    "metabat2",
                    "metadecoder",
                    "vamb",
                    "comebin",
                    "semibin2",
                    "lorbin",
                ],
                run_order,
            )
            self.assertTrue(
                all(
                    task.failure_tolerated
                    for task in tasks
                    if task.stage == "03_binning"
                )
            )
            prodigal = next(task for task in tasks if task.id == "04.prodigal.S")
            self.assertIn("parallel --jobs 4", prodigal.display_command())
            self.assertEqual(4, prodigal.cpus)
            threaded_commands = {
                "02.index.S": "--threads 4",
                "02.map.S.S": "-p 4",
                "03.bin.metabat2.S": "-t 4",
                "03.bin.metadecoder.S": "--threads 4",
                "03.bin.vamb.S": "-p 4",
                "03.bin.comebin.S": "-t 4",
                "03.bin.semibin2.S": "--threads 4",
                "03.bin.lorbin.S": "--num_process 4",
                "04.hmm.pfam.S": "--cpu 4",
                "05.qc.checkm2": "--threads 4",
                "05.qc.gunc": "--threads 4",
                "05.qc.rna_domain": "--cpus 4",
                "05.qc.rna": "--threads 4",
                "06.dereplicate.galah": "--threads 4",
            }
            by_id = {task.id: task for task in tasks}
            for binner in ("vamb", "comebin", "semibin2", "lorbin"):
                self.assertEqual(1, by_id[f"03.bin.{binner}.S"].gpus)
            for binner in ("metabat2", "metadecoder"):
                self.assertEqual(0, by_id[f"03.bin.{binner}.S"].gpus)
            for task_id, expected_argument in threaded_commands.items():
                self.assertIn(expected_argument, by_id[task_id].display_command(), task_id)
            self.assertIn(
                "conda run --name metabaw-comebin-py37 run_comebin.sh",
                by_id["03.bin.comebin.S"].display_command(),
            )
            comebin_command = by_id["03.bin.comebin.S"].display_command()
            self.assertIn("rm -rf ", comebin_command)
            self.assertIn(" && mkdir -p ", comebin_command)
            self.assertIn(" && conda run ", comebin_command)
            self.assertIn(
                "SemiBin2 single_easy_bin",
                by_id["03.bin.semibin2.S"].display_command(),
            )
            self.assertIn(
                "mapping/S/bam/S.bam",
                by_id["03.bin.semibin2.S"]
                .display_command()
                .replace("\\", "/"),
            )
            self.assertIn(
                "--environment human_gut",
                by_id["03.bin.semibin2.S"].display_command(),
            )
            self.assertIn(
                "--engine gpu",
                by_id["03.bin.semibin2.S"].display_command(),
            )
            self.assertNotIn(
                "--device cuda",
                by_id["03.bin.semibin2.S"].display_command(),
            )
            self.assertIn(
                "conda run --name metabaw-checkm2-py312 checkm2 predict",
                by_id["05.qc.checkm2"].display_command(),
            )
            self.assertEqual("1", by_id["05.qc.checkm2"].env["OMP_NUM_THREADS"])
            self.assertEqual("1", by_id["05.qc.checkm2"].env["OPENBLAS_NUM_THREADS"])
            self.assertEqual("1", by_id["05.qc.checkm2"].env["TF_NUM_INTRAOP_THREADS"])
            self.assertEqual("1", by_id["05.qc.checkm2"].env["TF_NUM_INTEROP_THREADS"])
            self.assertIn(
                "conda run --name metabaw-lorbin-py310 LorBin bin",
                by_id["03.bin.lorbin.S"].display_command(),
            )
            self.assertIn(
                "--min-bin-bp 200000",
                by_id["03.map.lorbin.S"].display_command(),
            )
            self.assertIn(
                "--prefix S_MetaBAT2",
                by_id["03.publish.metabat2.S"].display_command(),
            )
            self.assertIn(
                "--prefix S_MAGScoT",
                by_id["04.bins.S"].display_command(),
            )
            self.assertIn("--file_suffix .fa", by_id["05.qc.gunc"].display_command())
            self.assertIn("--extension fa", by_id["05.qc.rna_domain"].display_command())
            rna_output = root / "result" / "quality_control_files" / "rna"
            rna_summary = root / "result" / "quality_control_files" / "rna_quality.tsv"
            rna_completion = rna_output / "rna_qc.v2.complete"
            rna_command = by_id["05.qc.rna"].display_command()
            self.assertIn("--output-dir", rna_command)
            self.assertIn(str(rna_output), rna_command)
            self.assertEqual(
                (rna_output, rna_summary, rna_completion),
                by_id["05.qc.rna"].outputs,
            )
            prediction_only = replace(
                options,
                trna_pass=None,
                rrna_pass=False,
            )
            prediction_tasks = DirectBinBuilder(
                [analysis],
                prediction_only,
                root,
            ).build()
            prediction_by_id = {task.id: task for task in prediction_tasks}
            prediction_rna = prediction_by_id["05.qc.rna"].display_command()
            prediction_filter = prediction_by_id["05.qc.filter"].display_command()
            self.assertIn("--trna", prediction_rna)
            self.assertIn("--rrna", prediction_rna)
            self.assertNotIn("--trna-pass", prediction_rna)
            self.assertNotIn("--rrna-pass", prediction_rna)
            self.assertNotIn("--rna-summary", prediction_filter)
            self.assertIn(
                "05.qc.rna",
                prediction_by_id["05.qc.filter"].deps,
            )
            self.assertIn(
                "--genome-fasta-extension fa",
                by_id["06.dereplicate.galah"].display_command(),
            )
            combine = next(task for task in tasks if task.id == "04.combine.S")
            self.assertTrue(combine.allow_failed_deps)
            self.assertEqual(
                root / "result" / "non_redundant_bins",
                next(task for task in tasks if task.id == "07.tag_contigs").outputs[0],
            )
            second_sample = ReadSample("B", read1, read2, contigs)
            second_analysis = Analysis(
                "B",
                (second_sample,),
                (contigs,),
                False,
            )
            multi_sample_tasks = DirectBinBuilder(
                [analysis, second_analysis],
                options,
                root,
            ).build()
            multi_barrier = next(
                task
                for task in multi_sample_tasks
                if task.id == "03.binning.complete"
            )
            self.assertTrue(
                any(task_id.endswith(".S") for task_id in multi_barrier.deps)
            )
            self.assertTrue(
                any(task_id.endswith(".B") for task_id in multi_barrier.deps)
            )
            metawrap_options = replace(
                options,
                binners=("metabat2",),
                refiner="metawrap",
            )
            metawrap_tasks = DirectBinBuilder(
                [analysis],
                metawrap_options,
                root,
            ).build()
            metawrap = next(
                task
                for task in metawrap_tasks
                if task.id == "04.refine.metawrap.S"
            )
            self.assertIn(
                "conda run --name metabaw-metawrap-py27 metawrap bin_refinement",
                metawrap.display_command(),
            )
            self.assertIn("mkdir -p", metawrap.display_command())
            self.assertIn(
                str(root / "work" / "refinement"),
                metawrap.display_command(),
            )
            metawrap_publish = next(
                task for task in metawrap_tasks if task.id == "04.bins.S"
            )
            self.assertIn(
                "--prefix S_MetaWRAP",
                metawrap_publish.display_command(),
            )
            self.assertIn(
                "normalize-refined-fasta",
                metawrap_publish.display_command(),
            )
            metawrap_bins = (
                root
                / "work"
                / "refinement"
                / "S"
                / "metawrap_50_10_bins"
            )
            self.assertEqual(
                (metawrap_bins, metawrap_bins / "manifest.tsv"),
                metawrap_publish.outputs,
            )
            metawrap_catalog = next(
                task
                for task in metawrap_tasks
                if task.id == "05.catalog.candidates"
            )
            self.assertEqual((metawrap_bins,), metawrap_catalog.inputs)
            metawrap_candidates = (
                root / "result" / "quality_control_files" / "candidate_bins"
            )
            self.assertEqual(
                (
                    metawrap_candidates,
                    metawrap_candidates / "manifest.tsv",
                ),
                metawrap_catalog.outputs,
            )

            dastool_tasks = DirectBinBuilder(
                [analysis],
                replace(options, binners=("metabat2",), refiner="das_tool"),
                root,
            ).build()
            dastool_publish = next(
                task for task in dastool_tasks if task.id == "04.bins.S"
            )
            self.assertIn(
                "--prefix S_DASTool",
                dastool_publish.display_command(),
            )
            self.assertIn(
                "normalize-refined-fasta",
                dastool_publish.display_command(),
            )
            dastool_bins = (
                root / "work" / "refinement" / "S" / "S_DASTool_bins"
            )
            self.assertIn("--source-dir", dastool_publish.display_command())
            self.assertIn(str(dastool_bins), dastool_publish.display_command())
            self.assertEqual(
                (dastool_bins, dastool_bins / "manifest.tsv"),
                dastool_publish.outputs,
            )
            candidate_catalog = next(
                task
                for task in dastool_tasks
                if task.id == "05.catalog.candidates"
            )
            self.assertEqual((dastool_bins,), candidate_catalog.inputs)

    def test_annotation_defaults_to_family_niche(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["annotation", "-p", "mags", "-r", "reads", "--dry-run"])
        self.assertEqual("family", args.niche_rank)
        self.assertTrue(args.place_species)

    def test_coverm_aemb_and_taxonomy_merges(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = root / "a.aemb"
            b = root / "b.aemb"
            a.write_text("contig1\t1.5\ncontig2\t2\n", encoding="utf-8")
            b.write_text("contig1\t3\ncontig2\t4\n", encoding="utf-8")
            merged = root / "abundance.tsv"
            merge_aemb([f"A={a}", f"B={b}"], merged)
            self.assertEqual("contigname\tA\tB", merged.read_text(encoding="utf-8").splitlines()[0])

            taxonomy = root / "taxonomy"
            taxonomy.mkdir()
            (taxonomy / "gtdbtk.bac120.summary.tsv").write_text(
                "user_genome\tclassification\n"
                "bin1\td__Bacteria;p__P;c__C;o__O;f__F;g__G;s__S\n",
                encoding="utf-8",
            )
            coverm = root / "sample.tsv"
            coverm.write_text(
                "Genome\tA Relative Abundance (%)\tA RPKM\tA TPM\tA Mean\tA Read Count\n"
                "bin1\t1\t2\t3\t4\t5\n",
                encoding="utf-8",
            )
            output = root / "merged"
            merge_coverm_taxonomy([f"A={coverm}"], taxonomy, output, ".tsv")
            self.assertTrue((output / "coverm_rel_abd.tsv").is_file())
            header = (output / "coverm_rel_abd.tsv").read_text(encoding="utf-8").splitlines()[0]
            self.assertEqual("Genome\tdomain\tphylum\tclass\torder\tfamily\tgenus\tspecies\tA", header)

    def test_annotation_dag_contains_taxonomy_abundance_merge_and_niche(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mags = root / "mags"
            mags.mkdir()
            (mags / "bin.fa").write_text(">c\nAAAA\n", encoding="utf-8")
            reads = []
            for name in ("A", "B"):
                path = root / f"{name}.fastq.gz"
                path.write_bytes(b"")
                reads.append(ReadSample(name, path))
            options = AnnotationOptions(
                mag_dir=mags,
                mag_suffix="fa",
                reads=tuple(reads),
                output=root / "annotation",
                output_suffix=".tsv",
                threads=2,
                read_type="short",
                methods=("relative_abundance", "count"),
                place_species=True,
                niche_rank="family",
                no_niche=False,
                gtdbtk_data=None,
            )
            tasks = AnnotationBuilder(options, root).build()
            self.assertEqual(
                {
                    "01.annotation.gtdbtk",
                    "02.annotation.coverm.A",
                    "02.annotation.coverm.B",
                    "03.annotation.merge",
                    "04.annotation.niche",
                },
                {task.id for task in tasks},
            )
            self.assertTrue(all(task.cwd == root for task in tasks))
            annotation_by_id = {task.id: task for task in tasks}
            self.assertIn(
                "--cpus 2",
                annotation_by_id["01.annotation.gtdbtk"].display_command(),
            )
            self.assertIn(
                "--threads 2",
                annotation_by_id["02.annotation.coverm.A"].display_command(),
            )

    def test_bin_cli_dry_run_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reads = root / "reads"
            reads.mkdir()
            (reads / "S_R1.fastq.gz").write_bytes(b"")
            (reads / "S_R2.fastq.gz").write_bytes(b"")
            contigs = root / "S.fa"
            contigs.write_text(">c\n" + "A" * 2000 + "\n", encoding="utf-8")
            args = build_parser().parse_args(
                [
                    "bin",
                    "-p",
                    str(reads),
                    "-c",
                    str(contigs),
                    "--tools",
                    "metabat2,vamb",
                    "--dry-run",
                    "--tmp-files",
                    str(root / "visible_tmp"),
                    "-o",
                    str(root / "result"),
                ]
            )
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(0, command_bin(args))
            plan = output.getvalue()
            self.assertIn("03.aemb.merge.S", plan)
            self.assertIn("06.dereplicate.galah", plan)
            self.assertIn(
                "metabaw-checkm2-py312 checkm2 predict",
                plan,
            )
            self.assertIn(str((root / "visible_tmp").resolve()), plan)
            self.assertNotIn(".metabaw", plan)

    def test_multi_sample_comebin_and_semibin2_commands_are_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reads = root / "reads"
            contigs = root / "contigs"
            reads.mkdir()
            contigs.mkdir()
            for sample in ("A", "B"):
                (reads / f"{sample}_R1.fastq.gz").write_bytes(b"")
                (reads / f"{sample}_R2.fastq.gz").write_bytes(b"")
                (contigs / f"{sample}.fa").write_text(
                    ">c\n" + "A" * 2000 + "\n",
                    encoding="utf-8",
                )
            args = build_parser().parse_args(
                [
                    "bin",
                    "-p",
                    str(reads),
                    "-c",
                    str(contigs),
                    "--multi",
                    "--tools",
                    "comebin,semibin2",
                    "--environment",
                    "human_gut",
                    "--gpu",
                    "--dry-run",
                    "-o",
                    str(root / "result"),
                ]
            )
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(0, command_bin(args))
            plan = output.getvalue()
            semibin_commands = [
                line
                for line in plan.splitlines()
                if line.lstrip().startswith("$") and "SemiBin2 multi_easy_bin" in line
            ]
            self.assertEqual(1, len(semibin_commands))
            self.assertIn("--self-supervised", semibin_commands[0])
            self.assertIn("--engine gpu", semibin_commands[0])
            self.assertNotIn("SemiBin2 single_easy_bin", plan)
            self.assertNotIn("--device cuda", plan)
            self.assertNotIn("--environment human_gut", plan)
            self.assertIn("--separator :", plan)
            normalized_semibin_command = semibin_commands[0].replace("\\", "/")
            self.assertIn(
                "contigs/semibin2_multisample_input.fna",
                normalized_semibin_command,
            )
            self.assertIn(
                "mapping/semibin2_multisample_input/bam/A.bam",
                normalized_semibin_command,
            )
            self.assertIn(
                "mapping/semibin2_multisample_input/bam/B.bam",
                normalized_semibin_command,
            )
            self.assertIn("01.prepare.semibin2_multisample_input", plan)
            self.assertIn("02.map.semibin2_multisample_input.A", plan)
            self.assertIn("02.map.semibin2_multisample_input.B", plan)
            self.assertIn("03.publish.semibin2.A", plan)
            self.assertIn("03.publish.semibin2.B", plan)
            self.assertNotIn("/semibin_multi/", plan.replace("\\", "/"))
            self.assertIn(
                str(
                    (
                        root
                        / "result"
                        / "tmp"
                        / "work"
                        / "binning"
                        / "semibin2_multisample_input"
                        / "semibin2"
                        / "samples"
                        / "A"
                        / "output_bins"
                    ).resolve()
                ),
                plan,
            )
            comebin_commands = [
                line for line in plan.splitlines() if "run_comebin.sh" in line
            ]
            self.assertEqual(2, len(comebin_commands))
            self.assertTrue(
                all("rm -rf " in line and " && mkdir -p " in line for line in comebin_commands)
            )

    def test_multisample_cross_mapping_keeps_bins_separate_by_target_assembly(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reads = root / "reads"
            contigs = root / "contigs"
            reads.mkdir()
            contigs.mkdir()
            for sample in ("A", "B"):
                (reads / f"{sample}_R1.fastq.gz").write_bytes(b"")
                (reads / f"{sample}_R2.fastq.gz").write_bytes(b"")
                (contigs / f"{sample}.fa").write_text(
                    f">{sample}_contig\n" + "A" * 2000 + "\n",
                    encoding="utf-8",
                )
            args = build_parser().parse_args(
                [
                    "bin",
                    "-p",
                    str(reads),
                    "-c",
                    str(contigs),
                    "--multi",
                    "--tools",
                    "metabat2,metadecoder,vamb,comebin",
                    "--dry-run",
                    "-o",
                    str(root / "result"),
                ]
            )
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(0, command_bin(args))
            plan = output.getvalue().replace("\\", "/")

            expected_bams = {
                "A": ("A_to_A.bam", "B_to_A.bam"),
                "B": ("A_to_B.bam", "B_to_B.bam"),
            }
            for target, target_bams in expected_bams.items():
                other = "B" if target == "A" else "A"
                other_target_bams = expected_bams[other]
                for read_sample in ("A", "B"):
                    self.assertIn(f"02.map.{target}.{read_sample}", plan)
                for bam in target_bams:
                    self.assertIn(f"/mapping/{target}/bam/{bam}", plan)

                depth_command = next(
                    line
                    for line in plan.splitlines()
                    if line.lstrip().startswith("$")
                    and "jgi_summarize_bam_contig_depths" in line
                    and f"/binning/{target}/metabat2/" in line
                )
                coverage_command = next(
                    line
                    for line in plan.splitlines()
                    if line.lstrip().startswith("$")
                    and "metadecoder coverage" in line
                    and f"/binning/{target}/metadecoder/" in line
                )
                for command in (depth_command, coverage_command):
                    for bam in target_bams:
                        self.assertIn(f"/mapping/{target}/bam/{bam}", command)
                    for bam in other_target_bams:
                        self.assertNotIn(
                            f"/mapping/{other}/bam/{bam}",
                            command,
                        )

                comebin_command = next(
                    line
                    for line in plan.splitlines()
                    if line.lstrip().startswith("$")
                    and "run_comebin.sh" in line
                    and f"/binning/{target}/comebin" in line
                )
                self.assertIn(f"/mapping/{target}/bamset", comebin_command)
                self.assertNotIn(f"/mapping/{other}/bamset", comebin_command)

                for read_sample in ("A", "B"):
                    self.assertIn(f"03.aemb.{target}.{read_sample}", plan)
                    aemb_command = next(
                        line
                        for line in plan.splitlines()
                        if line.lstrip().startswith("$")
                        and "strobealign --aemb" in line
                        and f"/binning/{target}/vamb/aemb/{read_sample}.tsv"
                        in line
                    )
                    self.assertIn(f"/contigs/{target}.fna", aemb_command)
                    self.assertIn(
                        f"/reads/{read_sample}_R1.fastq.gz",
                        aemb_command,
                    )
                vamb_command = next(
                    line
                    for line in plan.splitlines()
                    if line.lstrip().startswith("$")
                    and "vamb bin default" in line
                    and f"/binning/{target}/vamb/" in line
                )
                self.assertIn(
                    f"/binning/{target}/vamb/abundance.tsv",
                    vamb_command,
                )
                self.assertNotIn(
                    f"/binning/{other}/vamb/abundance.tsv",
                    vamb_command,
                )

                for binner in ("metabat2", "metadecoder", "vamb", "comebin"):
                    self.assertIn(f"03.publish.{binner}.{target}", plan)
                    self.assertIn(
                        f"/bin_files/{binner}/{target}",
                        plan,
                    )
                    self.assertNotIn(
                        f"/bin_files/{binner}/{target}/{other}_",
                        plan,
                    )

    def test_semibin2_multisample_cohorts_do_not_mix_bams(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reads = root / "reads"
            contigs = root / "contigs"
            reads.mkdir()
            contigs.mkdir()
            for sample in ("A", "B", "C", "D"):
                (reads / f"{sample}_R1.fastq.gz").write_bytes(b"")
                (reads / f"{sample}_R2.fastq.gz").write_bytes(b"")
                (contigs / f"{sample}.fa").write_text(
                    ">c\n" + "A" * 2000 + "\n",
                    encoding="utf-8",
                )
            args = build_parser().parse_args(
                [
                    "bin",
                    "-p",
                    str(reads),
                    "-c",
                    str(contigs),
                    "--multi",
                    "--cohort-size",
                    "2",
                    "--tools",
                    "semibin2",
                    "--dry-run",
                    "-o",
                    str(root / "result"),
                ]
            )
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(0, command_bin(args))
            plan = output.getvalue().replace("\\", "/")
            commands = [
                line
                for line in plan.splitlines()
                if line.lstrip().startswith("$")
                and "SemiBin2 multi_easy_bin" in line
            ]
            self.assertEqual(2, len(commands))
            first, second = commands
            self.assertIn("semibin2_multisample_input_001", first)
            self.assertIn("/bam/A.bam", first)
            self.assertIn("/bam/B.bam", first)
            self.assertNotIn("/bam/C.bam", first)
            self.assertNotIn("/bam/D.bam", first)
            self.assertIn("semibin2_multisample_input_002", second)
            self.assertIn("/bam/C.bam", second)
            self.assertIn("/bam/D.bam", second)
            self.assertNotIn("/bam/A.bam", second)
            self.assertNotIn("/bam/B.bam", second)
            self.assertIn("03.bin.semibin2.multi.001", plan)
            self.assertIn("03.bin.semibin2.multi.002", plan)

    def test_shared_coassembly_input_is_rejected_until_supported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reads = root / "reads"
            reads.mkdir()
            for sample in ("A", "B"):
                (reads / f"{sample}_R1.fastq.gz").write_bytes(b"")
                (reads / f"{sample}_R2.fastq.gz").write_bytes(b"")
            contigs = root / "coassembly.fa"
            contigs.write_text(">c\n" + "A" * 2000 + "\n", encoding="utf-8")
            args = build_parser().parse_args(
                [
                    "bin",
                    "-p",
                    str(reads),
                    "-c",
                    str(contigs),
                    "--multi",
                    "--tools",
                    "semibin2",
                    "--environment",
                    "human_gut",
                    "--dry-run",
                    "-o",
                    str(root / "result"),
                ]
            )
            with self.assertRaisesRegex(
                ValueError,
                "Shared co-assembly input is not enabled",
            ):
                command_bin(args)

    def test_long_reads_use_lorbin_instead_of_metabat2_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reads = root / "reads"
            reads.mkdir()
            (reads / "S.fastq.gz").write_bytes(b"")
            contigs = root / "S.fa"
            contigs.write_text(">c\n" + "A" * 2000 + "\n", encoding="utf-8")
            args = build_parser().parse_args(
                [
                    "bin",
                    "-p",
                    str(reads),
                    "-c",
                    str(contigs),
                    "--type",
                    "long",
                    "--dry-run",
                    "-o",
                    str(root / "result"),
                ]
            )
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(0, command_bin(args))
            plan = output.getvalue()
            self.assertEqual(["metadecoder", "vamb", "lorbin"], args.tools)
            self.assertIn("03.bin.lorbin.S", plan)
            self.assertIn("metabaw-lorbin-py310 LorBin bin", plan)
            self.assertNotIn("03.bin.metabat2.S", plan)

    def test_auto_cli_profiles_cohort_and_records_selected_strategy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reads = root / "reads"
            contigs = root / "contigs"
            reads.mkdir()
            contigs.mkdir()
            for sample in ("A", "B", "C"):
                (reads / f"{sample}_R1.fastq.gz").write_bytes(b"")
                (reads / f"{sample}_R2.fastq.gz").write_bytes(b"")
                (contigs / f"{sample}.fa").write_text(
                    ">c1\n" + "A" * 10_000 + "\n>c2\n" + "C" * 2_000 + "\n",
                    encoding="utf-8",
                )
            args = build_parser().parse_args(
                [
                    "bin",
                    "-p",
                    str(reads),
                    "-s",
                    "fastq.gz",
                    "-c",
                    str(contigs),
                    "-f",
                    "fa",
                    "--auto",
                    "--dry-run",
                    "-o",
                    str(root / "result"),
                ]
            )
            resources = ServerResourceProfile(
                cpu_total=32,
                cpu_affinity=32,
                cpu_load_1m=8.0,
                cpu_available=24,
                memory_total_gib=128,
                memory_available_gib=96,
                disk_free_gib=500,
                disk_path=root,
                gpu_available=False,
                gpu_count=0,
                gpu_total_memory_mib=0,
                gpu_free_memory_mib=0,
                gpu_names=(),
                cuda_driver_version=None,
                advertised_cuda_version=None,
                cuda_error="not detected",
            )
            terminal = io.StringIO()
            with patch(
                "metabaw.cli.profile_server_resources",
                return_value=resources,
            ), redirect_stdout(terminal):
                self.assertEqual(0, command_bin(args))
            self.assertEqual("multi", args.mode)
            self.assertEqual(
                ["metabat2", "metadecoder", "vamb", "semibin2"],
                args.tools,
            )
            self.assertEqual(1500, args.min_contig_length)
            self.assertIn("[AUTO] Selected mode=multi", terminal.getvalue())
            self.assertIn("03.bin.semibin2.multi", terminal.getvalue())
            script = root / "result" / "metabaw_auto_run.sh"
            self.assertTrue(script.is_file())
            script_text = script.read_text(encoding="utf-8")
            self.assertIn("metabaw bin", script_text)
            self.assertIn("-t 8", script_text)
            self.assertIn("--task 3", script_text)
            self.assertIn("--max-memory 27", script_text)
            self.assertIn("--cohort-size 3", script_text)
            self.assertIn("--tools metabat2 metadecoder vamb semibin2", script_text)
            command_text = script_text.split(
                "# Reproducible resolved command:\n",
                1,
            )[1]
            self.assertNotIn("--auto", command_text)
            for option in (
                "--magscot-dir",
                "--checkm2-db",
                "--gunc-db",
                "--comebin-env",
                "--lorbin-env",
                "--metawrap-env",
                "--checkm2-env",
            ):
                self.assertNotIn(option, command_text)

    def test_auto_script_records_effective_gpu_batch_and_uncapped_memory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reads = root / "reads"
            contigs = root / "contigs"
            reads.mkdir()
            contigs.mkdir()
            for sample in ("A", "B", "C"):
                (reads / f"{sample}_R1.fastq.gz").write_bytes(b"")
                (reads / f"{sample}_R2.fastq.gz").write_bytes(b"")
                (contigs / f"{sample}.fa").write_text(
                    ">c1\n" + "A" * 10_000 + "\n>c2\n" + "C" * 2_000 + "\n",
                    encoding="utf-8",
                )
            args = build_parser().parse_args(
                [
                    "bin",
                    "-p",
                    str(reads),
                    "-s",
                    "fastq.gz",
                    "-c",
                    str(contigs),
                    "-f",
                    "fa",
                    "--auto",
                    "--dry-run",
                    "-o",
                    str(root / "result"),
                ]
            )
            resources = ServerResourceProfile(
                cpu_total=64,
                cpu_affinity=64,
                cpu_load_1m=16.0,
                cpu_available=48,
                memory_total_gib=512,
                memory_available_gib=480,
                disk_free_gib=500,
                disk_path=root,
                gpu_available=True,
                gpu_count=1,
                gpu_total_memory_mib=49140,
                gpu_free_memory_mib=49140,
                gpu_names=("NVIDIA RTX A6000",),
                cuda_driver_version="570.133.07",
                advertised_cuda_version="12.8",
                cuda_error=None,
            )
            with patch(
                "metabaw.cli.profile_server_resources",
                return_value=resources,
            ), redirect_stdout(io.StringIO()):
                self.assertEqual(0, command_bin(args))
            script_text = (
                root / "result" / "metabaw_auto_run.sh"
            ).read_text(encoding="utf-8")
            self.assertIn("--max-memory 136", script_text)
            self.assertIn("--batch-size 256", script_text)
            self.assertIn("comebin_batch_size_requested=1024", script_text)
            self.assertIn("comebin_batch_size_effective=256", script_text)
            self.assertIn("Automatically resolved effective parameters", script_text)

    def test_auto_preserves_every_explicit_resource_and_strategy_override(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reads = root / "reads"
            contigs = root / "contigs"
            reads.mkdir()
            contigs.mkdir()
            for sample in ("A", "B", "C"):
                (reads / f"{sample}_R1.fastq.gz").write_bytes(b"")
                (reads / f"{sample}_R2.fastq.gz").write_bytes(b"")
                (contigs / f"{sample}.fa").write_text(
                    ">c1\n" + "A" * 12_000 + "\n",
                    encoding="utf-8",
                )
            args = build_parser().parse_args(
                [
                    "bin",
                    "-p",
                    str(reads),
                    "-s",
                    "fastq.gz",
                    "-c",
                    str(contigs),
                    "-f",
                    "fa",
                    "--auto",
                    "--threads",
                    "3",
                    "--task",
                    "5",
                    "--max-memory",
                    "17",
                    "--no-gpu",
                    "--max-gpu-memory",
                    "7G",
                    "--batch-size",
                    "64",
                    "--retries",
                    "4",
                    "--single",
                    "--cohort-size",
                    "9",
                    "--tools",
                    "metabat2",
                    "--min-contig-length",
                    "2500",
                    "--dry-run",
                    "-o",
                    str(root / "result"),
                ]
            )
            resources = ServerResourceProfile(
                cpu_total=64,
                cpu_affinity=64,
                cpu_load_1m=0.0,
                cpu_available=64,
                memory_total_gib=512,
                memory_available_gib=480,
                disk_free_gib=500,
                disk_path=root,
                gpu_available=True,
                gpu_count=1,
                gpu_total_memory_mib=49140,
                gpu_free_memory_mib=49140,
                gpu_names=("NVIDIA RTX A6000",),
                cuda_driver_version="570.133.07",
                advertised_cuda_version="12.8",
                cuda_error=None,
            )
            with patch(
                "metabaw.cli.profile_server_resources",
                return_value=resources,
            ), redirect_stdout(io.StringIO()):
                self.assertEqual(0, command_bin(args))
            self.assertEqual(3, args.threads)
            self.assertEqual(5, args.task)
            self.assertEqual(17, args.max_memory)
            self.assertFalse(args.gpu)
            self.assertEqual("7G", args.max_gpu_memory)
            self.assertEqual(64, args.batch_size)
            self.assertEqual(4, args.retries)
            self.assertEqual("single", args.mode)
            self.assertEqual(9, args.cohort_size)
            self.assertEqual(["metabat2"], args.tools)
            self.assertEqual(2500, args.min_contig_length)
            script_text = (
                root / "result" / "metabaw_auto_run.sh"
            ).read_text(encoding="utf-8")
            self.assertIn("--task 5", script_text)
            self.assertIn("--no-gpu", script_text)
            self.assertIn("#   task=5", script_text)
            self.assertIn("#   gpu=false", script_text)
            self.assertIn("#   group_size=9", script_text)
            self.assertIn(
                'manual_overrides=["batch_size", "gpu", "group_size", '
                '"max_gpu_memory", "max_memory", "min_contig_length", '
                '"mode", "retries", "task", "threads", "tools"]',
                script_text,
            )

    def test_annotation_cli_dry_run_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mags = root / "mags"
            reads = root / "reads"
            mags.mkdir()
            reads.mkdir()
            (mags / "bin.fa").write_text(">c\nAAAA\n", encoding="utf-8")
            for sample in ("A", "B"):
                (reads / f"{sample}.fastq.gz").write_bytes(b"")
            args = build_parser().parse_args(
                [
                    "annotation",
                    "-p",
                    str(mags),
                    "-r",
                    str(reads),
                    "--dry-run",
                    "--tmp-files",
                    str(root / "visible_tmp"),
                    "-o",
                    str(root / "annotation"),
                ]
            )
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(0, command_annotation(args))
            plan = output.getvalue()
            self.assertIn("03.annotation.merge", plan)
            self.assertIn("04.annotation.niche", plan)
            self.assertIn(str((root / "visible_tmp").resolve()), plan)
            self.assertNotIn(".metabaw", plan)


if __name__ == "__main__":
    unittest.main()
