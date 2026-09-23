from __future__ import annotations

import io
import os
import subprocess
import tempfile
import tomllib
import unittest
from unittest.mock import call, patch
from pathlib import Path

from metabaw.dependencies import (
    CudaRuntimeStatus,
    ISOLATED_TOOLS,
    IsolatedEnvironmentStatus,
    SoftwareRequirement,
    SoftwareRuntimeIssue,
    bin_requirements,
    install_isolated_cuda_runtime,
    normalize_isolated_environment,
    install_main_cuda_runtime,
    install_software,
    persist_conda_environment_variable,
    print_isolated_status,
    run_conda_transaction,
    software_runtime_issues,
)
from metabaw.cli import (
    _check_isolated_tools_interactively,
    _check_software_interactively,
    _offer_cuda_repair,
    build_parser,
)


class _SuccessfulProcess:
    def __init__(self) -> None:
        self.stdout = io.BytesIO(b"")

    def wait(self) -> int:
        return 0


class CondaInstallerTests(unittest.TestCase):
    def test_gtdbtk_1_is_reported_incompatible_with_current_database(self) -> None:
        requirement = SoftwareRequirement(
            "gtdbtk",
            "gtdbtk=2.7.2",
            "GTDB-Tk taxonomy",
        )
        probe = subprocess.CompletedProcess(
            ["/env/bin/gtdbtk", "--version"],
            0,
            "GTDB-Tk v1.0.2\n",
            "",
        )
        with (
            patch("metabaw.dependencies.shutil.which", return_value="/env/bin/gtdbtk"),
            patch("metabaw.dependencies.subprocess.run", return_value=probe),
        ):
            issues = software_runtime_issues([requirement])
        self.assertEqual(len(issues), 1)
        self.assertIn("requires version 2.7.2", issues[0].detail)
        self.assertEqual(issues[0].repair_packages, ("gtdbtk=2.7.2",))

    def test_gtdbtk_272_runtime_probe_is_healthy(self) -> None:
        requirement = SoftwareRequirement(
            "gtdbtk",
            "gtdbtk=2.7.2",
            "GTDB-Tk taxonomy",
        )
        probe = subprocess.CompletedProcess(
            ["/env/bin/gtdbtk", "--version"],
            0,
            "GTDB-Tk v2.7.2\n",
            "",
        )
        with (
            patch("metabaw.dependencies.shutil.which", return_value="/env/bin/gtdbtk"),
            patch("metabaw.dependencies.subprocess.run", return_value=probe),
        ):
            issues = software_runtime_issues([requirement])
        self.assertEqual(issues, [])

    def test_gtdbtk_uses_owning_conda_prefix_version_when_cli_text_is_ambiguous(self) -> None:
        requirement = SoftwareRequirement(
            "gtdbtk",
            "gtdbtk=2.7.2",
            "GTDB-Tk taxonomy",
        )
        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary)
            executable = prefix / "bin" / "gtdbtk"
            executable.parent.mkdir(parents=True)
            executable.write_text("#!/bin/sh\n", encoding="utf-8")
            metadata_directory = prefix / "conda-meta"
            metadata_directory.mkdir()
            (metadata_directory / "gtdbtk-2.7.2-test.json").write_text(
                '{"name": "gtdbtk", "version": "2.7.2"}',
                encoding="utf-8",
            )
            probe = subprocess.CompletedProcess(
                [str(executable), "--version"],
                0,
                "GTDB-Tk installed\n",
                "",
            )
            with (
                patch("metabaw.dependencies.shutil.which", return_value=str(executable)),
                patch("metabaw.dependencies.subprocess.run", return_value=probe),
            ):
                issues = software_runtime_issues([requirement])

        self.assertEqual(issues, [])

    def test_invalid_gtdbtk_database_is_not_a_software_runtime_issue(self) -> None:
        requirement = SoftwareRequirement(
            "gtdbtk",
            "gtdbtk=2.7.2",
            "GTDB-Tk taxonomy",
        )
        probe = subprocess.CompletedProcess(
            ["/env/bin/gtdbtk", "--version"],
            1,
            "",
            (
                "The GTDB-Tk reference data does not exist or is corrupted.\n"
                "GTDBTK_DATA_PATH=/env/share/gtdbtk-1.0.2/db"
            ),
        )
        with (
            patch("metabaw.dependencies.shutil.which", return_value="/env/bin/gtdbtk"),
            patch("metabaw.dependencies.configured_database_path", return_value=None),
            patch("metabaw.dependencies.subprocess.run", return_value=probe),
        ):
            issues = software_runtime_issues([requirement])

        self.assertEqual(issues, [])

    def test_gtdbtk_runtime_probe_uses_saved_valid_database(self) -> None:
        requirement = SoftwareRequirement(
            "gtdbtk",
            "gtdbtk=2.7.2",
            "GTDB-Tk taxonomy",
        )
        database = Path("/db/release232")
        probe = subprocess.CompletedProcess(
            ["/env/bin/gtdbtk", "--version"],
            0,
            "GTDB-Tk v2.7.2\n",
            "",
        )
        with (
            patch("metabaw.dependencies.shutil.which", return_value="/env/bin/gtdbtk"),
            patch(
                "metabaw.dependencies.configured_database_path",
                return_value=database,
            ),
            patch("metabaw.dependencies.gtdbtk_database_valid", return_value=True),
            patch("metabaw.dependencies.subprocess.run", return_value=probe) as run,
        ):
            issues = software_runtime_issues([requirement])

        self.assertEqual(issues, [])
        self.assertEqual(
            run.call_args.kwargs["env"]["GTDBTK_DATA_PATH"],
            str(database),
        )

    def test_gtdbtk_database_path_is_persisted_in_running_conda_prefix(self) -> None:
        completed = subprocess.CompletedProcess([], 0, "", "")
        with (
            patch.dict(os.environ, {}, clear=False),
            patch(
                "metabaw.dependencies._running_conda_prefix",
                return_value=Path("/env/metabaw"),
            ),
            patch("metabaw.dependencies.shutil.which", return_value="/base/bin/conda"),
            patch("metabaw.dependencies.subprocess.run", return_value=completed) as run,
        ):
            persisted, prefix = persist_conda_environment_variable(
                "GTDBTK_DATA_PATH",
                "/db/release232",
            )
            active_value = os.environ["GTDBTK_DATA_PATH"]

        self.assertTrue(persisted)
        self.assertEqual(prefix, str(Path("/env/metabaw")))
        self.assertEqual(active_value, "/db/release232")
        self.assertEqual(
            run.call_args.args[0],
            [
                "/base/bin/conda",
                "env",
                "config",
                "vars",
                "set",
                "--prefix",
                str(Path("/env/metabaw")),
                "GTDBTK_DATA_PATH=/db/release232",
            ],
        )

    def test_isolated_status_does_not_pad_the_status_label(self) -> None:
        status = IsolatedEnvironmentStatus(
            ISOLATED_TOOLS["comebin"],
            "metabaw-comebin-py37",
            "mamba",
            "3.7",
            "/env/bin/run_comebin.sh",
        )
        output = io.StringIO()
        with patch("sys.stdout", output):
            print_isolated_status(status)
        self.assertTrue(output.getvalue().startswith("[OK] COMEBin environment"))
        self.assertNotIn("[OK     ]", output.getvalue())

    def test_comebin_cuda_repair_uses_a_python37_compatible_profile(self) -> None:
        completed = subprocess.CompletedProcess([], 0, "", None)
        with patch(
            "metabaw.dependencies.run_conda_transaction",
            return_value=completed,
        ) as transaction:
            install_isolated_cuda_runtime(
                "comebin",
                "metabaw-comebin-py37",
                "mamba",
            )

        self.assertEqual(
            transaction.call_args.args[0],
            [
                "mamba",
                "install",
                "--yes",
                "--name",
                "metabaw-comebin-py37",
                "--channel",
                "pytorch",
                "--channel",
                "conda-forge",
                "pytorch==1.10.2",
                "cudatoolkit=11.3",
            ],
        )

    def test_main_cuda_repair_uses_the_official_pinned_profile(self) -> None:
        completed = subprocess.CompletedProcess([], 0, "", None)
        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary)
            (prefix / "conda-meta").mkdir()
            with (
                patch("metabaw.dependencies.sys.prefix", str(prefix)),
                patch(
                    "metabaw.dependencies.run_conda_transaction",
                    return_value=completed,
                ) as transaction,
            ):
                install_main_cuda_runtime("mamba")

        self.assertEqual(
            transaction.call_args.args[0],
            [
                "mamba",
                "install",
                "--yes",
                "--prefix",
                str(prefix.resolve()),
                "--channel",
                "pytorch",
                "--channel",
                "nvidia",
                "pytorch==2.5.1",
                "torchvision==0.20.1",
                "torchaudio==2.5.1",
                "pytorch-cuda=11.8",
            ],
        )

    def test_comebin_cuda_repair_falls_back_to_the_cu113_wheel(self) -> None:
        failed = subprocess.CompletedProcess([], 1, "", None)
        succeeded = subprocess.CompletedProcess([], 0, "", None)
        with (
            patch(
                "metabaw.dependencies.run_conda_transaction",
                return_value=failed,
            ),
            patch(
                "metabaw.dependencies.subprocess.run",
                return_value=succeeded,
            ) as pip_run,
        ):
            install_isolated_cuda_runtime(
                "comebin",
                "metabaw-comebin-py37",
                "mamba",
            )

        self.assertEqual(
            pip_run.call_args.args[0],
            [
                "mamba",
                "run",
                "--name",
                "metabaw-comebin-py37",
                "python",
                "-m",
                "pip",
                "install",
                "--upgrade",
                "--ignore-installed",
                "--no-cache-dir",
                "torch==1.10.2+cu113",
                "--find-links",
                "https://download.pytorch.org/whl/cu113/torch_stable.html",
            ],
        )

    def test_main_cuda_repair_falls_back_to_the_official_wheel(self) -> None:
        failed = subprocess.CompletedProcess([], 1, "", None)
        succeeded = subprocess.CompletedProcess([], 0, "", None)
        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary)
            (prefix / "conda-meta").mkdir()
            with (
                patch("metabaw.dependencies.sys.prefix", str(prefix)),
                patch(
                    "metabaw.dependencies.sys.executable",
                    str(prefix / "bin" / "python"),
                ),
                patch(
                    "metabaw.dependencies.run_conda_transaction",
                    return_value=failed,
                ),
                patch(
                    "metabaw.dependencies.subprocess.run",
                    return_value=succeeded,
                ) as pip_run,
            ):
                install_main_cuda_runtime("mamba")

        command = pip_run.call_args.args[0]
        self.assertIn("torch==2.5.1", command)
        self.assertIn("torchvision==0.20.1", command)
        self.assertIn("torchaudio==2.5.1", command)
        self.assertIn("https://download.pytorch.org/whl/cu118", command)

    def test_main_cuda_repair_retries_with_pip_after_false_conda_success(self) -> None:
        missing = CudaRuntimeStatus(
            "MetaBAW environment", "3.11", "2.5.1", None, None,
            0, (), False, "torch.cuda.is_available() returned False",
        )
        repaired = CudaRuntimeStatus(
            "MetaBAW environment", "3.11", "2.5.1+cu118", "11.8", None,
            1, ("NVIDIA RTX A6000",), True, None,
        )
        with (
            patch("metabaw.cli.confirm_install", return_value=True),
            patch(
                "metabaw.cli.install_main_cuda_runtime",
                side_effect=(False, True),
            ) as install,
            patch(
                "metabaw.cli._main_cuda_status",
                side_effect=(missing, repaired),
            ),
            patch("sys.stdout", io.StringIO()),
        ):
            result = _offer_cuda_repair(missing)

        self.assertIs(result, repaired)
        self.assertEqual(
            install.call_args_list,
            [call(), call(pip_only=True)],
        )

    def test_main_cuda_pip_only_repair_skips_conda_and_broken_uninstall(self) -> None:
        succeeded = subprocess.CompletedProcess([], 0, "", None)
        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary)
            (prefix / "conda-meta").mkdir()
            with (
                patch("metabaw.dependencies.sys.prefix", str(prefix)),
                patch(
                    "metabaw.dependencies.sys.executable",
                    str(prefix / "bin" / "python"),
                ),
                patch("metabaw.dependencies.run_conda_transaction") as transaction,
                patch(
                    "metabaw.dependencies.subprocess.run",
                    return_value=succeeded,
                ) as pip_run,
            ):
                used_pip = install_main_cuda_runtime(pip_only=True)

        self.assertTrue(used_pip)
        transaction.assert_not_called()
        command = pip_run.call_args.args[0]
        self.assertIn("--ignore-installed", command)
        self.assertNotIn("--force-reinstall", command)
        self.assertIn("https://download.pytorch.org/whl/cu118", command)

    def test_package_metadata_pins_the_main_runtime_to_python_311(self) -> None:
        pyproject = tomllib.loads(
            (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            pyproject["project"]["requires-python"],
            ">=3.11,<3.12",
        )

    def test_user_group_coassembly_requires_megahit_in_main_environment(self) -> None:
        args = build_parser().parse_args(
            [
                "binning",
                "--input_reads_files",
                "reads.tsv",
                "--input_contig_files",
                "contigs.tsv",
                "--assembly-strategy",
                "coassembly",
                "--coassembly-file",
                "groups.tsv",
            ]
        )
        args.align_tool = "bowtie2"
        args.type = "short"
        args.tools = ["metabat2", "metadecoder", "vamb"]
        executables = {
            requirement.executable for requirement in bin_requirements(args)
        }
        self.assertIn("megahit", executables)
        args.assembly_strategy = "existing-contigs"
        self.assertNotIn("megahit", {item.executable for item in bin_requirements(args)})

    def test_supported_isolated_environment_registry_and_normalization(self) -> None:
        self.assertEqual(set(ISOLATED_TOOLS), {"comebin", "checkm2", "metawrap", "lorbin"})
        args = build_parser().parse_args(["check", "--all"])
        self.assertEqual({key for key in vars(args) if key.endswith("_env")},
                         {f"{key}_env" for key in ISOLATED_TOOLS})
        self.assertEqual(normalize_isolated_environment("checkm2", " /envs/checkm2 "), "/envs/checkm2")
        with self.assertRaisesRegex(ValueError, "Unknown isolated tool"):
            normalize_isolated_environment("unknown", "unused")

    def test_transaction_preserves_user_conda_configuration(self) -> None:
        captured_environment: dict[str, str] = {}

        def fake_popen(_command, **kwargs):
            captured_environment.update(kwargs["env"])
            return _SuccessfulProcess()

        user_environment = {
            "CONDA_PKGS_DIRS": "/existing/conda/pkgs",
            "CONDA_NO_PLUGINS": "false",
            "CONDA_REMOTE_READ_TIMEOUT_SECS": "777",
        }
        with patch.dict(os.environ, user_environment, clear=True):
            with patch("metabaw.dependencies.subprocess.Popen", side_effect=fake_popen):
                result = run_conda_transaction(
                    ("mamba", "install", "--yes", "metabat2"),
                    retry_count=0,
                )

        self.assertEqual(result.returncode, 0)
        for name, value in user_environment.items():
            self.assertEqual(captured_environment[name], value)

    def test_missing_packages_are_installed_as_restartable_transactions(self) -> None:
        requirements = [
            SoftwareRequirement("tool-a", "package-a", "test tool A"),
            SoftwareRequirement("tool-b", "package-b", "test tool B"),
        ]
        completed = subprocess.CompletedProcess([], 0, "", None)

        with patch(
            "metabaw.dependencies.missing_software",
            side_effect=[requirements, []],
        ):
            with patch("metabaw.dependencies.software_runtime_issues", return_value=[]):
                with patch("metabaw.dependencies.conda_frontend", return_value="mamba"):
                    with patch(
                        "metabaw.dependencies.run_conda_transaction",
                        return_value=completed,
                    ) as transaction:
                        remaining = install_software(requirements)

        self.assertEqual(remaining, [])
        self.assertEqual(transaction.call_count, 2)
        commands = [call.args[0] for call in transaction.call_args_list]
        self.assertEqual(commands[0][-1], "package-a")
        self.assertEqual(commands[1][-1], "package-b")

    def test_failed_package_does_not_stop_later_installation_transactions(self) -> None:
        requirements = [
            SoftwareRequirement("tool-a", "package-a", "test tool A"),
            SoftwareRequirement("tool-b", "package-b", "test tool B"),
        ]
        failed = subprocess.CompletedProcess([], 1, "", None)
        completed = subprocess.CompletedProcess([], 0, "", None)

        with patch(
            "metabaw.dependencies.missing_software",
            side_effect=[requirements, requirements[:1]],
        ):
            with patch("metabaw.dependencies.software_runtime_issues", return_value=[]):
                with patch("metabaw.dependencies.conda_frontend", return_value="mamba"):
                    with patch(
                        "metabaw.dependencies.run_conda_transaction",
                        side_effect=[failed, completed],
                    ) as transaction:
                        remaining = install_software(requirements)

        self.assertEqual(remaining, requirements[:1])
        self.assertEqual(transaction.call_count, 2)
        self.assertEqual(transaction.call_args_list[1].args[0][-1], "package-b")

    def test_isolated_installation_is_confirmed_once_and_attempts_every_environment(self) -> None:
        selected = ["comebin", "checkm2", "metawrap"]
        environments = {key: f"test-{key}" for key in selected}

        def unavailable(spec, environment):
            return IsolatedEnvironmentStatus(
                spec,
                environment,
                "mamba",
                None,
                None,
                "environment not found",
            )

        def install(spec, environment, frontend):
            if spec.key == "checkm2":
                raise RuntimeError("simulated failure")
            return IsolatedEnvironmentStatus(
                spec,
                environment,
                frontend,
                spec.python_version,
                f"/envs/{environment}/bin/{spec.executable}",
                None,
            )

        with (
            patch("metabaw.cli.isolated_environment_status", side_effect=unavailable),
            patch("metabaw.cli.confirm_install", return_value=True) as confirm,
            patch("metabaw.cli.install_isolated_environment", side_effect=install) as installer,
        ):
            statuses, available = _check_isolated_tools_interactively(
                selected,
                environments,
            )

        self.assertFalse(available)
        self.assertTrue(statuses["comebin"].available)
        self.assertFalse(statuses["checkm2"].available)
        self.assertTrue(statuses["metawrap"].available)
        confirm.assert_called_once()
        self.assertEqual(installer.call_count, 3)

    def test_new_runtime_issue_is_repaired_in_the_same_check_invocation(self) -> None:
        requirement = SoftwareRequirement(
            "gtdbtk",
            "gtdbtk=2.7.2",
            "GTDB-Tk taxonomy",
        )
        runtime_issue = SoftwareRuntimeIssue(
            "gtdbtk",
            "simulated post-install runtime issue",
            ("gtdbtk=2.7.2",),
        )
        with (
            patch("metabaw.cli.print_software_status"),
            patch(
                "metabaw.cli.missing_software",
                side_effect=[[requirement], [], []],
            ),
            patch(
                "metabaw.cli.software_runtime_issues",
                side_effect=[[], [runtime_issue], []],
            ),
            patch("metabaw.cli.confirm_install", return_value=True) as confirm,
            patch("metabaw.cli.install_software", return_value=[]) as installer,
        ):
            available = _check_software_interactively([requirement])

        self.assertTrue(available)
        confirm.assert_called_once()
        self.assertEqual(installer.call_count, 2)

    def test_vamb_uses_pip_instead_of_conda(self) -> None:
        requirements = [SoftwareRequirement("vamb", "vamb", "VAMB binning")]
        completed = subprocess.CompletedProcess([], 0, "", None)

        with patch(
            "metabaw.dependencies.missing_software",
            side_effect=[requirements, []],
        ):
            with patch("metabaw.dependencies.software_runtime_issues", return_value=[]):
                with patch("metabaw.dependencies.subprocess.run", return_value=completed) as run:
                    with patch("metabaw.dependencies.run_conda_transaction") as transaction:
                        remaining = install_software(requirements)

        self.assertEqual(remaining, [])
        transaction.assert_not_called()
        command = run.call_args.args[0]
        self.assertEqual(command[:4], [os.sys.executable, "-m", "pip", "install"])
        self.assertEqual(command[-1], "vamb")
        self.assertNotIn("--upgrade", command)

    def test_selected_bioconda_tools_use_the_requested_channel_order(self) -> None:
        requirements = [
            SoftwareRequirement("SemiBin2", "semibin", "SemiBin2 binning"),
            SoftwareRequirement("gunc", "gunc", "GUNC contamination checking"),
            SoftwareRequirement("gtdbtk", "gtdbtk=2.7.2", "GTDB-Tk taxonomy"),
            SoftwareRequirement("galah", "galah", "MAG dereplication"),
        ]
        completed = subprocess.CompletedProcess([], 0, "", None)

        with patch(
            "metabaw.dependencies.missing_software",
            side_effect=[requirements, []],
        ):
            with patch("metabaw.dependencies.software_runtime_issues", return_value=[]):
                with patch("metabaw.dependencies.conda_frontend", return_value="mamba"):
                    with patch(
                        "metabaw.dependencies.run_conda_transaction",
                        return_value=completed,
                    ) as transaction:
                        remaining = install_software(requirements)

        self.assertEqual(remaining, [])
        commands = [call.args[0] for call in transaction.call_args_list]
        self.assertEqual(
            commands,
            [
                [
                    "mamba",
                    "install",
                    "-y",
                    "-c",
                    "bioconda",
                    "-c",
                    "conda-forge",
                    package,
                ]
                for package in (
                    "semibin",
                    "gunc",
                    "gtdbtk=2.7.2",
                    "galah",
                )
            ],
        )


if __name__ == "__main__":
    unittest.main()
