import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
import unittest

from metabaw.dependencies import (
    CHECKM2_ENV_DEFAULT,
    COMEBIN_ENV_DEFAULT,
    ISOLATED_TOOLS,
    LORBIN_ENV_DEFAULT,
    METAWRAP_ENV_DEFAULT,
    RPackageRequirement,
    SOFTWARE,
    SoftwareRuntimeIssue,
    annotation_requirements,
    bin_requirements,
    comebin_environment_status,
    comebin_run_prefix,
    configured_isolated_environment,
    configured_magscot_directory,
    configured_database_path,
    cuda_runtime_status,
    host_cuda_status,
    install_comebin_environment,
    install_isolated_cuda_runtime,
    install_isolated_environment,
    install_r_packages,
    install_software,
    isolated_environment_status,
    isolated_environment_config_path,
    save_isolated_environment,
    save_magscot_directory,
    software_runtime_issues,
)


class DependencyTests(unittest.TestCase):
    def test_barrnap_runtime_probe_detects_missing_perl_module(self) -> None:
        failed = Mock(
            returncode=2,
            stdout="",
            stderr="Can't locate Path/Tiny.pm in @INC\n",
        )
        with patch(
            "metabaw.dependencies.shutil.which",
            return_value="/env/bin/barrnap",
        ), patch(
            "metabaw.dependencies.subprocess.run",
            return_value=failed,
        ) as runner:
            issues = software_runtime_issues([SOFTWARE["barrnap"]])

        self.assertEqual(1, len(issues))
        self.assertIn("Path/Tiny.pm", issues[0].detail)
        self.assertEqual(("barrnap", "perl-path-tiny"), issues[0].repair_packages)
        self.assertEqual(
            ["/env/bin/barrnap", "--version"],
            runner.call_args.args[0],
        )

    def test_r_package_installer_uses_cran_install_packages(self) -> None:
        completed = Mock(returncode=0)
        with patch(
            "metabaw.dependencies.shutil.which",
            return_value="/env/bin/Rscript",
        ), patch(
            "metabaw.dependencies.subprocess.run",
            return_value=completed,
        ) as runner:
            install_r_packages(
                [
                    RPackageRequirement("optparse"),
                    RPackageRequirement("dplyr"),
                ]
            )

        command = runner.call_args.args[0]
        self.assertEqual(["/env/bin/Rscript", "-e"], command[:2])
        self.assertIn("install.packages", command[2])
        self.assertIn('"dplyr"', command[2])
        self.assertIn('"optparse"', command[2])
        self.assertNotIn("BiocManager::install", command[2])

    def test_r_package_installer_bootstraps_biocmanager(self) -> None:
        completed = Mock(returncode=0)
        with patch(
            "metabaw.dependencies.shutil.which",
            return_value="/env/bin/Rscript",
        ), patch(
            "metabaw.dependencies.subprocess.run",
            return_value=completed,
        ) as runner:
            install_r_packages(
                [RPackageRequirement("Biostrings", "bioconductor")]
            )

        expression = runner.call_args.args[0][2]
        self.assertIn("install.packages('BiocManager'", expression)
        self.assertIn("BiocManager::install", expression)
        self.assertIn('"Biostrings"', expression)

    def test_barrnap_runtime_repair_uses_conda_perl_dependency(self) -> None:
        issue = SoftwareRuntimeIssue(
            "barrnap",
            "Can't locate Path/Tiny.pm in @INC",
            ("barrnap", "perl-path-tiny"),
        )
        completed = Mock(returncode=0)
        with patch(
            "metabaw.dependencies.missing_software",
            return_value=[],
        ), patch(
            "metabaw.dependencies.software_runtime_issues",
            return_value=[issue],
        ), patch(
            "metabaw.dependencies.conda_frontend",
            return_value="mamba",
        ), patch(
            "metabaw.dependencies.subprocess.run",
            return_value=completed,
        ) as runner:
            install_software([SOFTWARE["barrnap"]])

        command = runner.call_args.args[0]
        self.assertEqual(["mamba", "install", "--yes"], command[:3])
        self.assertIn("barrnap", command)
        self.assertIn("perl-path-tiny", command)

    def test_install_software_delegates_r_packages_to_rscript(self) -> None:
        with patch(
            "metabaw.dependencies.missing_software",
            return_value=[],
        ), patch(
            "metabaw.dependencies.missing_r_packages",
            return_value=["optparse", "digest"],
        ), patch(
            "metabaw.dependencies.software_runtime_issues",
            return_value=[],
        ), patch(
            "metabaw.dependencies.install_r_packages",
        ) as r_installer, patch(
            "metabaw.dependencies.conda_frontend",
        ) as conda:
            install_software([SOFTWARE["Rscript"]])

        installed = {requirement.name for requirement in r_installer.call_args.args[0]}
        self.assertEqual({"optparse", "digest"}, installed)
        conda.assert_not_called()

    def test_comebin_cuda_repair_uses_the_official_python37_build(self) -> None:
        completed = Mock(returncode=0)
        with patch(
            "metabaw.dependencies.conda_frontend",
            return_value="mamba",
        ), patch(
            "metabaw.dependencies.subprocess.run",
            return_value=completed,
        ) as runner:
            install_isolated_cuda_runtime("comebin", "comebin-py37")
        command = runner.call_args.args[0]
        self.assertIn("pytorch=1.10.2=py3.7_cuda11.1_cudnn8.0.5_0", command)
        self.assertIn("cudatoolkit=11.1.1", command)
        self.assertIn("pytorch-mutex=1.0=cuda", command)

    def test_cuda_runtime_probe_requires_a_real_tensor_allocation(self) -> None:
        payload = {
            "python": "3.7",
            "torch": "1.13.1",
            "torch_cuda": "11.7",
            "visible": "0",
            "available": True,
            "count": 1,
            "devices": ["NVIDIA A100"],
            "allocation": True,
            "error": None,
        }
        completed = Mock(
            returncode=0,
            stdout="METABAW_CUDA_STATUS=" + json.dumps(payload) + "\n",
            stderr="",
        )
        with patch("metabaw.dependencies.subprocess.run", return_value=completed):
            status = cuda_runtime_status("COMEBin", ("conda", "run", "python"))
        self.assertTrue(status.available)
        self.assertEqual("11.7", status.torch_cuda_version)
        self.assertEqual(("NVIDIA A100",), status.devices)

    def test_host_cuda_probe_reports_driver_cuda_and_memory(self) -> None:
        query = Mock(
            returncode=0,
            stdout="0, NVIDIA A100, 40960, 550.54, 32768\n",
            stderr="",
        )
        banner = Mock(
            returncode=0,
            stdout="NVIDIA-SMI 550.54 CUDA Version: 12.4\n",
            stderr="",
        )
        with patch(
            "metabaw.dependencies.shutil.which",
            return_value="/usr/bin/nvidia-smi",
        ), patch(
            "metabaw.dependencies.subprocess.run",
            side_effect=(query, banner),
        ):
            status = host_cuda_status()
        self.assertTrue(status.available)
        self.assertEqual("12.4", status.advertised_cuda_version)
        self.assertEqual(40960, status.devices[0].memory_total_mib)
        self.assertEqual(32768, status.devices[0].memory_free_mib)

    def test_default_bin_dependency_profile_includes_parallel_prodigal(self) -> None:
        args = argparse.Namespace(
            align_tool="bowtie2",
            tools=["metabat2", "metadecoder", "vamb"],
            type="short",
            refinement="magscot",
            quality_control="checkm2",
            gunc=False,
            trna=False,
            rrna=False,
            dereplication_tool="galah",
        )
        names = {requirement.executable for requirement in bin_requirements(args)}
        self.assertTrue(
            {
                "bowtie2",
                "bowtie2-build",
                "samtools",
                "metabat2",
                "jgi_summarize_bam_contig_depths",
                "metadecoder",
                "vamb",
                "strobealign",
                "parallel",
                "prodigal",
                "hmmsearch",
                "Rscript",
                "galah",
            }
            <= names
        )
        self.assertNotIn("checkm2", names)

    def test_dastool_profile_includes_r_and_external_dependencies(self) -> None:
        args = argparse.Namespace(
            align_tool="bowtie2",
            tools=["metabat2"],
            type="short",
            refinement="das_tool",
            quality_control="checkm2",
            gunc=False,
            trna=False,
            rrna=False,
            dereplication_tool="galah",
        )
        names = {requirement.executable for requirement in bin_requirements(args)}
        self.assertTrue(
            {
                "DAS_Tool",
                "Rscript",
                "diamond",
                "prodigal",
                "pullseq",
                "ruby",
            }
            <= names
        )

    def test_annotation_dependency_profile(self) -> None:
        self.assertEqual(
            {"bash", "gtdbtk", "coverm", "minimap2"},
            {
                requirement.executable
                for requirement in annotation_requirements()
            },
        )

    def test_conflicting_tools_are_not_required_in_main_python_environment(self) -> None:
        args = argparse.Namespace(
            align_tool="bowtie2",
            tools=["comebin"],
            type="short",
            refinement="metawrap",
            quality_control="checkm2",
            gunc=False,
            trna=False,
            rrna=False,
            dereplication_tool="galah",
        )
        names = {requirement.executable for requirement in bin_requirements(args)}
        self.assertNotIn("run_comebin.sh", names)
        self.assertNotIn("checkm2", names)
        self.assertNotIn("metawrap", names)

        args.tools = ["lorbin"]
        names = {requirement.executable for requirement in bin_requirements(args)}
        self.assertNotIn("LorBin", names)

    def test_comebin_runner_supports_named_environment_and_prefix(self) -> None:
        self.assertEqual(
            ("mamba", "run", "--name", COMEBIN_ENV_DEFAULT),
            comebin_run_prefix(COMEBIN_ENV_DEFAULT, "mamba"),
        )
        prefix = Path("/envs/comebin").resolve()
        self.assertEqual(
            ("conda", "run", "--prefix", str(prefix)),
            comebin_run_prefix(str(prefix), "conda"),
        )

    def test_comebin_environment_requires_python_37_and_executable(self) -> None:
        probe = Mock(
            returncode=0,
            stdout='{"python": "3.7", "executable": "/env/bin/run_comebin.sh"}\n',
            stderr="",
        )
        with patch(
            "metabaw.dependencies.subprocess.run",
            return_value=probe,
        ) as run:
            status = comebin_environment_status("comebin-py37", "mamba")
        self.assertTrue(status.available)
        self.assertEqual("3.7", status.python_version)
        self.assertEqual(
            [
                "mamba",
                "run",
                "--name",
                "comebin-py37",
                "python",
                "-c",
            ],
            run.call_args.args[0][:6],
        )

    def test_comebin_installer_creates_python_37_environment(self) -> None:
        missing = Mock(returncode=1, stdout="", stderr="environment not found")
        installed = Mock(
            returncode=0,
            stdout='{"python": "3.7", "executable": "/env/bin/run_comebin.sh"}\n',
            stderr="",
        )
        created = Mock(returncode=0, stdout="", stderr="")
        with patch(
            "metabaw.dependencies.subprocess.run",
            side_effect=(missing, created, installed),
        ) as run:
            status = install_comebin_environment("comebin-py37", "mamba")
        self.assertTrue(status.available)
        probe_source = run.call_args_list[0].args[0][-1]
        self.assertIn("distutils.spawn", probe_source)
        self.assertNotIn("sys.version_info.major", probe_source)
        create_command = run.call_args_list[1].args[0]
        self.assertEqual(["mamba", "create", "--yes"], create_command[:3])
        self.assertIn("python=3.7", create_command)
        self.assertIn("comebin", create_command)

    def test_comebin_installer_does_not_downgrade_existing_environment(self) -> None:
        incompatible = Mock(
            returncode=0,
            stdout='{"python": "3.11", "executable": null}\n',
            stderr="",
        )
        with patch(
            "metabaw.dependencies.subprocess.run",
            return_value=incompatible,
        ) as run:
            with self.assertRaisesRegex(RuntimeError, "will not change Python"):
                install_comebin_environment("main-py311", "mamba")
        self.assertEqual(1, run.call_count)

    def test_all_isolated_environment_defaults_and_python_versions(self) -> None:
        self.assertEqual("metabaw-comebin-py37", COMEBIN_ENV_DEFAULT)
        self.assertEqual("metabaw-checkm2-py312", CHECKM2_ENV_DEFAULT)
        self.assertEqual("metabaw-metawrap-py27", METAWRAP_ENV_DEFAULT)
        self.assertEqual("metabaw-lorbin-py310", LORBIN_ENV_DEFAULT)
        self.assertEqual("3.7", ISOLATED_TOOLS["comebin"].python_version)
        self.assertEqual("3.12", ISOLATED_TOOLS["checkm2"].python_version)
        self.assertEqual("2.7", ISOLATED_TOOLS["metawrap"].python_version)
        self.assertEqual("3.10", ISOLATED_TOOLS["lorbin"].python_version)
        self.assertEqual("metawrap-refinement", ISOLATED_TOOLS["metawrap"].package)

    def test_lorbin_installer_uses_official_python_310_environment(self) -> None:
        missing = Mock(returncode=1, stdout="", stderr="environment not found")
        created = Mock(returncode=0, stdout="", stderr="")
        source_installed = Mock(returncode=0, stdout="", stderr="")
        installed = Mock(
            returncode=0,
            stdout='{"python": "3.10", "executable": "/env/bin/LorBin"}\n',
            stderr="",
        )
        with patch(
            "metabaw.dependencies.subprocess.run",
            side_effect=(missing, created, source_installed, installed),
        ) as run:
            status = install_isolated_environment(
                ISOLATED_TOOLS["lorbin"],
                LORBIN_ENV_DEFAULT,
                "mamba",
            )
        self.assertTrue(status.available)
        create_command = run.call_args_list[1].args[0]
        self.assertIn("python=3.10", create_command)
        self.assertIn("biopython=1.83", create_command)
        self.assertNotIn("biopython=1.78", create_command)
        self.assertIn("pytorch=1.11.0", create_command)
        self.assertIn("numpy=1.23.3", create_command)
        self.assertNotIn("lorbin", create_command)
        source_command = run.call_args_list[2].args[0]
        self.assertEqual(
            ["mamba", "run", "--name", LORBIN_ENV_DEFAULT, "python", "-m", "pip"],
            source_command[:7],
        )
        self.assertIn(
            "ee10232282c2b71ed3ce2a34d5dbd78af3dd0b0a.tar.gz",
            source_command[-1],
        )

    def test_isolated_environment_configuration_round_trip_and_precedence(self) -> None:
        with TemporaryDirectory() as directory:
            config_path = Path(directory) / "settings" / "config.json"
            comebin_prefix = Path(directory) / "envs" / "comebin"
            with patch.dict(
                os.environ,
                {"METABAW_CONFIG_FILE": str(config_path)},
                clear=False,
            ):
                self.assertEqual(config_path.resolve(), isolated_environment_config_path())
                saved = save_isolated_environment("comebin", str(comebin_prefix))
                self.assertEqual(str(comebin_prefix.resolve()), saved)
                self.assertEqual(saved, configured_isolated_environment("comebin"))
                payload = json.loads(config_path.read_text(encoding="utf-8"))
                self.assertEqual(
                    saved,
                    payload["isolated_environments"]["comebin"],
                )
                with patch.dict(
                    os.environ,
                    {"METABAW_COMEBIN_ENV": "environment-override"},
                    clear=False,
                ):
                    self.assertEqual(
                        "environment-override",
                        configured_isolated_environment("comebin"),
                    )

    def test_magscot_directory_configuration_round_trip(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "settings" / "config.json"
            magscot = root / "software" / "MAGScoT"
            with patch.dict(
                os.environ,
                {"METABAW_CONFIG_FILE": str(config_path)},
                clear=False,
            ):
                saved = save_magscot_directory(magscot)
                self.assertEqual(magscot.resolve(), saved)
                self.assertEqual(saved, configured_magscot_directory())
                payload = json.loads(config_path.read_text(encoding="utf-8"))
                self.assertEqual(
                    str(magscot.resolve()),
                    payload["magscot_directory"],
                )

    def test_checkm2_environment_uses_python_312(self) -> None:
        probe = Mock(
            returncode=0,
            stdout='{"python": "3.12", "executable": "/env/bin/checkm2"}\n',
            stderr="",
        )
        with patch("metabaw.dependencies.subprocess.run", return_value=probe):
            status = isolated_environment_status(
                ISOLATED_TOOLS["checkm2"],
                CHECKM2_ENV_DEFAULT,
                "mamba",
            )
        self.assertTrue(status.available)

    def test_metawrap_installer_uses_refinement_package_and_python_27(self) -> None:
        missing = Mock(returncode=1, stdout="", stderr="environment not found")
        created = Mock(returncode=0, stdout="", stderr="")
        installed = Mock(
            returncode=0,
            stdout='{"python": "2.7", "executable": "/env/bin/metawrap"}\n',
            stderr="",
        )
        with patch(
            "metabaw.dependencies.subprocess.run",
            side_effect=(missing, created, installed),
        ) as run:
            status = install_isolated_environment(
                ISOLATED_TOOLS["metawrap"],
                METAWRAP_ENV_DEFAULT,
                "mamba",
            )
        self.assertTrue(status.available)
        create_command = run.call_args_list[1].args[0]
        self.assertIn("python=2.7", create_command)
        self.assertIn("metawrap-refinement", create_command)

    def test_installer_uses_available_conda_frontend_immediately(self) -> None:
        requirement = SOFTWARE["bowtie2"]

        def fake_which(executable: str) -> str | None:
            return "/env/bin/mamba" if executable == "mamba" else None

        completed = Mock(returncode=0)
        with patch("metabaw.dependencies.shutil.which", side_effect=fake_which), patch(
            "metabaw.dependencies.subprocess.run",
            return_value=completed,
        ) as run:
            remaining = install_software([requirement])
        self.assertEqual([requirement], remaining)
        command = run.call_args.args[0]
        self.assertEqual("mamba", command[0])
        self.assertIn("install", command)
        self.assertIn("bowtie2", command)

    def test_metadecoder_uses_upstream_wheel_in_main_python_environment(self) -> None:
        requirement = SOFTWARE["metadecoder"]
        completed = Mock(returncode=0)
        with patch(
            "metabaw.dependencies.shutil.which",
            return_value=None,
        ), patch(
            "metabaw.dependencies.subprocess.run",
            return_value=completed,
        ) as run:
            remaining = install_software([requirement])
        self.assertEqual([requirement], remaining)
        command = run.call_args.args[0]
        self.assertEqual("-m", command[1])
        self.assertEqual("pip", command[2])
        self.assertIn("metadecoder-1.2.2-py3-none-any.whl", command[-1])

    def test_database_directory_resolves_to_database_artifact(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "nested" / "database.dmnd"
            database.parent.mkdir()
            database.touch()
            resolved = configured_database_path(
                str(root),
                "UNUSED_DATABASE_VARIABLE",
                artifact_suffix=".dmnd",
            )
        self.assertEqual(database.resolve(), resolved)

    def test_invalid_environment_database_falls_back_to_saved_valid_path(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            stale = root / "old" / "uniref100.KO.1.dmnd"
            saved = root / "checkm2_db" / "uniref100.KO.1.dmnd"
            saved.parent.mkdir()
            saved.touch()
            with patch.dict(
                os.environ,
                {"CHECKM2DB": str(stale)},
                clear=False,
            ), patch(
                "metabaw.dependencies._database_config",
                return_value={"checkm2": str(saved)},
            ):
                resolved = configured_database_path(
                    None,
                    "CHECKM2DB",
                    "checkm2",
                    artifact_suffix=".dmnd",
                )
        self.assertEqual(saved.resolve(), resolved)

    def test_explicit_database_path_remains_highest_priority(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            explicit = root / "explicit.dmnd"
            environment = root / "environment.dmnd"
            saved = root / "saved.dmnd"
            for path in (explicit, environment, saved):
                path.touch()
            with patch.dict(
                os.environ,
                {"CHECKM2DB": str(environment)},
                clear=False,
            ), patch(
                "metabaw.dependencies._database_config",
                return_value={"checkm2": str(saved)},
            ):
                resolved = configured_database_path(
                    str(explicit),
                    "CHECKM2DB",
                    "checkm2",
                    artifact_suffix=".dmnd",
                )
        self.assertEqual(explicit.resolve(), resolved)

    def test_check_saves_explicit_gtdbtk_database_path(self) -> None:
        from metabaw.cli import build_parser, command_check

        with TemporaryDirectory() as directory:
            root = Path(directory)
            gtdbtk = root / "gtdbtk_db"
            gtdbtk.mkdir()
            config = root / "databases.json"
            environment = {
                key: value
                for key, value in os.environ.items()
                if key != "GTDBTK_DATA_PATH"
            }
            with patch.dict(os.environ, environment, clear=True), patch(
                "metabaw.dependencies._database_config_path",
                return_value=config,
            ), patch(
                "metabaw.cli._check_software_interactively",
                return_value=True,
            ):
                args = build_parser().parse_args(
                    ["check", "--scope", "annotation", "--gtdbtk-db", str(gtdbtk)]
                )
                self.assertEqual(0, command_check(args))
            saved = json.loads(config.read_text(encoding="utf-8"))
            self.assertEqual(str(gtdbtk.resolve()), saved["gtdbtk"])

    def test_check_does_not_save_environment_resolved_database(self) -> None:
        from metabaw.cli import build_parser, command_check

        with TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "uniref100.KO.1.dmnd"
            database.touch()
            config = root / "databases.json"
            environment = dict(os.environ)
            environment["CHECKM2DB"] = str(database)
            environment["METABAW_CONFIG_FILE"] = str(root / "config.json")
            with patch.dict(os.environ, environment, clear=True), patch(
                "metabaw.dependencies._database_config_path",
                return_value=config,
            ), patch(
                "metabaw.cli._check_software_interactively",
                return_value=True,
            ), patch(
                "metabaw.cli._ensure_isolated_tool",
                return_value=Mock(),
            ):
                args = build_parser().parse_args(["check", "--scope", "checkm2"])
                self.assertEqual(0, command_check(args))
            self.assertFalse(config.exists())


if __name__ == "__main__":
    unittest.main()
