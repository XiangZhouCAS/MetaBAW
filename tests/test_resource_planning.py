from __future__ import annotations

from argparse import Namespace
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from metabaw.cli import (
    _comebin_minimum_memory_gib,
    _apply_gpu_memory_minima,
    _gpu_task_slots,
    _run_direct,
    _thread_budget,
)
from metabaw.dependencies import CudaDevice, HostCudaStatus
from metabaw.direct import DirectBinBuilder
from metabaw.model import Task


class ResourcePlanningTests(unittest.TestCase):
    def test_total_threads_are_divided_across_concurrent_samples(self) -> None:
        self.assertEqual(_thread_budget(128, 3, 3), (42, 3))
        self.assertEqual(_thread_budget(128, 8, 3), (42, 3))
        self.assertEqual(_thread_budget(128, 3, 320), (42, 3))
        self.assertEqual(_thread_budget(2, 3, 3), (1, 2))

    def test_gpu_training_is_serialized_without_device_affinity(self) -> None:
        status = HostCudaStatus(
            executable="nvidia-smi",
            devices=(
                CudaDevice("0", "GPU 0", 48 * 1024, "580", 40 * 1024),
                CudaDevice("1", "GPU 1", 48 * 1024, "580", 40 * 1024),
            ),
            advertised_cuda_version="13.0",
        )
        self.assertEqual(_gpu_task_slots(status, 1, "4G", 3), 1)
        self.assertEqual(_gpu_task_slots(status, 2, "4G", 3), 1)
        self.assertEqual(_gpu_task_slots(status, 2, "4G", 1), 1)

    def test_low_gpu_budget_disables_only_binners_below_their_minimum(self) -> None:
        args = Namespace(
            tools=["vamb", "comebin", "semibin2", "lorbin"],
            max_gpu_memory="4G",
            batch_size=1024,
            requested_batch_size=1024,
            parameter_warnings=[],
        )
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            enabled = _apply_gpu_memory_minima(args, 48 * 1024)

        self.assertEqual(enabled, ("vamb", "semibin2", "lorbin"))
        self.assertEqual(args.gpu_binners, enabled)
        self.assertIn("COMEBin will use CPU", stderr.getvalue())
        self.assertNotIn("VAMB will use CPU", stderr.getvalue())

    def test_budget_below_four_gib_disables_all_gpu_binners(self) -> None:
        args = Namespace(
            tools=["vamb", "comebin", "semibin2", "lorbin"],
            max_gpu_memory="3G",
            batch_size=1024,
            requested_batch_size=1024,
            parameter_warnings=[],
        )
        with redirect_stderr(io.StringIO()):
            enabled = _apply_gpu_memory_minima(args, 48 * 1024)
        self.assertEqual(enabled, ())

    def test_comebin_minimum_memory_tracks_requested_batch_size(self) -> None:
        expected = {
            128: 1.0,
            256: 2.0,
            512: 4.0,
            1024: 8.0,
            2048: 16.0,
        }
        for batch_size, minimum_gib in expected.items():
            with self.subTest(batch_size=batch_size):
                self.assertEqual(
                    _comebin_minimum_memory_gib(batch_size),
                    minimum_gib,
                )

    def test_comebin_falls_back_to_cpu_when_dynamic_minimum_is_not_met(self) -> None:
        cases = (
            (512, "4G", True),
            (512, "3G", False),
            (1024, "8G", True),
            (1024, "4G", False),
            (2048, "16G", True),
            (2048, "8G", False),
        )
        for batch_size, budget, should_use_gpu in cases:
            with self.subTest(batch_size=batch_size, budget=budget):
                args = Namespace(
                    tools=["comebin"],
                    max_gpu_memory=budget,
                    batch_size=batch_size,
                    requested_batch_size=batch_size,
                    parameter_warnings=[],
                )
                stderr = io.StringIO()
                with redirect_stderr(stderr):
                    enabled = _apply_gpu_memory_minima(args, 48 * 1024)

                self.assertEqual(bool(enabled), should_use_gpu)
                if not should_use_gpu:
                    self.assertIn(
                        f"at --batch-size {batch_size}",
                        stderr.getvalue(),
                    )

    def test_builder_hides_cuda_only_from_cpu_fallback_binner(self) -> None:
        options = Namespace(
            gpu=True,
            gpu_binners=("vamb", "semibin2"),
            max_gpu_memory="4G",
            threads=42,
            total_threads=128,
        )
        builder = DirectBinBuilder([], options, Path("."))

        self.assertTrue(builder._uses_gpu("vamb"))
        self.assertFalse(builder._uses_gpu("comebin"))
        self.assertNotEqual(
            builder._gpu_env("vamb")["CUDA_VISIBLE_DEVICES"],
            "",
        )
        self.assertEqual(
            builder._gpu_env("comebin")["CUDA_VISIBLE_DEVICES"],
            "",
        )
        self.assertEqual(builder._cohort_threads(), 128)

    def test_executor_receives_total_threads_without_parallel_multiplier(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = Namespace(
                dry_run=False,
                threads=128,
                threads_per_task=42,
                task=3,
                concurrent_sample_limit=3,
                max_memory=100.0,
                gpu=False,
                cuda_device_count=0,
                cuda_task_slots=0,
                delete_tmp_files=False,
                retries=0,
                force=False,
            )
            task = Task(
                id="01.sample.A",
                stage="01_sample",
                command=("true",),
                cwd=root,
                sample="A",
                cpus=42,
            )
            state = MagicMock()
            executor = MagicMock()
            executor.run.return_value = {task.id: "success"}
            output = io.StringIO()
            with (
                patch("metabaw.cli.StateStore", return_value=state),
                patch("metabaw.cli.Executor", return_value=executor) as constructor,
                redirect_stdout(output),
            ):
                result = _run_direct(
                    args,
                    [task],
                    root / "output",
                    root / "output" / "tmp",
                    {"module": "bin"},
                )

            self.assertEqual(result, 0)
            self.assertEqual(constructor.call_args.kwargs["max_cpus"], 128)
            self.assertEqual(constructor.call_args.kwargs["max_parallel"], 3)
            manifest = json.loads(
                (root / "output" / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["resources"]["total_memory_gb"], 100.0)
            self.assertEqual(
                manifest["resources"]["memory_scope"],
                "combined_pss_of_workflow_task_process_trees",
            )
            self.assertEqual(
                manifest["resources"]["estimated_minimum_total_memory_gb"],
                0.0,
            )
            self.assertNotIn("total workflow PSS budget", output.getvalue())
            self.assertIn("[START INFO]", output.getvalue())
            self.assertIn("total workflow PSS budget", (root / "output" / "start_info.txt").read_text())

    def test_known_memory_shortfall_is_reported_before_executor_starts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = Namespace(dry_run=False, max_memory=100.0)
            task = Task(
                id="01.annotation.gtdbtk",
                stage="01_taxonomy",
                command=("gtdbtk",),
                cwd=root,
                description="Classify MAGs with GTDB-Tk",
                minimum_memory_gb=128.0,
                memory_requirement_hint=(
                    "increase --max-memory to at least 128 GiB, or provide "
                    "a complete matching --gtdbtk_res path to skip GTDB-Tk"
                ),
            )
            stdout = io.StringIO()
            stderr = io.StringIO()
            with (
                patch("metabaw.cli.Executor") as constructor,
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                result = _run_direct(
                    args,
                    [task],
                    root / "output",
                    root / "output" / "tmp",
                    {"module": "annotation"},
                )

            self.assertEqual(result, 1)
            constructor.assert_not_called()
            startup = (root / "output" / "start_info.txt").read_text()
            self.assertNotIn("[MEMORY ESTIMATE]", stdout.getvalue())
            self.assertIn("[MEMORY ESTIMATE]", startup)
            self.assertIn("estimated minimum total workflow memory=128 GiB", startup)
            self.assertIn("shortfall=28 GiB", startup)
            self.assertIn("[MEMORY CONFIG ERROR]", stderr.getvalue())
            self.assertIn("No workflow task process was started", stderr.getvalue())
            log = root / "output" / "tmp" / "runtime" / "logs" / "memory_preflight.log"
            self.assertIn("[MEMORY CONFIG ERROR]", log.read_text(encoding="utf-8"))

    def test_final_failure_summary_repeats_runtime_memory_reason(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = Namespace(
                dry_run=False,
                threads=8,
                threads_per_task=4,
                task=2,
                concurrent_sample_limit=2,
                max_memory=100.0,
                gpu=False,
                cuda_device_count=0,
                cuda_task_slots=0,
                delete_tmp_files=False,
                retries=0,
                force=False,
            )
            task = Task(
                id="01.sample.A",
                stage="01_sample",
                command=("true",),
                cwd=root,
            )
            state = MagicMock()
            executor = MagicMock()
            executor.run.return_value = {task.id: "blocked"}
            executor.peak_memory_gb = 103.43
            executor.memory_abort_reason = (
                "total workflow PSS=103.43 GiB exceeded --max-memory=100 GiB"
            )
            stderr = io.StringIO()
            with (
                patch("metabaw.cli.StateStore", return_value=state),
                patch("metabaw.cli.Executor", return_value=executor),
                redirect_stdout(io.StringIO()),
                redirect_stderr(stderr),
            ):
                result = _run_direct(
                    args,
                    [task],
                    root / "output",
                    root / "output" / "tmp",
                    {"module": "annotation"},
                )

            self.assertEqual(result, 1)
            self.assertIn("Workflow stopped by --max-memory", stderr.getvalue())
            self.assertIn("total workflow PSS=103.43 GiB", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
