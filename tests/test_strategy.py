from pathlib import Path
import tempfile
import unittest

from metabaw.discovery import ReadSample
from metabaw.strategy import (
    ServerResourceProfile,
    build_auto_analyses,
    choose_auto_resources,
    choose_auto_strategy,
)


class AutoStrategyTests(unittest.TestCase):
    def _resources(
        self,
        root: Path,
        *,
        cpu_available: int = 64,
        memory_available_gib: float = 256,
        gpu_free_memory_mib: int = 0,
    ) -> ServerResourceProfile:
        return ServerResourceProfile(
            cpu_total=64,
            cpu_affinity=64,
            cpu_load_1m=float(64 - cpu_available),
            cpu_available=cpu_available,
            memory_total_gib=512,
            memory_available_gib=memory_available_gib,
            disk_free_gib=1000,
            disk_path=root,
            gpu_available=gpu_free_memory_mib > 0,
            gpu_count=1 if gpu_free_memory_mib > 0 else 0,
            gpu_total_memory_mib=49140 if gpu_free_memory_mib > 0 else 0,
            gpu_free_memory_mib=gpu_free_memory_mib,
            gpu_names=("NVIDIA RTX A6000",) if gpu_free_memory_mib > 0 else (),
            cuda_driver_version="570.133.07" if gpu_free_memory_mib > 0 else None,
            advertised_cuda_version="12.8" if gpu_free_memory_mib > 0 else None,
            cuda_error=None if gpu_free_memory_mib > 0 else "not detected",
        )

    def _sample(self, root: Path, name: str, shared: Path | None = None) -> ReadSample:
        read1 = root / f"{name}_R1.fastq.gz"
        read2 = root / f"{name}_R2.fastq.gz"
        read1.write_bytes(b"")
        read2.write_bytes(b"")
        contigs = shared or root / f"{name}.fa"
        if not contigs.exists():
            contigs.write_text(
                ">large\n" + "A" * 10_000 + "\n>small\n" + "C" * 2_000 + "\n",
                encoding="utf-8",
            )
        return ReadSample(name, read1, read2, contigs)

    def test_short_read_cohort_selects_multi_sample_ensemble(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            samples = [self._sample(root, name) for name in ("A", "B", "C")]
            strategy = choose_auto_strategy(
                samples,
                read_type="short",
                gpu=False,
                requested_mode=None,
                requested_tools=None,
                requested_min_contig_length=None,
            )
            self.assertEqual("multi", strategy.mode)
            self.assertEqual(3, strategy.group_size)
            self.assertEqual(
                ("metabat2", "metadecoder", "vamb", "semibin2"),
                strategy.tools,
            )
            self.assertEqual(1500, strategy.min_contig_length)
            self.assertEqual(3, strategy.evidence["sample_count"])

    def test_auto_keeps_explicit_strategy_values(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            samples = [self._sample(root, name) for name in ("A", "B", "C")]
            strategy = choose_auto_strategy(
                samples,
                read_type="short",
                gpu=True,
                requested_mode="single",
                requested_tools=["metabat2"],
                requested_min_contig_length=2500,
            )
            self.assertEqual("single", strategy.mode)
            self.assertEqual(("metabat2",), strategy.tools)
            self.assertEqual(2500, strategy.min_contig_length)
            self.assertEqual(
                ("mode", "tools", "min_contig_length"),
                strategy.manual_overrides,
            )

    def test_large_cohort_is_grouped_in_batches_of_twenty(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            samples = [
                self._sample(root, f"S{number:02d}")
                for number in range(1, 24)
            ]
            strategy = choose_auto_strategy(
                samples,
                read_type="short",
                gpu=False,
                requested_mode=None,
                requested_tools=["metabat2"],
                requested_min_contig_length=None,
            )
            analyses = build_auto_analyses(
                samples,
                strategy.mode,
                strategy.group_size,
            )
            self.assertEqual(20, strategy.group_size)
            self.assertEqual(23, len(analyses))
            self.assertEqual(20, len(analyses[0].samples))
            self.assertEqual(3, len(analyses[-1].samples))

    def test_shared_coassembly_is_rejected_until_supported(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            shared = root / "coassembly.fa"
            samples = [
                self._sample(root, name, shared)
                for name in ("A", "B", "C", "D")
            ]
            with self.assertRaisesRegex(
                ValueError,
                "Shared co-assembly input is not enabled",
            ):
                choose_auto_strategy(
                    samples,
                    read_type="short",
                    gpu=False,
                    requested_mode=None,
                    requested_tools=None,
                    requested_min_contig_length=None,
                )

    def test_auto_resources_use_current_free_cpu_memory_and_vram(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            profile = self._resources(
                Path(raw),
                cpu_available=48,
                memory_available_gib=192,
                gpu_free_memory_mib=12288,
            )
            plan = choose_auto_resources(profile, 3)
            self.assertEqual(3, plan.task)
            self.assertEqual(16, plan.threads)
            self.assertEqual(54, plan.max_memory_gib)
            self.assertTrue(plan.gpu)
            self.assertEqual("3481M", plan.max_gpu_memory)
            self.assertEqual(1, plan.retries)

    def test_auto_resources_keep_explicit_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            profile = self._resources(Path(raw))
            plan = choose_auto_resources(
                profile,
                5,
                requested_threads=12,
                requested_task=2,
                requested_max_memory_gib=40,
                requested_gpu=False,
                requested_max_gpu_memory="3G",
                requested_batch_size=256,
                requested_retries=4,
            )
            self.assertEqual(12, plan.threads)
            self.assertEqual(2, plan.task)
            self.assertEqual(40, plan.max_memory_gib)
            self.assertFalse(plan.gpu)
            self.assertEqual("3G", plan.max_gpu_memory)
            self.assertEqual(256, plan.batch_size)
            self.assertEqual(4, plan.retries)

    def test_auto_task_count_respects_explicit_threads(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            profile = self._resources(
                Path(raw),
                cpu_available=64,
                memory_available_gib=512,
            )
            plan = choose_auto_resources(
                profile,
                8,
                requested_threads=32,
            )
            self.assertEqual(32, plan.threads)
            self.assertEqual(2, plan.task)

    def test_auto_memory_is_not_capped_at_the_manual_default(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            profile = self._resources(
                Path(raw),
                cpu_available=48,
                memory_available_gib=480,
            )
            plan = choose_auto_resources(profile, 3)
            self.assertEqual(136, plan.max_memory_gib)
