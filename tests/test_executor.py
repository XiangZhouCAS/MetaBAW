from pathlib import Path
from contextlib import redirect_stdout
import io
import re
import sys
import tempfile
import unittest

from metabaw.executor import Executor, _failure_tail
from metabaw.model import Task
from metabaw.state import StateStore


class ExecutorTests(unittest.TestCase):
    def test_failure_tail_ignores_shell_timing_lines(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            log = Path(raw) / "task.log"
            log.write_text(
                "mkdir: cannot create directory '/work/refinement/A': "
                "No such file or directory\n"
                "cannot make /work/refinement/A\n"
                "real 0m0.194s\n"
                "user 0m0.027s\n"
                "sys 0m0.053s\n",
                encoding="utf-8",
            )
            tail = _failure_tail(log)
            self.assertIn("No such file or directory", tail)
            self.assertIn("cannot make /work/refinement/A", tail)
            self.assertNotIn("real 0m", tail)
            self.assertNotIn("user 0m", tail)
            self.assertNotIn("sys 0m", tail)

    def test_global_binning_barrier_delays_refinement_until_all_samples_finish(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            a_end = root / "A.end"
            b_end = root / "B.end"
            refinement_start = root / "refinement.start"
            bin_a = Task(
                "03.bin.example.A",
                "03_binning",
                (
                    sys.executable,
                    "-c",
                    (
                        "import time; from pathlib import Path; time.sleep(0.35); "
                        f"Path({str(a_end)!r}).write_text(str(time.time()))"
                    ),
                ),
                root,
                outputs=(a_end,),
                sample="A",
            )
            bin_b = Task(
                "03.bin.example.B",
                "03_binning",
                (
                    sys.executable,
                    "-c",
                    (
                        "import time; from pathlib import Path; time.sleep(0.05); "
                        f"Path({str(b_end)!r}).write_text(str(time.time()))"
                    ),
                ),
                root,
                outputs=(b_end,),
                sample="B",
            )
            barrier = Task(
                "03.binning.complete",
                "03_binning",
                (sys.executable, "-c", "pass"),
                root,
                deps=(bin_a.id, bin_b.id),
                allow_failed_deps=True,
            )
            refinement = Task(
                "04.refine.B",
                "04_refinement",
                (
                    sys.executable,
                    "-c",
                    (
                        "import time; from pathlib import Path; "
                        f"Path({str(refinement_start)!r}).write_text(str(time.time()))"
                    ),
                ),
                root,
                deps=(bin_b.id,),
                wait_for=(barrier.id,),
                outputs=(refinement_start,),
                sample="B",
            )
            state = StateStore(root / "state.sqlite3")
            try:
                statuses = Executor(
                    state,
                    root / "logs",
                    max_cpus=2,
                    max_parallel=2,
                ).run([bin_a, bin_b, barrier, refinement])
            finally:
                state.close()
            self.assertEqual({"success"}, set(statuses.values()))
            self.assertGreaterEqual(
                float(refinement_start.read_text()),
                float(a_end.read_text()),
            )

    def test_sample_slots_run_distinct_samples_concurrently(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            tasks = []
            for task_id, sample in (
                ("A.one", "A"),
                ("A.two", "A"),
                ("B.one", "B"),
                ("C.one", "C"),
            ):
                output = root / f"{task_id}.txt"
                tasks.append(
                    Task(
                        task_id,
                        "03_binning",
                        (
                            sys.executable,
                            "-c",
                            (
                                "import time; from pathlib import Path; "
                                "start=time.time(); time.sleep(0.3); "
                                f"Path({str(output)!r}).write_text("
                                "str(start) + ',' + str(time.time()))"
                            ),
                        ),
                        root,
                        outputs=(output,),
                        sample=sample,
                    )
                )
            state = StateStore(root / "state.sqlite3")
            try:
                statuses = Executor(
                    state,
                    root / "logs",
                    max_cpus=3,
                    max_parallel=3,
                ).run(tasks)
            finally:
                state.close()
            self.assertTrue(all(status == "success" for status in statuses.values()))

            intervals = {
                task.id: tuple(
                    float(value)
                    for value in (root / f"{task.id}.txt").read_text().split(",")
                )
                for task in tasks
            }

            def overlaps(left: tuple[float, float], right: tuple[float, float]) -> bool:
                return left[0] < right[1] and right[0] < left[1]

            self.assertFalse(overlaps(intervals["A.one"], intervals["A.two"]))
            self.assertTrue(overlaps(intervals["A.one"], intervals["B.one"]))
            self.assertTrue(overlaps(intervals["A.one"], intervals["C.one"]))

    def test_empty_binner_directory_is_validated_and_retried_once(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            bins = root / "bins"
            counter = root / "attempts.txt"
            command = (
                sys.executable,
                "-c",
                (
                    "from pathlib import Path; "
                    f"counter=Path({str(counter)!r}); bins=Path({str(bins)!r}); "
                    "attempt=int(counter.read_text()) + 1 if counter.exists() else 1; "
                    "counter.write_text(str(attempt)); bins.mkdir(parents=True, exist_ok=True); "
                    "(bins / 'bin.1.fa').write_text('>contig\\nACGT\\n') "
                    "if attempt >= 2 else None"
                ),
            )
            task = Task(
                "03.bin.example.S",
                "03_binning",
                command,
                root,
                outputs=(bins,),
                fasta_output_dirs=(bins,),
                automatic_retries=1,
                sample="S",
            )
            state = StateStore(root / "state.sqlite3")
            terminal = io.StringIO()
            try:
                executor = Executor(
                    state,
                    root / "logs",
                    max_cpus=1,
                    max_parallel=1,
                    show_progress=True,
                )
                with redirect_stdout(terminal):
                    statuses = executor.run([task])
            finally:
                state.close()
            self.assertEqual("success", statuses[task.id])
            self.assertEqual("2", counter.read_text())
            self.assertIn("[RETRY 1/1]", terminal.getvalue())
            self.assertIn("no FASTA bins were generated", terminal.getvalue())

    def test_running_heartbeat_is_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            tasks = []
            for name in ("one", "two"):
                output = root / f"{name}.txt"
                tasks.append(
                    Task(
                        name,
                        "01_test",
                        (
                            sys.executable,
                            "-c",
                            (
                                "import time; from pathlib import Path; "
                                "time.sleep(0.1); "
                                f"Path({str(output)!r}).write_text('ok')"
                            ),
                        ),
                        root,
                        outputs=(output,),
                    )
                )
            state = StateStore(root / "state.sqlite3")
            terminal = io.StringIO()
            try:
                executor = Executor(
                    state,
                    root / "logs",
                    max_cpus=2,
                    max_parallel=2,
                    show_progress=True,
                )
                with redirect_stdout(terminal):
                    statuses = executor.run(tasks)
            finally:
                state.close()
            self.assertEqual({"one": "success", "two": "success"}, statuses)
            heartbeats = [
                line for line in terminal.getvalue().splitlines()
                if "[RUNNING]" in line
            ]
            self.assertEqual([], heartbeats)

    def test_gpu_tasks_are_serialized_when_only_one_gpu_is_visible(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            tasks = []
            for name in ("one", "two"):
                output = root / f"{name}.txt"
                tasks.append(
                    Task(
                        name,
                        "01_test",
                        (
                            sys.executable,
                            "-c",
                            (
                                "import time; from pathlib import Path; "
                                "started=time.time(); time.sleep(0.25); "
                                f"Path({str(output)!r}).write_text("
                                "str(started) + ',' + str(time.time()))"
                            ),
                        ),
                        root,
                        outputs=(output,),
                        gpus=1,
                    )
                )
            state = StateStore(root / "state.sqlite3")
            try:
                statuses = Executor(
                    state,
                    root / "logs",
                    max_cpus=2,
                    max_parallel=2,
                    max_gpus=1,
                ).run(tasks)
            finally:
                state.close()
            self.assertEqual({"one": "success", "two": "success"}, statuses)
            one_start, one_end = (
                float(value)
                for value in (root / "one.txt").read_text().split(",")
            )
            two_start, two_end = (
                float(value)
                for value in (root / "two.txt").read_text().split(",")
            )
            self.assertTrue(one_end <= two_start or two_end <= one_start)

    def test_progress_reports_time_step_task_and_completion(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            output = root / "output.txt"
            task = Task(
                "write",
                "01_test",
                (
                    sys.executable,
                    "-c",
                    "from pathlib import Path; Path('output.txt').write_text('ok')",
                ),
                root,
                outputs=(output,),
                description="Write the test output",
            )
            state = StateStore(root / "state.sqlite3")
            terminal = io.StringIO()
            try:
                executor = Executor(
                    state,
                    root / "logs",
                    1,
                    1,
                    max_memory_gb=100,
                    show_progress=True,
                )
                with redirect_stdout(terminal):
                    self.assertEqual(executor.run([task])["write"], "success")
            finally:
                state.close()
            progress = terminal.getvalue()
            self.assertRegex(progress, re.compile(r"\[\d{4}-\d{2}-\d{2}T"))
            self.assertIn("[START 1/1] write", progress)
            self.assertIn("Write the test output", progress)
            self.assertIn("[DONE 1/1] write", progress)
            self.assertNotIn("max_memory=", progress)
            self.assertNotIn("write.log", progress.split("[START", 1)[1].split("[DONE", 1)[0])
            log = (root / "logs" / "write.log").read_text(encoding="utf-8")
            self.assertIn("# started:", log)
            self.assertIn("# finished:", log)
            self.assertIn("# exit_code: 0", log)

    def test_failure_reports_exit_code_last_output_and_log_path(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            task = Task(
                "fail",
                "01_test",
                (
                    sys.executable,
                    "-c",
                    "import sys; print('specific failure detail'); sys.exit(7)",
                ),
                root,
                description="Fail intentionally",
            )
            state = StateStore(root / "state.sqlite3")
            terminal = io.StringIO()
            try:
                executor = Executor(
                    state,
                    root / "logs",
                    1,
                    1,
                    show_progress=True,
                )
                with redirect_stdout(terminal):
                    self.assertEqual(executor.run([task])["fail"], "failed")
            finally:
                state.close()
            progress = terminal.getvalue()
            self.assertIn("[FAIL 1/1] fail", progress)
            self.assertIn("command exited with code 7", progress)
            self.assertIn("specific failure detail", progress)
            self.assertIn(str(root / "logs" / "fail.log"), progress)

    def test_successful_task_is_cached(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            output = root / "output.txt"
            task = Task(
                "write",
                "test",
                (sys.executable, "-c", "from pathlib import Path; Path('output.txt').write_text('ok')"),
                root,
                outputs=(output,),
            )
            state = StateStore(root / "state.sqlite3")
            try:
                executor = Executor(state, root / "logs", 1, 1)
                self.assertEqual(executor.run([task])["write"], "success")
                self.assertEqual(executor.run([task])["write"], "skipped")
            finally:
                state.close()

    def test_startup_scan_adopts_complete_work_outputs_without_state(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            prepared = root / "work" / "contigs" / "S.fna"
            bam = root / "work" / "mapping" / "S" / "S.bam"
            bai = Path(str(bam) + ".bai")
            for path in (prepared, bam, bai):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("complete", encoding="utf-8")
            prepare_task = Task(
                id="01.prepare.S",
                stage="01_prepare",
                command=(sys.executable, "-c", "raise SystemExit(91)"),
                cwd=root,
                outputs=(prepared,),
            )
            mapping_task = Task(
                id="02.map.S.S",
                stage="02_mapping",
                command=(sys.executable, "-c", "raise SystemExit(92)"),
                cwd=root,
                deps=(prepare_task.id,),
                inputs=(prepared,),
                outputs=(bam, bai),
            )
            state = StateStore(root / "state.sqlite3")
            terminal = io.StringIO()
            try:
                with redirect_stdout(terminal):
                    statuses = Executor(
                        state,
                        root / "logs",
                        1,
                        1,
                        show_progress=True,
                    ).run([mapping_task, prepare_task])
                adopted = state.get(mapping_task.id)
            finally:
                state.close()

            self.assertEqual("skipped", statuses[prepare_task.id])
            self.assertEqual("skipped", statuses[mapping_task.id])
            self.assertIsNotNone(adopted)
            self.assertEqual("success", adopted.status)
            self.assertIn("Adopted complete outputs", adopted.message)
            progress = terminal.getvalue()
            self.assertEqual(1, progress.count("[RESUME]"))
            self.assertIn("Reusing 2/2 complete tasks", progress)
            self.assertIn("adopted=2", progress)

    def test_complete_output_is_reused_after_interrupted_or_changed_state(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            output = root / "quality_report.tsv"
            marker = root / "unexpected_rerun.txt"
            output.write_text("Name\tCompleteness\n", encoding="utf-8")
            task = Task(
                id="05.qc.checkm2",
                stage="05_quality",
                command=(
                    sys.executable,
                    "-c",
                    f"from pathlib import Path; Path({str(marker)!r}).write_text('rerun')",
                ),
                cwd=root,
                outputs=(output,),
            )
            state = StateStore(root / "state.sqlite3")
            terminal = io.StringIO()
            try:
                state.start(task.id, "obsolete-fingerprint", root / "old.log")
                with redirect_stdout(terminal):
                    status = Executor(
                        state,
                        root / "logs",
                        1,
                        1,
                        show_progress=True,
                    ).run([task])[task.id]
                record = state.get(task.id)
            finally:
                state.close()

            self.assertEqual("skipped", status)
            self.assertFalse(marker.exists())
            self.assertIsNotNone(record)
            self.assertEqual("success", record.status)
            self.assertEqual(task.fingerprint([]), record.fingerprint)
            self.assertIn("Reusing 1/1 complete tasks", terminal.getvalue())
            self.assertIn("adopted=1", terminal.getvalue())

    def test_force_reruns_a_task_with_complete_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            output = root / "output.txt"
            output.write_text("old", encoding="utf-8")
            task = Task(
                id="rewrite",
                stage="01_prepare",
                command=(
                    sys.executable,
                    "-c",
                    "from pathlib import Path; Path('output.txt').write_text('new')",
                ),
                cwd=root,
                outputs=(output,),
            )
            state = StateStore(root / "state.sqlite3")
            try:
                status = Executor(state, root / "logs", 1, 1).run(
                    [task],
                    force=True,
                )[task.id]
            finally:
                state.close()

            self.assertEqual("success", status)
            self.assertEqual("new", output.read_text(encoding="utf-8"))

    def test_startup_scan_runs_only_incomplete_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            prepared = root / "work" / "contigs" / "S.fna"
            bam = root / "work" / "mapping" / "S" / "S.bam"
            bai = Path(str(bam) + ".bai")
            bins = root / "work" / "binning" / "S" / "bins"
            count = root / "mapping_runs.txt"
            prepared.parent.mkdir(parents=True, exist_ok=True)
            prepared.write_text(">c\nA\n", encoding="utf-8")
            bam.parent.mkdir(parents=True, exist_ok=True)
            bam.write_text("partial", encoding="utf-8")
            bins.mkdir(parents=True)
            (bins / "S_MetaBAT2_1.fa").write_text(">c\nA\n", encoding="utf-8")
            prepare_task = Task(
                id="01.prepare.S",
                stage="01_prepare",
                command=(sys.executable, "-c", "raise SystemExit(93)"),
                cwd=root,
                outputs=(prepared,),
            )
            mapping_task = Task(
                id="02.map.S.S",
                stage="02_mapping",
                command=(
                    sys.executable,
                    "-c",
                    (
                        "from pathlib import Path; "
                        f"bam=Path({str(bam)!r}); bai=Path({str(bai)!r}); "
                        f"count=Path({str(count)!r}); "
                        "bam.write_text('complete'); bai.write_text('complete'); "
                        "count.write_text('run\\n')"
                    ),
                ),
                cwd=root,
                deps=(prepare_task.id,),
                inputs=(prepared,),
                outputs=(bam, bai),
            )
            bin_task = Task(
                id="03.bin.metabat2.S",
                stage="03_binning",
                command=(sys.executable, "-c", "raise SystemExit(94)"),
                cwd=root,
                deps=(mapping_task.id,),
                inputs=(bam,),
                outputs=(bins,),
            )
            state = StateStore(root / "state.sqlite3")
            terminal = io.StringIO()
            try:
                with redirect_stdout(terminal):
                    statuses = Executor(
                        state,
                        root / "logs",
                        1,
                        1,
                        show_progress=True,
                    ).run([bin_task, mapping_task, prepare_task])
            finally:
                state.close()

            self.assertEqual("skipped", statuses[prepare_task.id])
            self.assertEqual("success", statuses[mapping_task.id])
            self.assertEqual("skipped", statuses[bin_task.id])
            self.assertEqual("run\n", count.read_text(encoding="utf-8"))
            progress = terminal.getvalue()
            self.assertEqual(1, progress.count("[RESUME]"))
            self.assertIn("Reusing 2/3 complete tasks", progress)
            self.assertNotIn("[CHECK]", progress)
            self.assertLess(
                progress.index("[RESUME]"),
                progress.index("[START 2/3]"),
            )

    def test_empty_output_directory_is_not_complete(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            bins = root / "work" / "binning" / "S" / "bins"
            bins.mkdir(parents=True)
            task = Task(
                id="03.bin.metabat2.S",
                stage="03_binning",
                command=(
                    sys.executable,
                    "-c",
                    (
                        "from pathlib import Path; "
                        f"output=Path({str(bins / 'bin.1.fa')!r}); "
                        "output.write_text('>c\\nA\\n')"
                    ),
                ),
                cwd=root,
                outputs=(bins,),
            )
            state = StateStore(root / "state.sqlite3")
            terminal = io.StringIO()
            try:
                with redirect_stdout(terminal):
                    status = Executor(
                        state,
                        root / "logs",
                        1,
                        1,
                        show_progress=True,
                    ).run([task])[task.id]
            finally:
                state.close()

            self.assertEqual("success", status)
            progress = terminal.getvalue()
            self.assertIn("[RESUME] Reusing 0/1 complete tasks", progress)
            self.assertIn("[START 1/1] 03.bin.metabat2.S", progress)

    def test_failed_checkm2_run_resumes_from_successful_candidate_mags(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            refined = root / "refined" / "S_MAGScoT_1.fa"
            candidates = root / "candidate_bins" / "S_MAGScoT_1.fa"
            report = root / "checkm2" / "quality_report.tsv"
            filtered = root / "filtered.tsv"
            refined_count = root / "refined_runs.txt"
            checkm2_count = root / "checkm2_runs.txt"

            refined_task = Task(
                id="04.bins.S",
                stage="04_refinement",
                command=(
                    sys.executable,
                    "-c",
                    (
                        "from pathlib import Path; "
                        f"output=Path({str(refined)!r}); count=Path({str(refined_count)!r}); "
                        "output.parent.mkdir(parents=True, exist_ok=True); "
                        "output.write_text('>c\\nA\\n'); "
                        "count.write_text(count.read_text() + 'run\\n' "
                        "if count.exists() else 'run\\n')"
                    ),
                ),
                cwd=root,
                outputs=(refined,),
            )
            catalog_task = Task(
                id="05.catalog.candidates",
                stage="05_quality",
                command=(
                    sys.executable,
                    "-c",
                    (
                        "from pathlib import Path; "
                        f"source=Path({str(refined)!r}); output=Path({str(candidates)!r}); "
                        "output.parent.mkdir(parents=True, exist_ok=True); "
                        "output.write_text(source.read_text())"
                    ),
                ),
                cwd=root,
                deps=(refined_task.id,),
                inputs=(refined,),
                outputs=(candidates,),
            )
            checkm2_task = Task(
                id="05.qc.checkm2",
                stage="05_quality",
                command=(
                    sys.executable,
                    "-c",
                    (
                        "from pathlib import Path; "
                        f"count=Path({str(checkm2_count)!r}); report=Path({str(report)!r}); "
                        "runs=len(count.read_text().splitlines()) if count.exists() else 0; "
                        "count.write_text(count.read_text() + 'run\\n' "
                        "if count.exists() else 'run\\n'); "
                        "report.parent.mkdir(parents=True, exist_ok=True); "
                        "report.write_text('Name\\tCompleteness\\n') if runs else None; "
                        "raise SystemExit(0 if runs else 11)"
                    ),
                ),
                cwd=root,
                deps=(catalog_task.id,),
                inputs=(candidates,),
                outputs=(report,),
            )
            filter_task = Task(
                id="05.qc.filter",
                stage="05_quality",
                command=(
                    sys.executable,
                    "-c",
                    f"from pathlib import Path; Path({str(filtered)!r}).write_text('ok')",
                ),
                cwd=root,
                deps=(checkm2_task.id,),
                inputs=(report,),
                outputs=(filtered,),
            )
            tasks = [refined_task, catalog_task, checkm2_task, filter_task]
            state = StateStore(root / "state.sqlite3")
            first_terminal = io.StringIO()
            second_terminal = io.StringIO()
            try:
                executor = Executor(
                    state,
                    root / "logs",
                    1,
                    1,
                    fail_fast=False,
                    show_progress=True,
                )
                with redirect_stdout(first_terminal):
                    first = executor.run(tasks)
                with redirect_stdout(second_terminal):
                    second = executor.run(tasks)
            finally:
                state.close()

            self.assertEqual("success", first[refined_task.id])
            self.assertEqual("failed", first[checkm2_task.id])
            self.assertEqual("blocked", first[filter_task.id])
            self.assertEqual("skipped", second[refined_task.id])
            self.assertEqual("skipped", second[catalog_task.id])
            self.assertEqual("success", second[checkm2_task.id])
            self.assertEqual("success", second[filter_task.id])
            self.assertEqual(["run"], refined_count.read_text(encoding="utf-8").splitlines())
            self.assertEqual(
                ["run", "run"],
                checkm2_count.read_text(encoding="utf-8").splitlines(),
            )
            blocked = first_terminal.getvalue()
            self.assertEqual(1, blocked.count("[BLOCKED]"))
            self.assertIn("1 downstream task(s) were not run", blocked)
            resumed = second_terminal.getvalue()
            self.assertIn("[RESUME] Reusing 2/4 complete tasks", resumed)

    def test_legacy_state_reuses_existing_refined_mags_after_upgrade(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            binned = root / "bins" / "S_MetaBAT2_1.fa"
            refined = root / "refined" / "S_MAGScoT_1.fa"
            candidates = root / "candidate_bins" / "S_MAGScoT_1.fa"
            report = root / "checkm2" / "quality_report.tsv"
            for path in (binned, refined, candidates):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(">c\nA\n", encoding="utf-8")

            bin_task = Task(
                id="03.publish.metabat2.S",
                stage="03_binning",
                command=(sys.executable, "-c", "raise SystemExit(90)"),
                cwd=root,
                outputs=(binned,),
            )
            refined_task = Task(
                id="04.bins.S",
                stage="04_refinement",
                command=(sys.executable, "-c", "raise SystemExit(91)"),
                cwd=root,
                deps=(bin_task.id,),
                inputs=(binned,),
                outputs=(refined,),
            )
            catalog_task = Task(
                id="05.catalog.candidates",
                stage="05_quality",
                command=(sys.executable, "-c", "raise SystemExit(92)"),
                cwd=root,
                deps=(refined_task.id,),
                inputs=(refined,),
                outputs=(candidates,),
            )
            checkm2_task = Task(
                id="05.qc.checkm2",
                stage="05_quality",
                command=(
                    sys.executable,
                    "-c",
                    (
                        "from pathlib import Path; "
                        f"output=Path({str(report)!r}); "
                        "output.parent.mkdir(parents=True, exist_ok=True); "
                        "output.write_text('Name\\tCompleteness\\n')"
                    ),
                ),
                cwd=root,
                deps=(catalog_task.id,),
                inputs=(candidates,),
                outputs=(report,),
            )
            state = StateStore(root / "state.sqlite3")
            terminal = io.StringIO()
            try:
                bin_fingerprint = bin_task.fingerprint([])
                refined_legacy = refined_task.fingerprint(
                    [bin_fingerprint],
                    assume_missing_inputs=frozenset({binned}),
                )
                catalog_legacy = catalog_task.fingerprint(
                    [refined_legacy],
                    assume_missing_inputs=frozenset({refined}),
                )
                for task, fingerprint in (
                    (bin_task, bin_fingerprint),
                    (refined_task, refined_legacy),
                    (catalog_task, catalog_legacy),
                ):
                    state.start(task.id, fingerprint, root / "legacy.log")
                    state.finish(task.id, "success", 0, "legacy success")

                with redirect_stdout(terminal):
                    statuses = Executor(
                        state,
                        root / "logs",
                        1,
                        1,
                        fail_fast=False,
                        show_progress=True,
                    ).run([bin_task, refined_task, catalog_task, checkm2_task])
            finally:
                state.close()

            self.assertEqual("skipped", statuses[bin_task.id])
            self.assertEqual("skipped", statuses[refined_task.id])
            self.assertEqual("skipped", statuses[catalog_task.id])
            self.assertEqual("success", statuses[checkm2_task.id])
            progress = terminal.getvalue()
            self.assertEqual(1, progress.count("[RESUME]"))
            self.assertIn("Reusing 3/4 complete tasks", progress)
            self.assertIn("[START 4/4] 05.qc.checkm2", progress)

    def test_memory_limit_is_forwarded_to_tasks_and_logs(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            output = root / "memory.txt"
            task = Task(
                "memory",
                "test",
                (
                    sys.executable,
                    "-c",
                    "import os; from pathlib import Path; "
                    "Path('memory.txt').write_text(os.environ['METABAW_MAX_MEMORY_GB'])",
                ),
                root,
                outputs=(output,),
            )
            state = StateStore(root / "state.sqlite3")
            try:
                executor = Executor(
                    state,
                    root / "logs",
                    1,
                    1,
                    max_memory_gb=100,
                )
                self.assertEqual(executor.run([task])["memory"], "success")
            finally:
                state.close()
            self.assertEqual("100", output.read_text(encoding="utf-8"))
            self.assertIn(
                "# max_memory_gb: 100",
                (root / "logs" / "memory.log").read_text(encoding="utf-8"),
            )

    def test_temp_environment_is_forwarded_and_logged(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            output = root / "temp_environment.txt"
            temp_dir = root / "visible_tmp" / "system_tmp"
            command = (
                sys.executable,
                "-c",
                (
                    "import os; from pathlib import Path; "
                    "Path('temp_environment.txt').write_text("
                    "'\\n'.join(os.environ[name] for name in ('TMPDIR', 'TMP', 'TEMP')))"
                ),
            )
            task = Task(
                id="temp_environment",
                stage="test",
                command=command,
                cwd=root,
                outputs=(output,),
            )
            state = StateStore(root / "state.sqlite3")
            try:
                statuses = Executor(
                    state,
                    root / "logs",
                    1,
                    1,
                    temp_dir=temp_dir,
                ).run([task])
            finally:
                state.close()
            self.assertEqual("success", statuses["temp_environment"])
            self.assertEqual([str(temp_dir)] * 3, output.read_text(encoding="utf-8").splitlines())
            log = (root / "logs" / "temp_environment.log").read_text(encoding="utf-8")
            self.assertIn(f"# temp_dir: {temp_dir}", log)

    def test_missing_alternative_output_invalidates_cached_index_task(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            marker = root / "bowtie2.index.done"
            index_file = root / "index" / "contigs.1.bt2"
            count = root / "runs.txt"
            script = (
                "from pathlib import Path; "
                f"marker=Path({str(marker)!r}); index=Path({str(index_file)!r}); "
                f"count=Path({str(count)!r}); "
                "index.parent.mkdir(parents=True, exist_ok=True); "
                "marker.write_text('ready'); index.write_text('index'); "
                "count.write_text(count.read_text() + 'run\\n' if count.exists() else 'run\\n')"
            )
            task = Task(
                id="bowtie2_index",
                stage="mapping",
                command=(sys.executable, "-c", script),
                cwd=root,
                outputs=(marker,),
                output_alternatives=((index_file,),),
            )
            state = StateStore(root / "state.sqlite3")
            terminal = io.StringIO()
            try:
                executor = Executor(state, root / "logs", 1, 1)
                self.assertEqual("success", executor.run([task])["bowtie2_index"])
                index_file.unlink()
                executor.show_progress = True
                with redirect_stdout(terminal):
                    self.assertEqual("success", executor.run([task])["bowtie2_index"])
            finally:
                state.close()
            self.assertEqual(["run", "run"], count.read_text(encoding="utf-8").splitlines())
            progress = terminal.getvalue()
            self.assertNotIn("[RERUN]", progress)
            self.assertIn("[START 1/1] bowtie2_index", progress)

    def test_rebuilt_index_does_not_invalidate_existing_bam(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            contigs = root / "contigs.fa"
            index = root / "index" / "contigs.1.bt2"
            bam = root / "mapping" / "sample.bam"
            index_count = root / "index_runs.txt"
            mapping_count = root / "mapping_runs.txt"
            contigs.write_text(">c\nA\n", encoding="utf-8")

            def counted_output_script(output: Path, count: Path) -> str:
                return (
                    "from pathlib import Path; "
                    f"output=Path({str(output)!r}); count=Path({str(count)!r}); "
                    "output.parent.mkdir(parents=True, exist_ok=True); "
                    "output.write_text('ok'); "
                    "count.write_text(count.read_text() + 'run\\n' "
                    "if count.exists() else 'run\\n')"
                )

            index_task = Task(
                id="02.index.S",
                stage="02_mapping",
                command=(
                    sys.executable,
                    "-c",
                    counted_output_script(index, index_count),
                ),
                cwd=root,
                inputs=(contigs,),
                outputs=(index,),
            )
            mapping_task = Task(
                id="02.map.S.S",
                stage="02_mapping",
                command=(
                    sys.executable,
                    "-c",
                    counted_output_script(bam, mapping_count),
                ),
                cwd=root,
                deps=(index_task.id,),
                inputs=(contigs,),
                outputs=(bam,),
            )
            state = StateStore(root / "state.sqlite3")
            terminal = io.StringIO()
            try:
                executor = Executor(
                    state,
                    root / "logs",
                    1,
                    1,
                    fail_fast=False,
                    show_progress=True,
                )
                with redirect_stdout(io.StringIO()):
                    first = executor.run([index_task, mapping_task])
                index.unlink()
                with redirect_stdout(terminal):
                    second = executor.run([index_task, mapping_task])
            finally:
                state.close()

            self.assertEqual("success", first[index_task.id])
            self.assertEqual("success", first[mapping_task.id])
            self.assertEqual("success", second[index_task.id])
            self.assertEqual("skipped", second[mapping_task.id])
            self.assertEqual(
                ["run", "run"],
                index_count.read_text(encoding="utf-8").splitlines(),
            )
            self.assertEqual(
                ["run"],
                mapping_count.read_text(encoding="utf-8").splitlines(),
            )
            progress = terminal.getvalue()
            self.assertLess(
                progress.index("[RESUME]"),
                progress.index("[START 1/2]"),
            )

    def test_cache_events_do_not_jump_to_late_tasks_before_early_start(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            prepared = root / "prepared.fa"
            index = root / "index.bt2"
            independent = root / "independent.tsv"

            def writer(path: Path) -> tuple[str, ...]:
                return (
                    sys.executable,
                    "-c",
                    (
                        "from pathlib import Path; "
                        f"output=Path({str(path)!r}); "
                        "output.parent.mkdir(parents=True, exist_ok=True); "
                        "output.write_text('ok')"
                    ),
                )

            prepare_task = Task(
                id="01.prepare.S",
                stage="01_prepare",
                command=writer(prepared),
                cwd=root,
                outputs=(prepared,),
            )
            index_task = Task(
                id="02.index.S",
                stage="02_mapping",
                command=writer(index),
                cwd=root,
                deps=(prepare_task.id,),
                inputs=(prepared,),
                outputs=(index,),
            )
            independent_task = Task(
                id="04.markers.S",
                stage="04_refinement",
                command=writer(independent),
                cwd=root,
                deps=(prepare_task.id,),
                inputs=(prepared,),
                outputs=(independent,),
            )
            tasks = [prepare_task, index_task, independent_task]
            state = StateStore(root / "state.sqlite3")
            terminal = io.StringIO()
            try:
                executor = Executor(
                    state,
                    root / "logs",
                    1,
                    1,
                    fail_fast=False,
                    show_progress=True,
                )
                with redirect_stdout(io.StringIO()):
                    executor.run(tasks)
                index.unlink()
                with redirect_stdout(terminal):
                    statuses = executor.run(tasks)
            finally:
                state.close()

            self.assertEqual("skipped", statuses[prepare_task.id])
            self.assertEqual("success", statuses[index_task.id])
            self.assertEqual("skipped", statuses[independent_task.id])
            progress = terminal.getvalue()
            self.assertLess(
                progress.index("[RESUME]"),
                progress.index("[START 2/3]"),
            )
            self.assertNotIn("[CHECK]", progress)

    def test_tolerated_binner_failure_does_not_stop_later_priority_or_merge(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            later_output = root / "later.tsv"
            merged_output = root / "merged.tsv"
            failed = Task(
                id="03.bin.metabat2.S",
                stage="03_binning",
                command=(sys.executable, "-c", "raise SystemExit(7)"),
                cwd=root,
                priority=10,
                failure_tolerated=True,
            )
            later = Task(
                id="03.bin.metadecoder.S",
                stage="03_binning",
                command=(
                    sys.executable,
                    "-c",
                    "from pathlib import Path; Path('later.tsv').write_text('ok')",
                ),
                cwd=root,
                outputs=(later_output,),
                priority=20,
                failure_tolerated=True,
            )
            merge = Task(
                id="04.combine.S",
                stage="04_refinement",
                command=(
                    sys.executable,
                    "-c",
                    "from pathlib import Path; Path('merged.tsv').write_text('ok')",
                ),
                cwd=root,
                deps=(failed.id, later.id),
                outputs=(merged_output,),
                allow_failed_deps=True,
            )
            progress_output = io.StringIO()
            state = StateStore(root / "state.sqlite3")
            try:
                with redirect_stdout(progress_output):
                    statuses = Executor(
                        state,
                        root / "logs",
                        1,
                        1,
                        fail_fast=False,
                        show_progress=True,
                    ).run([later, merge, failed])
            finally:
                state.close()
            self.assertEqual("failed", statuses[failed.id])
            self.assertEqual("success", statuses[later.id])
            self.assertEqual("success", statuses[merge.id])
            progress = progress_output.getvalue()
            self.assertLess(progress.index(failed.id), progress.index(later.id))


if __name__ == "__main__":
    unittest.main()

