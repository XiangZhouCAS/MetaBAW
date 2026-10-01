from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import replace
import io
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from metabaw.cli import build_parser, command_annotation
from metabaw.direct import AnnotationBuilder, AnnotationOptions
from metabaw.internal import _filesystem_type, _gtdbtk_temp_environment, run_gtdbtk_classify


class GtdbtkResourceTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.options = AnnotationOptions(
            mag_dir=self.root / "mags", mags=(self.root / "mags" / "A.fa",), mag_suffix="fa",
            reads=(), output=self.root / "out", output_suffix=".tsv", threads=128,
            total_threads=128, read_type="short", methods=("mean",), place_species=True,
            niche_rank="family", niche_method="cv", no_niche=True, gtdbtk_data=None,
            run_kegg=False, run_cazy=False, run_hydrogenase=False, kegg_db=None,
            dbcan_db=None, hydrogenase_db=None)

    def test_default_cap_override_and_total_budget_clamping(self):
        for total, cap, expected in ((128, 16, 16), (8, 16, 8), (128, 32, 32), (32, 128, 32)):
            options = replace(self.options, threads=total, total_threads=total, gtdbtk_threads=cap)
            tasks = AnnotationBuilder(options, self.root / "work").build()
            task = next(task for task in tasks if task.id == "01.annotation.gtdbtk")
            self.assertEqual(task.cpus, expected)
            self.assertEqual(task.command[task.command.index("--threads") + 1], str(expected))
            self.assertEqual(task.command[task.command.index("--pplacer-threads") + 1], "1")
            self.assertIn("--ipc-tmpdir", task.command)
            self.assertFalse(task.enforce_memory_limit)  # PSS guard, not virtual-address-space limit.

    def test_annotation_threads_for_other_steps_are_unchanged(self):
        options = replace(self.options, run_kegg=True, kegg_db=self.root / "kegg")
        tasks = AnnotationBuilder(options, self.root / "work").build()
        by_id = {task.id: task for task in tasks}
        self.assertEqual(by_id["01.annotation.gtdbtk"].cpus, 16)
        self.assertEqual(by_id["06.annotation.kegg"].cpus, 128)

    def test_new_options_are_in_annotation_help(self):
        for flag in ("-h", "--help"):
            output = io.StringIO()
            with redirect_stdout(output), self.assertRaises(SystemExit):
                build_parser().parse_args(["annotation", flag])
            for option in ("--gtdbtk-threads", "--gtdbtk-tmpdir"):
                self.assertIn(option, output.getvalue())

    def test_cli_passes_cap_and_records_effective_parameters(self):
        (self.root / "A.fa").write_text(">c\nACGT\n")
        (self.root / "r.fastq").touch()
        (self.root / "genomes.txt").write_text("A.fa\n")
        (self.root / "reads.tsv").write_text("S1\tr.fastq\n")
        args = build_parser().parse_args([
            "annotation", "--input_genome_files", str(self.root / "genomes.txt"),
            "--input_reads_files", str(self.root / "reads.tsv"), "-t", "128",
            "--gtdbtk-threads", "8", "--gtdbtk-tmpdir", str(self.root / "local"),
            "-o", str(self.root / "out")])
        with patch("metabaw.cli._preflight_annotation"), patch("metabaw.cli._run_direct", return_value=0) as run, redirect_stdout(io.StringIO()):
            self.assertEqual(command_annotation(args), 0)
        task = next(task for task in run.call_args.args[1] if task.id == "01.annotation.gtdbtk")
        self.assertEqual(task.cpus, 8)
        params = run.call_args.args[4]["gtdbtk_parameters"]
        self.assertEqual(params["cpus"], 8)
        self.assertEqual(params["requested_thread_cap"], 8)
        self.assertEqual(Path(params["subprocess_tmp_parent"]), (self.root / "local").resolve())
        args.gtdbtk_threads = 0
        with self.assertRaisesRegex(ValueError, "--gtdbtk-threads must be positive"):
            command_annotation(args)

    def test_private_tmpdir_is_real_unique_and_ignores_inherited_nfs_tmpdir(self):
        parent = self.root / "local"
        with (patch("metabaw.internal.sys.platform", "linux"),
              patch("metabaw.internal._filesystem_type", return_value="ext4"),
              patch("metabaw.internal._multiprocessing_socket_path_too_long", return_value=False),
              patch.dict("os.environ", {"TMPDIR": "/nfs/workflow/tmp"}), redirect_stdout(io.StringIO())):
            env1, tmp1 = _gtdbtk_temp_environment(parent)
            env2, tmp2 = _gtdbtk_temp_environment(parent)
        self.assertNotEqual(tmp1, tmp2)
        for environment, path in ((env1, tmp1), (env2, tmp2)):
            self.assertTrue(path.is_dir())
            self.assertFalse(path.is_symlink())
            self.assertEqual(path.parent, parent.resolve())
            self.assertEqual(environment["TMPDIR"], str(path))
            self.assertEqual(environment["TMP"], str(path))
            self.assertEqual(environment["TEMP"], str(path))

    def test_nfs_parent_is_rejected_instead_of_aliased(self):
        parent = self.root / "remote"
        with patch("metabaw.internal.sys.platform", "linux"), patch("metabaw.internal._filesystem_type", return_value="nfs4"):
            with self.assertRaisesRegex(ValueError, "is on nfs4"):
                _gtdbtk_temp_environment(parent)
        self.assertEqual(list(parent.iterdir()), [])

    def test_long_socket_parent_fails_without_allocating_a_directory(self):
        parent = self.root / "long"
        with (patch("metabaw.internal.sys.platform", "linux"),
              patch("metabaw.internal._filesystem_type", return_value="ext4"),
              patch("metabaw.internal._multiprocessing_socket_path_too_long", return_value=True)):
            with self.assertRaisesRegex(ValueError, "too long"):
                _gtdbtk_temp_environment(parent)
        self.assertEqual(list(parent.iterdir()), [])

    def test_mount_detection_selects_the_deepest_mount(self):
        root = self.root.as_posix()
        escaped_root = root.replace(" ", r"\040")
        remote = (self.root / "mounted" / "with spaces").as_posix().replace(" ", r"\040")
        mounts = f"1 0 8:1 / {escaped_root} rw - ext4 /dev/root rw\n2 1 0:1 / {remote} rw - nfs4 server:/share rw\n"
        with patch.object(Path, "read_text", return_value=mounts):
            self.assertEqual(_filesystem_type(self.root / "mounted" / "with spaces" / "job"), "nfs4")
            self.assertEqual(_filesystem_type(self.root / "local"), "ext4")

    def test_wrapper_cleans_only_its_private_directory_after_tool_failure(self):
        parent = self.root / "local"
        private = parent / "mbw-gtdb-owned"
        private.mkdir(parents=True)
        (private / "ipc").touch()
        sibling = parent / "keep.txt"
        sibling.write_text("user data")
        environment = {"TMPDIR": str(private), "TMP": str(private), "TEMP": str(private)}
        probe = subprocess.CompletedProcess([], 0, "--pplacer_cpus --scratch_dir --place_species", "")
        failed = subprocess.CompletedProcess([], 1, "", "failed")
        with (patch("metabaw.internal.shutil.which", return_value="gtdbtk"),
              patch("metabaw.internal._gtdbtk_temp_environment", return_value=(environment, private)),
              patch("metabaw.internal.subprocess.run", side_effect=[probe, failed])):
            with self.assertRaisesRegex(RuntimeError, "exited with code 1"):
                run_gtdbtk_classify(self.root, self.root / "taxonomy", "fa", 16, True)
        self.assertFalse(private.exists())
        self.assertEqual(sibling.read_text(), "user data")


if __name__ == "__main__":
    unittest.main()
