from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import io
import unittest
from unittest.mock import Mock, patch

from metabaw.cli import main


class CliBannerTests(unittest.TestCase):
    def _run_main(self, return_code: int) -> tuple[int, list[str], str]:
        parsed = argparse.Namespace(
            command="binning",
            func=lambda _args: return_code,
        )
        parser = Mock()
        parser.parse_args.return_value = parsed
        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch("metabaw.cli.build_parser", return_value=parser):
            with redirect_stdout(stdout), redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as stopped:
                    main(["binning", "-p", "/data/reads with spaces"])
        return int(stopped.exception.code), stdout.getvalue().splitlines(), stderr.getvalue()

    def test_success_prints_banner_command_and_final_marker(self) -> None:
        code, lines, stderr = self._run_main(0)

        self.assertEqual(code, 0)
        self.assertEqual(lines[0], "MetaBAW is Running.")
        self.assertEqual(
            lines[1],
            "Command: metabaw binning -p '/data/reads with spaces'",
        )
        self.assertEqual(lines[-1], "ALL DONE.")
        self.assertEqual(stderr, "")

    def test_failed_workflow_does_not_print_final_marker(self) -> None:
        code, lines, _stderr = self._run_main(1)

        self.assertEqual(code, 1)
        self.assertEqual(lines[:2], [
            "MetaBAW is Running.",
            "Command: metabaw binning -p '/data/reads with spaces'",
        ])
        self.assertNotIn("ALL DONE.", lines)


if __name__ == "__main__":
    unittest.main()
