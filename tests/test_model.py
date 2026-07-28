from pathlib import Path
import unittest

from metabaw.model import Task, topological_order


class TaskGraphTests(unittest.TestCase):
    def test_topological_order(self) -> None:
        root = Path(".")
        tasks = [
            Task("c", "s", ("true",), root, deps=("b",)),
            Task("a", "s", ("true",), root),
            Task("b", "s", ("true",), root, deps=("a",)),
        ]
        self.assertEqual([task.id for task in topological_order(tasks)], ["a", "b", "c"])

    def test_cycle_is_rejected(self) -> None:
        root = Path(".")
        with self.assertRaisesRegex(ValueError, "cycle"):
            topological_order(
                [
                    Task("a", "s", ("true",), root, deps=("b",)),
                    Task("b", "s", ("true",), root, deps=("a",)),
                ]
            )

    def test_ordering_only_dependency_does_not_invalidate_task_fingerprint(
        self,
    ) -> None:
        root = Path(".")
        upstream = Task("upstream", "03_binning", ("true",), root)
        refinement = Task(
            "refinement",
            "04_refinement",
            ("true",),
            root,
            wait_for=(upstream.id,),
        )
        self.assertEqual(
            [upstream.id, refinement.id],
            [task.id for task in topological_order([refinement, upstream])],
        )
        without_barrier = Task(
            "refinement",
            "04_refinement",
            ("true",),
            root,
        )
        self.assertEqual(
            without_barrier.fingerprint([]),
            refinement.fingerprint([]),
        )


if __name__ == "__main__":
    unittest.main()

