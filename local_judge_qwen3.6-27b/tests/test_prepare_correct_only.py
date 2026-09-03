import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_correct_only.py"


def load_module():
    spec = importlib.util.spec_from_file_location("prepare_correct_only", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class PrepareCorrectOnlyIntegrationTest(unittest.TestCase):
    def test_bags_exclude_correct_programs_without_none_rows(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            none_dir = temp / "none"
            none_dir.mkdir()
            (none_dir / "problem_1_misc_7.json").write_text(
                json.dumps(
                    {
                        "problem_id": 1,
                        "misconception_id": 7,
                        "solutions": [{"generated_code": "NONE"}],
                    }
                ),
                encoding="utf-8",
            )
            problems_file = temp / "problems.json"
            problems_file.write_text(
                json.dumps(
                    [
                        {"id": 1, "solutions": ["program one"]},
                        {"id": 2, "solutions": ["program two"]},
                    ]
                ),
                encoding="utf-8",
            )

            summary = module.prepare(none_dir, problems_file, temp / "out")
            bags = json.loads((temp / "out" / "bags.json").read_text(encoding="utf-8"))

            self.assertEqual(summary["extra_correct_programs_without_none_rows"], [2])
            self.assertEqual(summary["bag_sizes"], [1])
            self.assertEqual(
                [item["problem_id"] for bag in bags for item in bag["programs"]],
                [1],
            )

    def test_real_dataset_produces_traceable_cover_all_bags(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            summary = module.prepare(
                none_dir=ROOT / "dataset" / "pseudocode_track" / "pseudocode_codes_none",
                problems_file=ROOT / "dataset" / "pseudocode_track" / "problems_pseudocode.json",
                output_dir=Path(temp_dir),
                bag_size=5,
                passes=1,
                seed=42,
            )

            self.assertEqual(summary["correct_only_rows"], 96)
            self.assertEqual(summary["unique_programs"], 19)
            self.assertEqual(summary["duplicate_rows"], 77)
            self.assertEqual(summary["total_bags"], 4)
            self.assertEqual(summary["bag_sizes"], [5, 5, 5, 4])

            programs = [json.loads(line) for line in (Path(temp_dir) / "programs.jsonl").read_text(encoding="utf-8").splitlines()]
            bags = json.loads((Path(temp_dir) / "bags.json").read_text(encoding="utf-8"))
            saved_summary = json.loads((Path(temp_dir) / "summary.json").read_text(encoding="utf-8"))

            self.assertEqual(saved_summary, summary)
            self.assertEqual(len(programs), 19)
            self.assertEqual(len({p["problem_id"] for p in programs}), 19)
            self.assertTrue(all(p["correct_code"] and p["source_files"] for p in programs))

            bag_problem_ids = [item["problem_id"] for bag in bags for item in bag["programs"]]
            self.assertEqual(len(bag_problem_ids), 19)
            self.assertEqual(set(bag_problem_ids), {p["problem_id"] for p in programs})
            self.assertEqual(len(bag_problem_ids), len(set(bag_problem_ids)))


if __name__ == "__main__":
    unittest.main()
