#!/usr/bin/env python
"""Audit and materialize the pseudotrack correct-only population.

The 96 ``pseudocode_codes_none`` files are applicability rows, not 96 unique
programs. This script resolves each row to its problem's correct pseudocode,
collapses the rows by problem, and forms deterministic cover-all McMiner-M bags
using the repository's real bag-forming implementation.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NONE_DIR = ROOT / "dataset" / "pseudocode_track" / "pseudocode_codes_none"
DEFAULT_PROBLEMS = ROOT / "dataset" / "pseudocode_track" / "problems_pseudocode.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "correct_only"
MINER_MODULE = ROOT / "src" / "run_infer_misc_multi.py"


def _load_miner_module():
    src_dir = str(MINER_MODULE.parent)
    root_dir = str(ROOT)
    for path in (src_dir, root_dir):
        if path not in sys.path:
            sys.path.insert(0, path)
    spec = importlib.util.spec_from_file_location("correct_only_bag_former", MINER_MODULE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load bag former: {MINER_MODULE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def prepare(
    none_dir: Path,
    problems_file: Path,
    output_dir: Path,
    bag_size: int = 5,
    passes: int = 1,
    seed: int = 42,
) -> dict[str, Any]:
    """Write row, unique-program, bag, and summary artifacts and return summary."""
    none_dir = Path(none_dir)
    problems_file = Path(problems_file)
    output_dir = Path(output_dir)
    if bag_size < 1:
        raise ValueError("bag_size must be at least 1")
    if passes < 1:
        raise ValueError("passes must be at least 1")
    if not none_dir.is_dir():
        raise FileNotFoundError(f"correct-only directory not found: {none_dir}")
    if not problems_file.is_file():
        raise FileNotFoundError(f"problems file not found: {problems_file}")

    miner = _load_miner_module()
    problems = miner.load_json_data(str(problems_file))
    correct_solutions = miner.get_correct_solutions(problems)

    rows: list[dict[str, Any]] = []
    by_problem: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for source in sorted(none_dir.glob("*.json"), key=lambda path: path.name):
        data = json.loads(source.read_text(encoding="utf-8"))
        problem_id = data.get("problem_id")
        if problem_id is None:
            raise ValueError(f"missing problem_id: {source.name}")
        solutions = data.get("solutions") or []
        if not solutions or any(solution.get("generated_code") != "NONE" for solution in solutions):
            raise ValueError(f"expected only literal NONE placeholders: {source.name}")
        available = correct_solutions.get(problem_id)
        if not available:
            raise ValueError(f"no correct pseudocode for problem {problem_id}: {source.name}")
        row = {
            "source_file": source.name,
            "problem_id": problem_id,
            "inapplicable_misconception_id": data.get("misconception_id"),
            "correct_code": available[0],
        }
        rows.append(row)
        by_problem[problem_id].append(row)

    program_ids = set(by_problem)
    extra_programs = set(correct_solutions) - program_ids
    missing_programs = program_ids - set(correct_solutions)
    if missing_programs:
        raise ValueError(f"correct solutions missing for problem IDs: {sorted(missing_programs)}")

    programs = []
    for problem_id in sorted(by_problem):
        grouped_rows = by_problem[problem_id]
        programs.append(
            {
                "problem_id": problem_id,
                "correct_code": grouped_rows[0]["correct_code"],
                "source_files": [row["source_file"] for row in grouped_rows],
                "inapplicable_misconception_ids": [
                    row["inapplicable_misconception_id"] for row in grouped_rows
                ],
            }
        )

    if isinstance(problems, list):
        problems_for_bags = [problem for problem in problems if problem.get("id") in program_ids]
    else:
        problems_for_bags = {
            key: problem
            for key, problem in problems.items()
            if (int(key) if isinstance(key, str) and key.isdigit() else key) in program_ids
        }

    random.seed(seed)
    formed = miner.create_correct_only_bags(
        problems_for_bags,
        0,
        "fixed",
        None,
        None,
        bag_size,
        cover_all=True,
        passes=passes,
    )
    bags = []
    for index, bag in enumerate(formed):
        pass_index = index // ((len(programs) + bag_size - 1) // bag_size)
        bags.append(
            {
                "bag_id": f"correct_pass_{pass_index}_bag_{index % ((len(programs) + bag_size - 1) // bag_size)}",
                "pass_index": pass_index,
                "programs": [
                    {
                        "problem_id": item["problem_id"],
                        "correct_code": item["solutions"][0]["generated_code"],
                        "source_files": [row["source_file"] for row in by_problem[item["problem_id"]]],
                    }
                    for item in bag
                ],
            }
        )

    summary = {
        "correct_only_rows": len(rows),
        "unique_programs": len(programs),
        "duplicate_rows": len(rows) - len(programs),
        "bag_size": bag_size,
        "passes": passes,
        "seed": seed,
        "total_bags": len(bags),
        "bag_sizes": [len(bag["programs"]) for bag in bags],
        "extra_correct_programs_without_none_rows": sorted(extra_programs),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "rows.jsonl", rows)
    _write_jsonl(output_dir / "programs.jsonl", programs)
    (output_dir / "bags.json").write_text(
        json.dumps(bags, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--none-dir", type=Path, default=DEFAULT_NONE_DIR)
    parser.add_argument("--problems-file", type=Path, default=DEFAULT_PROBLEMS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bag-size", type=int, default=5)
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = prepare(
        args.none_dir,
        args.problems_file,
        args.output_dir,
        args.bag_size,
        args.passes,
        args.seed,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Artifacts written to: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
