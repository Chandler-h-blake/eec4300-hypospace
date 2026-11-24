"""Validate an experiment transcript and report paired HypoSpace metrics."""

import argparse
import json
from pathlib import Path
import random
from statistics import mean

from experiment import METHODS, SETTINGS, make_cases, parse_heights


METRICS = ("validity", "uniqueness", "recovery")


def validate(result, case, budget):
    records = result["records"]
    if len(records) != budget or result["budget"] != budget or result["space_size"] != case.space_size:
        raise ValueError("Incorrect query budget or hypothesis space size.")
    parsed, seen = [], []
    validity = 0
    for record in records:
        heights = parse_heights(record["response"])
        valid = case.valid(heights)
        new = valid and heights not in seen
        if valid != record["valid"] or new != record["new_valid"]:
            raise ValueError("A trial record disagrees with its parsed response.")
        validity += int(valid)
        if heights is not None and heights not in parsed:
            parsed.append(heights)
        if new:
            seen.append(heights)
    if (validity / budget, len(parsed) / budget, len(seen) / case.space_size) != tuple(
        result[metric] for metric in METRICS
    ):
        raise ValueError("Stored metrics disagree with the transcript.")


def read_results(path):
    with path.open(encoding="utf8") as source:
        config = json.loads(next(source))
        results = [json.loads(line) for line in source]
    cases = list(make_cases(config["per_setting"], config["seed"]))
    groups = []
    for case in cases:
        rows = [
            row for row in results
            if tuple(row["case"]["positions"]) == case.positions
            and row["case"]["max_height"] == case.max_height
            and row["case"]["seed"] == case.seed
        ]
        if sorted(row["method"] for row in rows) != sorted(METHODS):
            raise ValueError(f"Missing or duplicate methods for case {case}.")
        for row in rows:
            validate(row, case, config["budget"])
        groups.append({row["method"]: row for row in rows})
    if len(results) != len(groups) * len(METHODS):
        raise ValueError("The transcript contains unexpected cases.")
    return config, cases, groups


def paired_interval(deltas):
    rng = random.Random(4301)
    draws = sorted(
        mean(mean(rng.choices(setting, k=len(setting))) for setting in deltas)
        for _ in range(4000)
    )
    return draws[100], draws[3899]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transcript", type=Path)
    args = parser.parse_args()
    config, cases, groups = read_results(args.transcript)
    print(f"{config['model']}, {len(cases)} cases, {config['budget']} calls per method and case")
    print("Method             Validity  Uniqueness  Recovery  Input tokens  Output tokens")
    for method in METHODS:
        rows = [group[method] for group in groups]
        values = [mean(row[metric] for row in rows) for metric in METRICS]
        tokens = [
            mean(record[key] for row in rows for record in row["records"])
            for key in ("prompt_tokens", "completion_tokens")
        ]
        print(f"{method:18} {values[0]:8.2%} {values[1]:11.2%} {values[2]:9.2%}"
              f" {tokens[0]:13.2f} {tokens[1]:14.2f}")

    print("\nMean Recovery by setting")
    for width, height in SETTINGS:
        subset = [group for case, group in zip(cases, groups)
                  if (case.width, case.max_height) == (width, height)]
        values = "  ".join(f"{method}={mean(g[method]['recovery'] for g in subset):.2%}"
                           for method in METHODS)
        print(f"{height}^{width} = {height ** width:3} hypotheses  {values}")

    print("\nPaired Recovery differences for adaptive_prefix (percentage points, 95% bootstrap CI)")
    for control in METHODS[:-1]:
        deltas = [
            [group["adaptive_prefix"]["recovery"] - group[control]["recovery"]
             for case, group in zip(cases, groups) if (case.width, case.max_height) == setting]
            for setting in SETTINGS
        ]
        estimate = mean(mean(setting) for setting in deltas)
        lo, hi = paired_interval(deltas)
        print(f"vs {control:12} {estimate * 100:+6.2f} [{lo * 100:+.2f}, {hi * 100:+.2f}]")


if __name__ == "__main__":
    main()
