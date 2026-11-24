"""Run paired sampling experiments on the 3D HypoSpace task."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from itertools import combinations, product
import json
import math
from pathlib import Path
import random
import re
import time
from urllib.request import Request, urlopen


METHODS = ("memory", "complexity", "fixed_prefix", "adaptive_prefix")
SETTINGS = ((3, 3), (3, 5), (4, 3))
ARRAY_GRAMMAR = '''root ::= "[" ws integer (ws "," ws integer)* ws "]"
integer ::= "-"? ("0" | [1-9] [0-9]*)
ws ::= " "?
'''


@dataclass(frozen=True)
class Case:
    positions: tuple
    max_height: int
    seed: int

    @property
    def width(self):
        return len(self.positions)

    @property
    def space_size(self):
        return self.max_height ** self.width

    def valid(self, heights):
        return heights is not None and len(heights) == self.width and all(
            type(height) is int and 1 <= height <= self.max_height for height in heights
        )


def make_cases(per_setting, seed):
    rng = random.Random(seed)
    for width, max_height in SETTINGS:
        positions = list(combinations(range(9), width))
        for footprint in rng.sample(positions, per_setting):
            yield Case(footprint, max_height, rng.randrange(1, 2**30))


def complexity_schedule(case, budget):
    counts = [1]
    for _ in case.positions:
        next_counts = [0] * (len(counts) + case.max_height)
        for total, count in enumerate(counts):
            for height in range(1, case.max_height + 1):
                next_counts[total + height] += count
        counts = next_counts
    levels = [level for level, count in enumerate(counts) if count]
    allocated = [0] * len(counts)
    schedule = []
    for query in range(1, budget + 1):
        level = max(levels, key=lambda c: query * counts[c] - allocated[c] * case.space_size)
        allocated[level] += 1
        schedule.append(level)
    return schedule


class PrefixPolicy:
    """Treat each two-height prefix as an arm with a finite number of completions."""

    def __init__(self, case, adaptive):
        self.arms = list(product(range(1, case.max_height + 1), repeat=2))
        random.Random(case.seed).shuffle(self.arms)
        self.capacity = case.max_height ** (case.width - 2)
        self.adaptive = adaptive
        self.attempts = [0] * len(self.arms)
        self.successes = [0] * len(self.arms)

    def choose(self, query, seen):
        if not self.adaptive:
            return query % len(self.arms)
        covered = [sum(item[:2] == arm for item in seen) for arm in self.arms]

        def score(index):
            if covered[index] == self.capacity:
                return -1
            tries = self.attempts[index]
            mean = (1 + self.successes[index]) / (2 + tries)
            exploration = math.sqrt(math.log(query + 2) / (2 * (1 + tries)))
            return (mean + exploration) * (1 - covered[index] / self.capacity)

        return max(range(len(self.arms)), key=score)

    def observe(self, index, heights, is_new):
        self.attempts[index] += 1
        self.successes[index] += int(is_new and heights[:2] == self.arms[index])


def prompt(case, seen, prefix=None, complexity=None):
    grid = "\n".join(
        " ".join(str(int(cell in case.positions)) for cell in range(row * 3, row * 3 + 3))
        for row in range(3)
    )
    coordinates = ", ".join(f"({cell // 3 + 1},{cell % 3 + 1})" for cell in case.positions)
    lines = [
        "Reconstruct one gravity-supported voxel structure from this 3x3 top view:",
        grid,
        f"Occupied cells in row-major order: {coordinates}.",
        f"Return a JSON array of exactly {case.width} integer column heights in that order.",
        f"Each height must be from 1 to {case.max_height}. Empty cells are omitted.",
    ]
    if prefix is not None:
        lines.append(f"The first two heights must be {prefix[0]} and {prefix[1]}, in that order.")
    if complexity is not None:
        lines.append(f"The sum of the heights must be exactly {complexity}.")
    if seen:
        history = ", ".join(json.dumps(item) for item in seen[-12:])
        lines.append(f"Choose a different answer from these previous valid answers: {history}.")
    lines.append("Return only the JSON array.")
    return "\n".join(lines)


def parse_heights(content):
    match = re.fullmatch(r"\s*(?:```(?:json)?\s*)?(\[[^\[\]\n]*\])(?:\s*```)?\s*", content)
    if not match:
        return None
    try:
        heights = json.loads(match[1])
    except json.JSONDecodeError:
        return None
    return tuple(heights) if all(type(height) is int for height in heights) else None


def sample(endpoint, model, message, seed, constrain_format=False, *, grammar=None):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": message}],
        "temperature": 0.8,
        "top_p": 0.95,
        "max_tokens": 32,
        "seed": seed,
    }
    if grammar is not None or constrain_format:
        body["grammar"] = grammar if grammar is not None else ARRAY_GRAMMAR
    request = Request(
        endpoint.rstrip("/") + "/chat/completions",
        json.dumps(body).encode("utf8"),
        {"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=120) as response:
        result = json.load(response)
    return result["choices"][0]["message"]["content"], result["usage"]


def run_case(case, method, budget, generate):
    seen, parsed, records = [], [], []
    valid = 0
    policy = PrefixPolicy(case, method == "adaptive_prefix") if method.endswith("_prefix") else None
    schedule = complexity_schedule(case, budget) if method == "complexity" else None
    for i in range(budget):
        arm = policy.choose(i, seen) if policy else None
        prefix = policy.arms[arm] if policy else None
        complexity = schedule[i] if schedule else None
        content, usage = generate(prompt(case, seen, prefix, complexity), case.seed + i)
        heights = parse_heights(content)
        is_valid = case.valid(heights)
        is_new = is_valid and heights not in seen
        if heights is not None and heights not in parsed:
            parsed.append(heights)
        if is_new:
            seen.append(heights)
        valid += int(is_valid)
        if policy:
            policy.observe(arm, heights, is_new)
        records.append({
            "target_prefix": prefix,
            "target_complexity": complexity,
            "response": content,
            "valid": is_valid,
            "new_valid": is_new,
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
        })
    return {
        "case": asdict(case),
        "method": method,
        "budget": budget,
        "space_size": case.space_size,
        "validity": valid / budget,
        "uniqueness": len(parsed) / budget,
        "recovery": len(seen) / case.space_size,
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18385/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-setting", type=int, default=8)
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--constrain-format", action="store_true", help="Require an integer array through llama.cpp GBNF.")
    parser.add_argument("--seed", type=int, default=4300)
    args = parser.parse_args()
    if min(args.budget, args.workers) < 1 or not 1 <= args.per_setting <= 84:
        parser.error("Use positive budget and worker counts and 1 to 84 cases per setting.")
    config = {
        "model": args.model,
        "per_setting": args.per_setting,
        "budget": args.budget,
        "seed": args.seed,
    }
    if args.constrain_format:
        config["constrain_format"] = True
    previous = []
    if args.output.is_file() and args.output.stat().st_size:
        with args.output.open(encoding="utf8") as source:
            if json.loads(next(source)) != config:
                parser.error("Output already contains results for a different configuration.")
            previous = [json.loads(line) for line in source]
    completed = [(entry["case"], entry["method"]) for entry in previous]
    started = time.monotonic()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cases = list(make_cases(args.per_setting, args.seed))
    jobs = [
        (i, case, method)
        for i, case in enumerate(cases, start=1)
        for method in METHODS
        if ({**asdict(case), "positions": list(case.positions)}, method) not in completed
    ]

    def evaluate(job):
        i, case, method = job
        return i, run_case(
            case, method, args.budget,
            lambda message, seed: sample(
                args.endpoint, args.model, message, seed, args.constrain_format
            ),
        )

    with args.output.open("a", encoding="utf8") as output, ThreadPoolExecutor(
        max_workers=args.workers
    ) as workers:
        if output.tell() == 0:
            output.write(json.dumps(config) + "\n")
            output.flush()
        for i, result in workers.map(evaluate, jobs):
            output.write(json.dumps(result) + "\n")
            output.flush()
            print(
                f"{i}/{len(cases)} {result['method']}: V={result['validity']:.3f} "
                f"U={result['uniqueness']:.3f} R={result['recovery']:.3f} "
                f"({time.monotonic() - started:.0f}s)", flush=True,
            )


if __name__ == "__main__":
    main()
