"""Run and score the two-domain HypoSpace constraint sampling experiment."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from itertools import combinations
import json
from pathlib import Path
import random
from statistics import mean
import time

from constraint_sampling import GraphTask, VoxelTask, evaluate, methods_for, parse_encoding
from experiment import sample
from summarize import paired_interval


@dataclass(frozen=True)
class Instance:
    setting: str
    task: GraphTask | VoxelTask
    reference_count: int
    seed: int

    def header(self):
        return json.loads(json.dumps(asdict(self)))


def graph_instance(size, intervention_count, rng):
    order = rng.sample(range(size), size)
    density = rng.choice((0.25, 0.5, 0.75))
    edges = [(u, v) for u, v in combinations(order, 2) if rng.random() < density]
    observations = []
    for node in sorted(rng.sample(range(size), intervention_count)):
        reached = [node]
        for source in reached:
            reached.extend(v for u, v in edges if u == source and v not in reached)
        observations.append((node, tuple(sorted(v for v in reached if v != node))))
    return GraphTask(size, tuple(observations))


def make_instances(per_setting, seed, budget):
    rng = random.Random(seed)
    for columns, height in ((3, 5), (4, 5)):
        footprints = rng.sample(list(combinations(range(9), columns)), per_setting)
        for footprint in footprints:
            task = VoxelTask(footprint, height)
            yield Instance(f"voxel-{task.count()}", task, task.count(), rng.randrange(2**30))
    for size, interventions, low, high in ((4, 1, 30, 150), (5, 2, 80, 1200)):
        accepted, inspected = [], []
        for _ in range(5000):
            task = graph_instance(size, interventions, rng)
            if task in inspected:
                continue
            inspected.append(task)
            count = task.count()
            if max(budget, low) <= count <= high:
                accepted.append(task)
                yield Instance(f"causal-{size}", task, count, rng.randrange(2**30))
            if len(accepted) == per_setting:
                break
        if len(accepted) != per_setting:
            raise ValueError("Unable to sample enough distinct causal instances for those parameters.")


def run(args):
    config = {
        "model": args.model,
        "budget": args.budget,
        "per_setting": args.per_setting,
        "seed": args.seed,
    }
    jobs = [
        (case, method)
        for case in make_instances(args.per_setting, args.seed, args.budget)
        for method in methods_for(case.task)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    previous = []
    if args.output.is_file() and args.output.stat().st_size:
        saved, previous = load_transcript(args.output)
        if saved != config:
            raise ValueError("Output contains a different experiment configuration.")
    completed = [(row["instance"], row["method"]) for row in previous]
    todo = [(case, method) for case, method in jobs if (case.header(), method) not in completed]
    start = time.monotonic()

    def trial(job):
        case, method = job
        records = evaluate(
            case.task, method, args.budget, case.seed,
            lambda prompt, seed, grammar: sample(args.endpoint, args.model, prompt, seed, grammar=grammar),
        )
        return {"instance": case.header(), "method": method, "records": records}

    with args.output.open("a", encoding="utf8") as output, ThreadPoolExecutor(
        max_workers=args.workers
    ) as workers:
        if output.tell() == 0:
            output.write(json.dumps(config) + "\n")
        for index, result in enumerate(workers.map(trial, todo), start=1):
            output.write(json.dumps(result, separators=(",", ":")) + "\n")
            output.flush()
            valid = sum(record["status"] == "valid" for record in result["records"])
            print(f"{index}/{len(todo)} {result['instance']['setting']} {result['method']}"
                  f"  valid={valid}/{args.budget} ({time.monotonic() - start:.0f}s)", flush=True)


def load_transcript(path):
    with path.open(encoding="utf8") as source:
        return json.loads(next(source)), [json.loads(line) for line in source]


def score_row(case, method, records, budget):
    if len(records) != budget:
        raise ValueError("Trial does not match its call budget.")
    remaining = iter(records)
    rules = []

    def replay(prompt, seed, grammar):
        observed = next(remaining)
        rules.append(grammar.count("::="))
        return observed["response"], {
            "prompt_tokens": observed["input_tokens"],
            "completion_tokens": observed["output_tokens"],
        }

    expected = evaluate(case.task, method, budget, case.seed, replay)
    if json.loads(json.dumps(expected)) != json.loads(json.dumps(records)):
        raise ValueError("Replaying the transcript did not reproduce recorded decisions.")

    distinct, covered = [], []
    valid = conflicts = 0
    curve = []
    for index, record in enumerate(records, start=1):
        value = parse_encoding(record["response"], case.task.domains)
        if value is not None and value not in distinct:
            distinct.append(value)
        if record["status"] == "valid":
            valid += 1
            if value not in covered:
                covered.append(value)
        elif record["blocked"]:
            conflicts += int(len(record["blocked"]) < len(case.task.domains))
        if index in (5, 10, 20, budget):
            curve.append({"calls": index, "unique_valid": len(covered), "recovery": len(covered) / case.reference_count})
    return {
        "validity": valid / budget,
        "uniqueness": len(distinct) / budget,
        "recovery": len(covered) / case.reference_count,
        "unique_valid": len(covered),
        "ceiling_fraction": len(covered) / min(budget, case.reference_count),
        "input_tokens": mean(record["input_tokens"] for record in records),
        "output_tokens": mean(record["output_tokens"] for record in records),
        "generalized_conflicts": conflicts,
        "mean_grammar_rules": mean(rules),
        "failure_counts": {
            reason: sum(record["status"] == reason for record in records)
            for reason in ("format", "extra_effect", "missing_effect", "cycle")
        },
        "curve": curve,
    }


def summarize(path):
    config, rows = load_transcript(path)
    budget = config["budget"]
    trials = []
    for case in make_instances(config["per_setting"], config["seed"], budget):
        matches = [row for row in rows if row["instance"] == case.header()]
        methods = methods_for(case.task)
        if sorted(row["method"] for row in matches) != sorted(methods):
            raise ValueError(f"Missing or duplicate trials for {case.header()}.")
        trials.extend({
            "instance": case.header(), "method": row["method"],
            **score_row(case, row["method"], row["records"], budget),
        } for row in matches)
    if len(trials) != len(rows):
        raise ValueError("Unexpected trials in transcript.")
    return {"config": config, "trials": trials, "summary": aggregate(trials)}


def uniform_decode(grammar, seed):
    """Draw a uniformly random supported encoding from the ordered grammar."""
    lines = grammar.splitlines()
    current = lines[0].split()[3]
    rng = random.Random(seed)
    rules = {}
    counts = {"": 1}
    for line in lines[1:]:
        name, body = line.split(" ::= ")
        choices = []
        for option in body.split(" | "):
            terminal, *suffix = option.split()
            choices.append((json.loads(terminal), suffix[0] if suffix else ""))
        rules[name] = choices
        counts[name] = sum(counts[suffix] for _, suffix in choices)
    digits = []
    while current:
        choice = rng.randrange(counts[current])
        for symbol, suffix in rules[current]:
            if choice < counts[suffix]:
                digits.append(symbol)
                current = suffix
                break
            choice -= counts[suffix]
    return json.dumps("".join(digits)), {"prompt_tokens": 0, "completion_tokens": 0}


def uniform_controls(config, repeats):
    """Compare the causal policies when the generator uses no language model."""
    cases = [
        case for case in make_instances(config["per_setting"], config["seed"], config["budget"])
        if isinstance(case.task, GraphTask)
    ]
    metrics = {}
    for method in ("unique", "learned"):
        trials = []
        for case in cases:
            for repeat in range(repeats):
                seed = case.seed + 1000 * (repeat + 1)
                records = evaluate(
                    case.task, method, config["budget"], seed,
                    lambda prompt, step, grammar: uniform_decode(grammar, step),
                )
                trials.append(score_row(Instance(case.setting, case.task, case.reference_count, seed),
                                        method, records, config["budget"]))
        metrics[method] = {
            metric: mean(trial[metric] for trial in trials)
            for metric in ("validity", "uniqueness", "recovery", "unique_valid")
        }
    return {"repeats_per_instance": repeats, "methods": metrics}


def aggregate(trials):
    outputs = {}
    for family, control, proposed in (
        ("voxel", "fixed_prefix", "unique"),
        ("causal", "unique", "learned"),
    ):
        subset = [row for row in trials if row["instance"]["setting"].startswith(family)]
        episodes = [row for row in subset if row["method"] == proposed]
        metrics = ("validity", "uniqueness", "recovery", "unique_valid", "ceiling_fraction",
                   "input_tokens", "output_tokens", "generalized_conflicts", "mean_grammar_rules")
        by_method = {}
        for method in dict.fromkeys(row["method"] for row in subset):
            matching = [row for row in subset if row["method"] == method]
            by_method[method] = {
                **{metric: mean(row[metric] for row in matching) for metric in metrics},
                "failure_counts": {
                    reason: sum(row["failure_counts"][reason] for row in matching)
                    for reason in matching[0]["failure_counts"]
                },
                "curve": [
                    {"calls": point["calls"],
                     "unique_valid": mean(row["curve"][i]["unique_valid"] for row in matching),
                     "recovery": mean(row["curve"][i]["recovery"] for row in matching)}
                    for i, point in enumerate(matching[0]["curve"])
                ],
            }
        groups = []
        for setting in dict.fromkeys(row["instance"]["setting"] for row in episodes):
            deltas = []
            for treatment in episodes:
                if treatment["instance"]["setting"] == setting:
                    paired = next(row for row in subset if row["instance"] == treatment["instance"]
                                  and row["method"] == control)
                    deltas.append(treatment["recovery"] - paired["recovery"])
            groups.append(deltas)
        outputs[family] = {
            "instances": len(episodes), "methods": by_method,
            "comparison": {"treatment": proposed, "control": control,
                           "difference": mean(mean(group) for group in groups),
                           "interval": paired_interval(groups)},
        }
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run")
    runner.add_argument("--model", required=True)
    runner.add_argument("--output", type=Path, required=True)
    runner.add_argument("--endpoint", default="http://127.0.0.1:18385/v1")
    runner.add_argument("--budget", type=int, default=30)
    runner.add_argument("--per-setting", type=int, default=6)
    runner.add_argument("--seed", type=int, default=4312)
    runner.add_argument("--workers", type=int, default=8)
    scorer = commands.add_parser("score")
    scorer.add_argument("transcript", type=Path)
    scorer.add_argument("--uniform-repeats", type=int, default=0)
    args = parser.parse_args()
    if args.command == "run":
        if not 1 <= args.budget <= 30 or not 1 <= args.per_setting <= 8 or args.workers < 1:
            parser.error("Use 1–30 calls, 1–8 instances per setting, and at least one worker.")
        run(args)
    else:
        if args.uniform_repeats < 0:
            parser.error("The number of uniform repeats cannot be negative.")
        report = summarize(args.transcript)
        if args.uniform_repeats:
            report["uniform_causal"] = uniform_controls(report["config"], args.uniform_repeats)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
