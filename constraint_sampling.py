"""Sample distinct HypoSpace hypotheses using constraints learned from failed proposals."""

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
import json
import random


def full_pattern(value):
    return tuple(enumerate(value))


def add_pattern(patterns, pattern):
    """Keep only patterns that are not already covered by a more general one."""
    if any(all(literal in pattern for literal in old) for old in patterns):
        return
    patterns[:] = [old for old in patterns if not all(literal in old for literal in pattern)]
    patterns.append(pattern)


def compile_grammar(domains, patterns):
    """Compile the complement of forbidden partial assignments to a compact GBNF grammar."""
    rules = []

    @lru_cache(maxsize=None)
    def build(index, active):
        if () in active:
            return None
        if index == len(domains):
            return ""
        choices = []
        for symbol in domains[index]:
            remaining = []
            for clause in active:
                position, forbidden = clause[0]
                if position > index:
                    remaining.append(clause)
                elif symbol == forbidden:
                    remaining.append(clause[1:])
            suffix = build(index + 1, tuple(remaining))
            if suffix is not None:
                choices.append(" ".join(part for part in (json.dumps(symbol), suffix) if part))
        if not choices:
            return None
        name = f"s{len(rules)}"
        rules.append(f"{name} ::= " + " | ".join(choices))
        return name

    first = build(0, tuple(patterns))
    if first is None:
        raise ValueError("No admissible encoding remains under the generation constraints.")
    quote = json.dumps('"')
    return "\n".join([f"root ::= {quote} {first} {quote}", *rules]) + "\n"


def parse_encoding(response, domains):
    if not isinstance(response, str):
        return None
    try:
        value = json.loads(response)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, str) or len(value) != len(domains):
        return None
    return value if all(symbol in domain for symbol, domain in zip(value, domains)) else None


@dataclass(frozen=True)
class VoxelTask:
    positions: tuple
    height: int

    def __post_init__(self):
        if type(self.height) is not int or not 1 <= self.height <= 9:
            raise ValueError("The voxel encoding supports maximum heights from 1 to 9.")
        if (
            not self.positions
            or any(type(cell) is not int or not 0 <= cell < 9 for cell in self.positions)
            or any(a >= b for a, b in zip(self.positions, self.positions[1:]))
        ):
            raise ValueError("Provide at least one distinct 3x3 grid position in row-major order.")

    @property
    def domains(self):
        return ("123456789"[:self.height],) * len(self.positions)

    def allowed(self):
        return self.domains

    def describe(self):
        grid = "\n".join(
            " ".join(str(int(i in self.positions)) for i in range(row * 3, row * 3 + 3))
            for row in range(3)
        )
        return (
            f"Propose a gravity-supported voxel structure for this 3x3 top view:\n{grid}\n"
            f"In row-major order, choose a height from 1 through {self.height} for each occupied cell. "
            f"Return exactly {len(self.positions)} height digits as one quoted string. "
            "For example, digits one and two would be written as \"12\"."
        )

    def inspect(self, value):
        return "valid", ()

    def count(self):
        return self.height ** len(self.positions)


@dataclass(frozen=True)
class GraphTask:
    size: int
    observations: tuple

    def __post_init__(self):
        if type(self.size) is not int or not 2 <= self.size <= 5:
            raise ValueError("The graph encoding supports 2 to 5 labeled nodes.")
        sources = []
        for source, affected in self.observations:
            if type(source) is not int or not 0 <= source < self.size or source in sources:
                raise ValueError("Each intervention must name a distinct node in the graph.")
            if source in affected or any(type(node) is not int or not 0 <= node < self.size for node in affected):
                raise ValueError("Intervention effects must name other nodes in the graph.")
            if any(node in affected[:i] for i, node in enumerate(affected)):
                raise ValueError("An intervention cannot list the same affected node twice.")
            sources.append(source)

    @property
    def edges(self):
        return tuple((i, j) for i in range(self.size) for j in range(self.size) if i != j)

    @property
    def domains(self):
        return ("01",) * (self.size * (self.size - 1))

    def allowed(self):
        closures = [(source, *affected) for source, affected in self.observations]
        return tuple(
            "0" if any(i in group and j not in group for group in closures) else "01"
            for i, j in self.edges
        )

    def describe(self):
        labels = "ABCDE"[:self.size]
        observations = []
        for source, effects in self.observations:
            affected = ", ".join(labels[i] for i in effects) or "none"
            observations.append(f"Intervene on {labels[source]}: affected nodes are {affected}.")
        edge_order = " ".join(labels[i] + labels[j] for i, j in self.edges)
        return (
            f"Propose a directed acyclic graph on nodes {', '.join(labels)}. "
            "An intervention affects precisely the nodes reachable through outgoing edges. "
            "Unlisted nodes are unaffected.\n" + "\n".join(observations) +
            f"\nEdge order: {edge_order}. Each pair XY denotes an edge from X to Y. "
            "Return a quoted bit string in that order, with 1 for an edge and 0 for no edge."
        )

    def graph(self, value):
        adjacency = [[] for _ in range(self.size)]
        for index, ((source, target), present) in enumerate(zip(self.edges, value)):
            if present == "1":
                adjacency[source].append((target, index))
        return adjacency

    def inspect(self, value):
        adjacency = self.graph(value)
        for source, affected in self.observations:
            paths = reachable_paths(adjacency, source)
            outside = next((node for node in paths if node != source and node not in affected), None)
            if outside is not None:
                return "extra_effect", tuple(sorted((i, "1") for i in paths[outside]))
            if any(node not in paths for node in affected):
                cut = tuple(
                    (i, "0") for i, ((u, v), domain) in enumerate(zip(self.edges, self.allowed()))
                    if u in paths and v not in paths and "1" in domain
                )
                return "missing_effect", cut
        for edge, (source, target) in enumerate(self.edges):
            if value[edge] == "1":
                cycle = reachable_paths(adjacency, target).get(source)
                if cycle is not None:
                    return "cycle", tuple(sorted((i, "1") for i in (*cycle, edge)))
        return "valid", ()

    def count(self):
        """Count reference DAGs offline. The online policy only sees observations."""
        adjacency = [[] for _ in range(self.size)]
        candidates = [edge for edge, domain in zip(self.edges, self.allowed()) if "1" in domain]

        def count_dags(index):
            if index == len(candidates):
                return int(all(
                    all(v in reachable_paths(adjacency, u) for v in affected)
                    for u, affected in self.observations
                ))
            source, target = candidates[index]
            total = count_dags(index + 1)
            if source not in reachable_paths(adjacency, target):
                adjacency[source].append((target, index))
                total += count_dags(index + 1)
                adjacency[source].pop()
            return total

        return count_dags(0)


def reachable_paths(adjacency, source):
    paths = {source: ()}
    queue = [source]
    for vertex in queue:
        for target, edge in adjacency[vertex]:
            if target not in paths:
                paths[target] = (*paths[vertex], edge)
                queue.append(target)
    return paths


def methods_for(task):
    if isinstance(task, VoxelTask):
        return ("memory", "fixed_prefix", "unique")
    return ("memory", "static", "unique", "learned")


def evaluate(task, method, budget, seed, generate):
    """Use observations and clause feedback, with no call to the reference counter."""
    domains = task.domains if method == "memory" else task.allowed()
    patterns, valid, records = [], [], []
    fixed_order = None
    if method == "fixed_prefix":
        fixed_order = list(product(*domains[:2]))
        random.Random(seed).shuffle(fixed_order)
    base = task.describe()
    for i in range(budget):
        prefix = fixed_order[i % len(fixed_order)] if fixed_order else ()
        query_domains = (*prefix, *domains[len(prefix):])
        message = base
        if prefix:
            message += f"\nThe first two digits must be {''.join(prefix)}."
        if valid:
            message += "\nTry a different answer from: " + ", ".join(map(json.dumps, valid[-12:])) + "."
        response, usage = generate(message, seed + i, compile_grammar(query_domains, patterns))
        value = parse_encoding(response, task.domains)
        if value is not None and (
            any(symbol not in domain for symbol, domain in zip(value, query_domains))
            or any(all(value[i] == symbol for i, symbol in clause) for clause in patterns)
        ):
            raise ValueError("The inference server returned an encoding excluded by its grammar.")
        status, pattern = task.inspect(value) if value is not None else ("format", ())
        if status == "valid" and value not in valid:
            valid.append(value)
        learned = None
        if method in ("unique", "learned") and value is not None:
            learned = pattern if method == "learned" and status != "valid" else full_pattern(value)
            if not learned:
                raise ValueError("A failed graph produced an empty conflict under compatible observations.")
            add_pattern(patterns, learned)
        records.append({
            "response": response,
            "status": status,
            "prefix": "".join(prefix),
            "blocked": learned,
            "input_tokens": usage["prompt_tokens"],
            "output_tokens": usage["completion_tokens"],
        })
    return records
