"""Independent correctness checks for constraint-based HypoSpace sampling."""

from itertools import product
import json
import unittest

from constraint_sampling import (
    GraphTask,
    VoxelTask,
    add_pattern,
    compile_grammar,
    parse_encoding,
)


N4_CASES = (
    (4, ((0, 1), (1, 2), (2, 3)), (0, 1)),
    (4, ((0, 1), (0, 2), (1, 3), (2, 3)), (0, 1)),
    (4, ((2, 0), (2, 1), (1, 3)), (1, 2)),
)


def independent_edges(size):
    return tuple(
        (source, target)
        for source in range(size)
        for target in range(size)
        if source != target
    )


def independent_reachable(size, edges, source):
    reached = {source}
    pending = [source]
    while pending:
        current = pending.pop()
        for left, right in edges:
            if left == current and right not in reached:
                reached.add(right)
                pending.append(right)
    return reached


def independent_is_valid(size, observations, encoding):
    edges = [
        edge
        for edge, bit in zip(independent_edges(size), encoding)
        if bit == "1"
    ]
    outgoing = [[] for _ in range(size)]
    indegree = [0] * size
    for source, target in edges:
        outgoing[source].append(target)
        indegree[target] += 1

    ready = [node for node, degree in enumerate(indegree) if degree == 0]
    visited = 0
    while ready:
        node = ready.pop()
        visited += 1
        for target in outgoing[node]:
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    if visited != size:
        return False

    for source, affected in observations:
        actual = independent_reachable(size, edges, source) - {source}
        if actual != set(affected):
            return False
    return True


def independent_count(task):
    return sum(
        independent_is_valid(task.size, task.observations, "".join(bits))
        for bits in product("01", repeat=task.size * (task.size - 1))
    )


def observations_from_graph(size, edges, sources):
    return tuple(
        (source, tuple(sorted(independent_reachable(size, edges, source) - {source})))
        for source in sources
    )


def expand_grammar(grammar):
    rules = {}
    root_rhs = None
    for line in grammar.splitlines():
        name, rhs = line.split(" ::= ", 1)
        if name == "root":
            root_rhs = rhs
        else:
            rules[name] = rhs

    cache = {}

    def expand(name):
        if name in cache:
            return cache[name]
        outputs = set()
        for alternative in rules[name].split(" | "):
            parts = alternative.split()
            terminal = json.loads(parts[0])
            suffix = expand(parts[1]) if len(parts) == 2 else {""}
            outputs.update(terminal + value for value in suffix)
        cache[name] = outputs
        return outputs

    if root_rhs is None:
        raise AssertionError("Grammar has no root rule.")
    root_parts = root_rhs.split()
    left_quote = json.loads(root_parts[0])
    body = expand(root_parts[1])
    right_quote = json.loads(root_parts[2])
    return {left_quote + value + right_quote for value in body}


class ConstraintSamplingTests(unittest.TestCase):
    def n4_tasks(self):
        for size, edges, sources in N4_CASES:
            observations = observations_from_graph(size, edges, sources)
            yield GraphTask(size, observations)

    def test_n4_offline_counts_match_independent_enumeration(self):
        for task in self.n4_tasks():
            with self.subTest(observations=task.observations):
                self.assertEqual(task.count(), independent_count(task))

    def test_n4_certificates_exclude_only_invalid_graphs(self):
        observed_statuses = set()
        for task in self.n4_tasks():
            encodings = [
                "".join(bits)
                for bits in product("01", repeat=len(independent_edges(task.size)))
            ]
            independently_valid = {
                encoding: independent_is_valid(
                    task.size, task.observations, encoding
                )
                for encoding in encodings
            }
            certificates = set()

            for encoding in encodings:
                status, certificate = task.inspect(encoding)
                observed_statuses.add(status)
                self.assertEqual(status == "valid", independently_valid[encoding])
                if status == "valid":
                    continue

                self.assertTrue(certificate)
                self.assertTrue(
                    all(encoding[index] == symbol for index, symbol in certificate)
                )
                certificates.add(tuple(certificate))

            for certificate in certificates:
                for encoding in encodings:
                    matches = all(
                        encoding[index] == symbol for index, symbol in certificate
                    )
                    if matches:
                        self.assertFalse(independently_valid[encoding])

        self.assertTrue({"valid", "extra_effect", "missing_effect", "cycle"}
                        <= observed_statuses)

    def test_grammar_accepts_exactly_unblocked_valid_assignments(self):
        task = next(self.n4_tasks())
        domains = task.allowed()
        patterns = []
        assignments = ["".join(symbols) for symbols in product(*domains)]

        for assignment in assignments:
            status, certificate = task.inspect(assignment)
            if status != "valid":
                add_pattern(patterns, certificate)

        accepted = expand_grammar(compile_grammar(domains, patterns))
        expected = {
            json.dumps(assignment)
            for assignment in assignments
            if independent_is_valid(task.size, task.observations, assignment)
        }
        self.assertEqual(accepted, expected)

    def test_empty_observations_and_small_domains(self):
        self.assertEqual(GraphTask(2, ()).count(), 3)
        self.assertEqual(GraphTask(4, ()).count(), 543)
        self.assertEqual(GraphTask(5, ()).count(), 29281)

        voxel = VoxelTask((0, 2, 5), 5)
        self.assertEqual(voxel.count(), 125)
        self.assertEqual(tuple(map(len, voxel.domains)), (5, 5, 5))

        self.assertIsNone(parse_encoding("not JSON", ("01", "01")))
        self.assertIsNone(parse_encoding('"02"', ("01", "01")))
        self.assertIsNone(parse_encoding(None, ("01", "01")))
        self.assertEqual(parse_encoding('"01"', ("01", "01")), "01")

    def test_task_input_boundaries(self):
        self.assertEqual(len(VoxelTask((0,), 9).domains[0]), 9)
        for positions, height in (
            ((0,), 0),
            ((0,), 10),
            ((), 3),
            ((1, 1), 3),
            ((2, 1), 3),
            ((9,), 3),
        ):
            with self.subTest(positions=positions, height=height):
                with self.assertRaises(ValueError):
                    VoxelTask(positions, height)

        self.assertEqual(GraphTask(2, ()).size, 2)
        self.assertEqual(GraphTask(5, ()).size, 5)
        for size, observations in (
            (1, ()),
            (6, ()),
            (3, ((0, (0,)),)),
            (3, ((3, ()),)),
            (3, ((0, (1, 1)),)),
            (3, ((0, (3,)),)),
            (3, ((0, (1,)), (0, (2,)))),
        ):
            with self.subTest(size=size, observations=observations):
                with self.assertRaises(ValueError):
                    GraphTask(size, observations)

    def test_pattern_subsumption_and_exhausted_grammar(self):
        patterns = []
        add_pattern(patterns, ((0, "1"), (1, "0")))
        add_pattern(patterns, ((0, "1"),))
        add_pattern(patterns, ((0, "1"), (2, "0")))
        self.assertEqual(patterns, [((0, "1"),)])

        with self.assertRaises(ValueError):
            compile_grammar(("01",), [((0, "0"),), ((0, "1"),)])


if __name__ == "__main__":
    unittest.main()
