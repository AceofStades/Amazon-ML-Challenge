import itertools
import unittest

import numpy as np

from amzn.decoding import decode_expected_exact, expected_f05_curve


def enumerate_curve(probabilities):
    curve = np.zeros(len(probabilities) + 1)
    for outcome in itertools.product((0, 1), repeat=len(probabilities)):
        mass = np.prod([p if actual else 1 - p for p, actual in zip(probabilities, outcome)])
        actual = sum(outcome)
        curve[0] += mass * (actual == 0)
        for size in range(1, len(curve)):
            curve[size] += mass * 1.25 * sum(outcome[:size]) / (size + 0.25 * actual)
    return curve


class ExactDecodingTest(unittest.TestCase):
    def test_matches_every_outcome(self):
        for probabilities in ([], [0.55], [1, 0], [0.9, 0.65, 0.1], [1, 0.75, 0.5, 0.001, 0]):
            with self.subTest(probabilities=probabilities):
                np.testing.assert_allclose(expected_f05_curve(np.array(probabilities)),
                                           enumerate_curve(probabilities), atol=1e-12, rtol=1e-12)

    def test_one_uncertain_link_beats_empty_above_half(self):
        chosen = decode_expected_exact(np.array([0]), np.array([7]), np.array([0.55]))
        self.assertEqual(chosen, {0: [7]})

    def test_threshold_and_zero_tails(self):
        self.assertEqual(decode_expected_exact(np.array([0, 0]), np.array([7, 8]), np.array([0.55, 0.])),
                         {0: [7]})
        self.assertEqual(decode_expected_exact(np.array([0]), np.array([7]), np.array([0.55]), 0.6), {0: []})

    def test_empty_input(self):
        self.assertEqual(decode_expected_exact(np.array([], dtype=int), np.array([], dtype=int), np.array([])), {})

    def test_existing_decode_dispatch(self):
        from amzn.train_v3 import decode
        source, target, probability = np.array([0]), np.array([7]), np.array([0.55])
        self.assertEqual(decode(source, target, probability, "expected-exact", 0.05), {0: [7]})
        self.assertEqual(decode(source, target, probability, "expected", 0.05), {0: []})
        with self.assertRaises(ValueError):
            decode(source, target, probability, "expected-exact", 0.05, 0.2)

    def test_invalid_probabilities(self):
        for probabilities in ([np.nan], [-0.1], [1.1], [[0.5]]):
            with self.subTest(probabilities=probabilities):
                with self.assertRaises(ValueError):
                    expected_f05_curve(np.array(probabilities))


if __name__ == "__main__":
    unittest.main()
