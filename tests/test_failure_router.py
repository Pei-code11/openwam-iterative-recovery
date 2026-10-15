import unittest

import numpy as np

from openwam_recovery.failure_router import ACTIONS, SOURCES, FailureRouter
from scripts.run_failure_router_benchmark import build_dataset
from scripts.run_failure_router_hard_benchmark import (
    ACTIONS as HARD_ACTIONS,
    branch_success,
    build_dataset as build_hard_dataset,
)


class FailureRouterTests(unittest.TestCase):
    def test_fit_predict_and_decide(self):
        x, y, source = build_dataset(range(24), "train")
        router = FailureRouter(hidden=16, seed=3)
        router.fit(x, y, source, epochs=8)
        self.assertEqual(router.parameter_count, (15 + 1) * 16 + (16 + 1) * 16 + (16 + 1) * (len(ACTIONS) + len(SOURCES)))
        probs = router.predict_proba(x[:3])
        self.assertEqual(probs.shape, (3, len(ACTIONS)))
        self.assertTrue(np.isfinite(probs).all())
        decision = router.decide(x[0])
        self.assertIn(decision.action, ACTIONS)
        self.assertEqual(set(decision.source_probabilities), set(SOURCES))
        fast = decision.as_fast_json()
        self.assertEqual(fast["choice"], decision.action)
        self.assertGreaterEqual(fast["score"], 0.0)
        self.assertLessEqual(fast["score"], 10.0)
        self.assertIsInstance(fast["noul"], bool)

    def test_episode_rows_are_not_frame_split_by_builder(self):
        x1, y1, s1 = build_dataset(range(4), "train")
        x2, y2, s2 = build_dataset(range(4, 8), "train")
        self.assertEqual(x1.shape, x2.shape)
        self.assertEqual(y1.shape, y2.shape)
        self.assertEqual(s1.shape, s2.shape)
        self.assertFalse(np.array_equal(x1, x2))

    def test_hard_benchmark_uses_paired_terminal_branches(self):
        x, y, episodes = build_hard_dataset(range(12), "test", 8121)
        self.assertEqual(x.shape, (12, 15))
        self.assertEqual(y.shape, (12, len(HARD_ACTIONS)))
        for episode, labels in zip(episodes, y):
            expected = np.asarray([float(branch_success(episode, action))
                                   for action in HARD_ACTIONS])
            np.testing.assert_array_equal(labels, expected)


if __name__ == "__main__":
    unittest.main()
