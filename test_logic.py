import os
import sys
import unittest


# Allow running without installing the package
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))


from meta_smartgrid_rl.env import SustainableGridEnv  # noqa: E402


class DummyCritic:
    def __init__(self, score: int = 7, feedback: str = "ok") -> None:
        self._score = score
        self._feedback = feedback

    def generate_score(self, history):  # match env call signature
        return self._score, self._feedback


class TestSustainableGridEnv(unittest.TestCase):
    def test_reset_shape_and_ranges(self):
        env = SustainableGridEnv()
        obs, info = env.reset()

        self.assertEqual(obs.shape, (4,))
        solar, demand, battery, hour = obs

        # Observations are normalized to [0, 1]
        for v in (solar, demand, battery, hour):
            self.assertGreaterEqual(float(v), 0.0)
            self.assertLessEqual(float(v), 1.0)

        self.assertEqual(float(hour), 0.0)
        self.assertIsInstance(info, dict)

    def test_step_returns_expected_tuple(self):
        env = SustainableGridEnv()
        env.reset()

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        self.assertEqual(obs.shape, (4,))
        self.assertIsInstance(float(reward), float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIn("grid_import", info)

    def test_truncation_calls_critic_and_sets_info(self):
        env = SustainableGridEnv()
        env.reset()

        # Avoid real network calls; ensure deterministic scoring
        env.critic = DummyCritic(score=9, feedback="Great strategy.")

        info = {}
        for _ in range(24):
            action = env.action_space.sample()
            _, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        self.assertTrue(truncated)
        self.assertIn("llm_score", info)
        self.assertIn("llm_feedback", info)
        self.assertEqual(info["llm_score"], 9)
        self.assertEqual(info["llm_feedback"], "Great strategy.")

        # History should record exactly one entry per executed step
        self.assertEqual(len(env.history), 24)


if __name__ == "__main__":
    unittest.main(verbosity=2)

