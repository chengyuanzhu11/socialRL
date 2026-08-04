import unittest

import numpy as np

import main


def deterministic_single_bout_env(detect_probability: float) -> main.SocialLickEnv1D:
    """Create a tiny deterministic cue burst on steps 1, 2, and 3."""
    env = main.SocialLickEnv1D(p_detect_per_obs=detect_probability)
    env.t = 0
    env.bout_id_at_step[:] = -1
    env.bout_id_at_step[1:4] = 0
    env.n_bouts = 1
    env.last_lick_step = None
    env.last_completed_bout_id = None
    env.window_start_step = None
    env.detected_bout_id = None
    env.rewarded_bout_ids = set()
    env.asocial_hit_bout_ids = set()
    env.learner_pos = env.learner_food_pos
    return env


class SocialLickEnvironmentTests(unittest.TestCase):
    def test_reward_window_opens_only_after_cue_ends(self):
        env = deterministic_single_bout_env(detect_probability=1.0)

        _, observation_reward, _, observation_info = env.step(main.A_OBS)
        _, during_cue_reward, _, during_cue_info = env.step(main.A_STAY)
        _, _, _, final_cue_info = env.step(main.A_STAY)
        _, after_cue_reward, _, after_cue_info = env.step(main.A_EAT)

        self.assertEqual(observation_info["window_open"], 0)
        self.assertEqual(during_cue_info["window_open"], 0)
        self.assertEqual(final_cue_info["window_open"], 0)
        self.assertEqual(after_cue_info["window_open"], 1)
        self.assertEqual(after_cue_info["eat_valid"], 1)
        self.assertAlmostEqual(observation_reward, 0.002)
        self.assertAlmostEqual(after_cue_reward, 0.995)

    def test_failed_observation_does_not_receive_attention_bonus(self):
        env = deterministic_single_bout_env(detect_probability=0.0)
        _, reward, _, info = env.step(main.A_OBS)

        self.assertEqual(info["seen_lick"], 0)
        self.assertAlmostEqual(reward, -env.observe_cost)

    def test_asocial_window_hit_is_recorded_without_social_registration(self):
        env = deterministic_single_bout_env(detect_probability=0.0)
        env.step(main.A_STAY)
        env.step(main.A_STAY)
        env.step(main.A_STAY)
        _, _, _, info = env.step(main.A_EAT)

        self.assertEqual(info["window_open"], 1)
        self.assertEqual(info["asocial_window_hit"], 1)
        self.assertEqual(info["eat_valid"], 0)

    def test_missing_values_remain_missing_in_ema(self):
        y = main.ema_1d(np.array([np.nan, np.nan, 1.0, np.nan], dtype=np.float32))

        self.assertTrue(np.isnan(y[0]))
        self.assertTrue(np.isnan(y[1]))
        self.assertAlmostEqual(float(y[2]), 1.0)
        self.assertAlmostEqual(float(y[3]), 1.0)

    def test_early_stop_tail_is_not_forward_filled(self):
        logs = [{"metric": 0.25}, {"metric": 0.5}]
        values = main.logs_to_array_with_ffill(logs, "metric", target_len=4)

        np.testing.assert_allclose(values[:2], [0.25, 0.5])
        self.assertTrue(np.isnan(values[2]))
        self.assertTrue(np.isnan(values[3]))

    def test_familiarity_is_visible_by_default_and_can_be_hidden_for_legacy_runs(self):
        modern_env = main.SocialLickEnv1D()
        legacy_env = main.SocialLickEnv1D(include_familiarity_state=False)

        self.assertEqual(modern_env.state_dim, 7)
        self.assertEqual(len(modern_env.reset()), 7)
        self.assertEqual(legacy_env.state_dim, 6)
        self.assertEqual(len(legacy_env.reset()), 6)


if __name__ == "__main__":
    unittest.main()
