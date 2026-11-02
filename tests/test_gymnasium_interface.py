import unittest
import warnings

import gymnasium as gym
from gymnasium.error import ResetNeeded
from gymnasium.utils.env_checker import check_env
import numpy as np

from peim_env import PeimEnv, LegacyPeimEnv


class GymnasiumInterfaceTests(unittest.TestCase):
    def config(self, **kwargs):
        settings = dict(num_markets=2, time_to_live=4,
                        market_lambdas=np.array([[1.], [2.]]), lead_time=1,
                        max_timesteps=10, production_mu=10, production_sigma=2)
        settings.update(kwargs)
        return settings

    def test_official_checker_for_each_issuance_rule(self):
        for rule in ['random', 'fifo', 'lifo']:
            with self.subTest(rule=rule):
                env = gym.make('PerishableInventory-v0', **self.config(issuance_rule=rule))
                with warnings.catch_warnings():
                    # Physical inventory has no fixed upper capacity.
                    warnings.filterwarnings('ignore', message='.*maximum value is infinity.*')
                    check_env(env.unwrapped)
                env.close()

    def test_observations_contain_pipeline_and_horizon_and_are_snapshots(self):
        env = PeimEnv(**self.config(production_sigma=0))
        first, _ = env.reset(seed=4)
        saved = {key: value.copy() for key, value in first.items()}
        action = np.zeros((2, 4), dtype=np.float32)
        action[0, 0] = 0.5
        second, _, terminated, truncated, _ = env.step(action)
        self.assertTrue(env.observation_space.contains(first))
        self.assertTrue(env.observation_space.contains(second))
        self.assertEqual(second['pipeline'].sum(), 5)
        self.assertEqual(second['remaining_steps'][0], 9)
        self.assertFalse(terminated or truncated)
        for key in saved:
            np.testing.assert_array_equal(first[key], saved[key])
        second['inventory'][:] = 999
        self.assertFalse(np.any(env.simulator.state == 999))

    def test_sampled_actions_are_accepted_and_terminal_horizon_is_distinct(self):
        env = PeimEnv(**self.config(max_timesteps=3))
        with self.assertRaises(ResetNeeded):
            env.step(env.action_space.sample())
        env.reset(seed=11)
        env.action_space.seed(12)
        for step in range(3):
            obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            self.assertTrue(env.observation_space.contains(obs))
            self.assertIsInstance(reward, float)
            self.assertEqual(terminated, step == 2)
            self.assertFalse(truncated)
        self.assertEqual(obs['remaining_steps'][0], 0)
        with self.assertRaises(ResetNeeded):
            env.step(env.action_space.sample())
        env.reset()
        env.step(env.action_space.sample())

    def test_external_time_limit_truncates_without_terminal_horizon(self):
        env = gym.wrappers.TimeLimit(PeimEnv(**self.config()), max_episode_steps=2)
        env.reset(seed=8)
        env.step(env.action_space.sample())
        obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(obs['remaining_steps'][0], 8)

    def test_invalid_actions_are_rejected_without_advancing(self):
        env = PeimEnv(**self.config())
        env.reset(seed=10)
        for action in [np.zeros(8), np.full((2, 4), -0.1),
                       np.full((2, 4), 1.1), np.full((2, 4), np.nan)]:
            with self.assertRaises(ValueError):
                env.step(action)
            self.assertEqual(env.simulator.current_timestep, 0)

    def test_action_conversion_and_dynamics_match_experiment_convention(self):
        for rule in ['random', 'fifo', 'lifo']:
            config = self.config(issuance_rule=rule, lead_time=2, production_sigma=0)
            env = PeimEnv(**config)
            legacy = LegacyPeimEnv(**config)
            env.reset(seed=21)
            legacy.reset(seed=21)
            rng = np.random.default_rng(22)
            for _ in range(10):
                raw = rng.uniform(0, 1, (2, 4))
                scales = np.minimum(1., 1. / (raw.sum(axis=0) + 1e-8))
                requests = np.floor(raw * scales * np.floor(legacy.state[0])).astype(int)
                obs, reward, terminated, _, info = env.step(raw)
                old_obs, old_reward, done, old_info = legacy.step(requests)
                np.testing.assert_array_equal(obs['inventory'], old_obs)
                np.testing.assert_array_equal(obs['pipeline'], legacy.allocation_buffer)
                np.testing.assert_array_equal(info['action'], old_info['action'])
                self.assertEqual(reward, old_reward)
                self.assertEqual(terminated, done)

    def test_negative_production_draw_is_clamped_only_in_gymnasium_interface(self):
        config = self.config(production_mu=-5, production_sigma=0)
        env = PeimEnv(**config)
        obs, _ = env.reset(seed=1)
        self.assertEqual(obs['inventory'].sum(), 0)
        self.assertTrue(env.observation_space.contains(obs))
        obs, _, _, _, _ = env.step(np.zeros((2, 4)))
        self.assertTrue(env.observation_space.contains(obs))
        legacy = LegacyPeimEnv(**config)
        self.assertEqual(legacy.reset(seed=1)[0, 0], -5)

    def test_registered_env_and_flattened_observation_wrapper(self):
        env = gym.wrappers.FlattenObservation(gym.make(
            'PerishableInventory-v0', **self.config(), render_mode='ansi'))
        obs, _ = env.reset(seed=3)
        self.assertEqual(obs.ndim, 1)
        self.assertTrue(env.observation_space.contains(obs))
        obs, _, _, _, _ = env.step(env.action_space.sample())
        self.assertTrue(env.observation_space.contains(obs))
        self.assertIn('Inventory:', env.render())
        env.close()
        env.close()


if __name__ == '__main__':
    unittest.main()
