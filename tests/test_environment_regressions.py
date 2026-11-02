import unittest

import numpy as np

from peim_core import LegacyPeimEnv as PeimEnv


class EnvironmentRegressionTests(unittest.TestCase):
    def make_env(self, **kwargs):
        options = dict(num_markets=2, time_to_live=4,
                       market_lambdas=np.zeros((2, 1)), production_mu=0,
                       production_sigma=0, c_production=0, c_stock_keeping=0,
                       c_transportation=0, c_waste=3, seed=42)
        options.update(kwargs)
        env = PeimEnv(**options)
        env.reset(seed=42)
        return env

    def test_default_lead_time_is_usable_and_invalid_values_fail_early(self):
        env = self.make_env()
        self.assertEqual(env.lead_time, 1)
        env.step(np.zeros((2, 4), dtype=int))
        for lead_time in [0, -1, 1.5, True]:
            with self.subTest(lead_time=lead_time), self.assertRaises(ValueError):
                self.make_env(lead_time=lead_time)

    def test_random_issuance_serves_all_feasible_demand(self):
        env = self.make_env()
        env.state[1] = [1, 1, 0, 0]
        env.state[2] = [2, 3, 4, 1]
        for _ in range(50):
            sold = env.distribute_demand(np.array([2, 7]))
            np.testing.assert_array_equal(sold.sum(axis=1), [2, 7])
            self.assertTrue(np.all(sold >= 0))
            self.assertTrue(np.all(sold <= env.state[1:]))

    def test_two_period_lead_time_and_arrival_snapshot(self):
        env = self.make_env(lead_time=2)
        env.state[0, 0] = 3
        action = np.zeros((2, 4), dtype=int)
        action[0, 0] = 3
        _, _, _, first = env.step(action)
        _, _, _, second = env.step(np.zeros_like(action))
        self.assertEqual(env.state[1:].sum(), 0)
        self.assertEqual(first['currently_arriving_action'].sum(), 0)
        self.assertEqual(second['currently_arriving_action'].sum(), 0)
        _, _, _, third = env.step(np.zeros_like(action))
        np.testing.assert_array_equal(third['currently_arriving_action'][0], [0, 0, 3, 0])
        np.testing.assert_array_equal(env.state[1], [0, 0, 0, 3])
        env.step(np.zeros_like(action))
        np.testing.assert_array_equal(third['currently_arriving_action'][0], [0, 0, 3, 0])

    def test_waste_charges_cover_on_hand_dispatched_and_in_transit_goods(self):
        env = self.make_env(lead_time=2)
        env.state[0, -1] = 2
        env.state[1, -1] = 3
        env.allocation_buffer[1, -1, 0] = 4
        action = np.zeros((2, 4), dtype=int)
        action[0, -1] = 1
        _, reward, _, info = env.step(action)
        self.assertEqual(info['location_waste_t'].sum(), 4)
        self.assertEqual(info['allocation_waste_t'].sum(), 5)
        self.assertEqual(info['reward_components']['waste_units'], 9)
        self.assertEqual(reward, -27)
        self.assertEqual(env.allocation_buffer.sum(), 0)

    def test_reset_seed_reproduces_entire_trajectory(self):
        env = self.make_env(market_lambdas=np.array([[2.], [3.]]),
                            production_mu=10, production_sigma=2)
        def rollout():
            env.reset(seed=123)
            records = []
            for _ in range(12):
                state, reward, done, info = env.step(np.ones((2, 4), dtype=int)*4)
                records.append((state.copy(), reward, done, info['action'].copy(),
                                info['market_demands'].copy(), info['selloff'].copy()))
            return records
        for left, right in zip(rollout(), rollout()):
            for a, b in zip(left, right):
                np.testing.assert_array_equal(a, b)

    def test_demand_and_production_streams_do_not_depend_on_policy_or_issuance(self):
        a = self.make_env(issuance_rule='random', market_lambdas=np.ones((2, 1)),
                          production_mu=10, production_sigma=2)
        b = self.make_env(issuance_rule='fifo', market_lambdas=np.ones((2, 1)),
                          production_mu=10, production_sigma=2)
        for _ in range(12):
            _, _, _, ai = a.step(np.ones((2, 4), dtype=int)*20)
            _, _, _, bi = b.step(np.zeros((2, 4), dtype=int))
            np.testing.assert_array_equal(ai['market_demands'], bi['market_demands'])
            self.assertEqual(ai['inflow_t'], bi['inflow_t'])

    def test_holding_cost_uses_post_sales_on_hand_inventory_before_expiry(self):
        env = self.make_env(c_stock_keeping=0.5, c_waste=0)
        env.state[0] = [2, 0, 0, 1]
        env.state[1, -1] = 3
        env.allocation_buffer[1, 1, 0] = 4  # Due stock arrives before holding cost.
        _, reward, _, info = env.step(np.zeros((2, 4), dtype=int))
        self.assertEqual(info['reward_components']['stock_keeping_cost'], 5)
        self.assertEqual(reward, -5)


if __name__ == '__main__':
    unittest.main()
