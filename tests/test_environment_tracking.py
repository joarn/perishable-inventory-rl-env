import unittest

import numpy as np

from peim_core import LegacyPeimEnv as PeimEnv


class EnvironmentTrackingTests(unittest.TestCase):
    def make_env(self, issuance_rule="random", num_markets=1):
        env = PeimEnv(
            num_markets=num_markets,
            time_to_live=3,
            market_lambdas=np.ones((num_markets, 1)),
            lead_time=1,
            issuance_rule=issuance_rule,
        )
        env.reset()
        return env

    def test_fifo_and_lifo_use_opposite_age_orders(self):
        fifo = self.make_env("fifo")
        lifo = self.make_env("lifo")
        fifo.state[1] = np.array([2, 3, 4])
        lifo.state[1] = np.array([2, 3, 4])

        np.testing.assert_array_equal(
            fifo.distribute_demand(np.array([5]))[0], np.array([0, 1, 4])
        )
        np.testing.assert_array_equal(
            lifo.distribute_demand(np.array([5]))[0], np.array([2, 3, 0])
        )

    def test_flattened_sac_action_and_matrix_action_are_both_supported(self):
        flat_env = self.make_env("fifo")
        matrix_env = self.make_env("fifo")
        flat_env.reset(seed=12)
        matrix_env.reset(seed=12)
        action = np.zeros((1, 3), dtype=int)

        flat_state, flat_reward, _, flat_info = flat_env.step(action.flatten())
        matrix_state, matrix_reward, _, matrix_info = matrix_env.step(action)

        np.testing.assert_array_equal(flat_state, matrix_state)
        np.testing.assert_array_equal(
            flat_info["market_demands"], matrix_info["market_demands"]
        )
        self.assertEqual(flat_reward, matrix_reward)

    def test_info_contains_reward_and_action_decompositions(self):
        env = PeimEnv(
            num_markets=1,
            time_to_live=3,
            market_lambdas=np.zeros((1, 1)),
            lead_time=1,
            production_mu=0,
            production_sigma=0,
            c_production=0,
            c_stock_keeping=0,
            c_transportation=0,
            c_waste=2,
            p_selling=0,
            seed=4,
        )
        env.reset(seed=4)
        env.state[1, -1] = 3

        _, reward, _, info = env.step(np.zeros((1, 3), dtype=int))

        self.assertEqual(reward, -6)
        self.assertEqual(info["reward_components"]["waste_units"], 3)
        self.assertEqual(info["reward_components"]["waste_cost"], 6)
        self.assertIn("requested_action", info)
        self.assertIn("valid_action", info)

    def test_reset_exposes_opening_production_to_the_policy(self):
        env = PeimEnv(
            num_markets=1,
            time_to_live=3,
            market_lambdas=np.zeros((1, 1)),
            lead_time=1,
            production_mu=5,
            production_sigma=0,
            seed=8,
        )

        state = env.reset(seed=8)

        np.testing.assert_array_equal(state[0], np.array([5, 0, 0]))
        self.assertEqual(env.current_inflow, 5)

    def test_action_is_checked_before_next_period_production(self):
        env = PeimEnv(
            num_markets=1,
            time_to_live=3,
            market_lambdas=np.zeros((1, 1)),
            lead_time=1,
            production_mu=0,
            production_sigma=0,
            seed=9,
        )
        observed_state = env.reset(seed=9).copy()
        env.production_mu = 8

        next_state, _, _, info = env.step(np.array([[4, 0, 0]]))

        np.testing.assert_array_equal(observed_state[0], np.array([0, 0, 0]))
        np.testing.assert_array_equal(info["action"], np.zeros((1, 3)))
        self.assertFalse(info["valid_action"])
        self.assertEqual(info["inflow_t"], 0)
        np.testing.assert_array_equal(next_state[0], np.array([8, 0, 0]))

    def test_dispatched_goods_arrive_after_lead_time_and_age(self):
        env = PeimEnv(
            num_markets=1,
            time_to_live=3,
            market_lambdas=np.zeros((1, 1)),
            lead_time=1,
            production_mu=0,
            production_sigma=0,
            seed=10,
        )
        env.reset(seed=10)
        env.state[0] = np.array([4, 0, 0])

        first_state, _, _, first_info = env.step(np.array([[3, 0, 0]]))

        np.testing.assert_array_equal(
            first_info["currently_arriving_action"], np.zeros((1, 3))
        )
        np.testing.assert_array_equal(first_state[0], np.array([0, 1, 0]))
        np.testing.assert_array_equal(
            env.allocation_buffer[:, :, 0], np.array([[0, 3, 0]])
        )

        second_state, _, _, second_info = env.step(np.zeros((1, 3), dtype=int))

        np.testing.assert_array_equal(
            second_info["currently_arriving_action"], np.array([[0, 3, 0]])
        )
        np.testing.assert_array_equal(second_state[1], np.array([0, 0, 3]))

    def test_infeasible_action_is_projected_proportionally(self):
        env = self.make_env("fifo", num_markets=3)
        env.state[0] = np.array([4, 0, 0])
        requested = np.array(
            [
                [1, 0, 0],
                [2, 0, 0],
                [3, 0, 0],
            ]
        )

        projected, valid = env.adapt_action(requested)

        self.assertFalse(valid)
        np.testing.assert_array_equal(projected[:, 0], np.array([1, 1, 2]))
        self.assertEqual(int(projected[:, 0].sum()), 4)
        self.assertTrue(np.all(projected <= requested))

    def test_proportional_projection_is_market_permutation_equivariant(self):
        env_a = self.make_env("fifo", num_markets=3)
        env_b = self.make_env("fifo", num_markets=3)
        env_a.reset(seed=27)
        env_b.reset(seed=27)
        env_a.state[0] = np.array([4, 0, 0])
        env_b.state[0] = np.array([4, 0, 0])
        requested = np.array(
            [
                [1, 0, 0],
                [2, 0, 0],
                [3, 0, 0],
            ]
        )
        permutation = np.array([2, 0, 1])

        projected_a, _ = env_a.adapt_action(requested)
        projected_b, _ = env_b.adapt_action(requested[permutation])

        inverse = np.argsort(permutation)
        np.testing.assert_array_equal(projected_a, projected_b[inverse])

    def test_equal_projection_ties_do_not_always_favor_first_market(self):
        winners = set()
        requested = np.ones((3, 3), dtype=int)
        requested[:, 1:] = 0
        for seed in range(30):
            env = self.make_env("fifo", num_markets=3)
            env.reset(seed=seed)
            env.state[0] = np.array([1, 0, 0])

            projected, _ = env.adapt_action(requested)

            self.assertEqual(int(projected[:, 0].sum()), 1)
            winners.add(int(np.argmax(projected[:, 0])))

        self.assertEqual(winners, {0, 1, 2})


if __name__ == "__main__":
    unittest.main()
