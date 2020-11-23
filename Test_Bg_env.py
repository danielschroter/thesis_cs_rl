import unittest
from beer_game.envs.beergame import BeerGame
from Agent_Simulator import AgentSimulator
from utils import calculate_feedback, calculate_expectedReturn

class TestBgEnv(unittest.TestCase):

    def test_base_stock(self):
        env = BeerGame(demand_dist="classical", n_observed_periods=1, n_turns_per_game=20)
        obs = env.reset()
        done = False
        agent_sim = AgentSimulator(policy="base_stock")
        while not done:

            actions = agent_sim.get_other_actions(obs, env.demand_dist, env.turn, env.orders,
                                                  env.demands, env.shipments, original=True)
            obs, rew, done, _ = env.step(actions)
        total_cost = sum(env.cum_holding_cost)+sum(env.cum_stockout_cost)

        self.assertEqual(total_cost, 180)

    def test_sterman(self):
        env = BeerGame(demand_dist="classical", n_observed_periods=1, n_turns_per_game=36)
        obs = env.reset()
        done = False
        agent_sim = AgentSimulator(policy="sterman")
        while not done:

            actions = agent_sim.get_other_actions(obs, env.demand_dist, env.turn, env.orders,
                                                  env.demands, env.shipments, original=True)
            obs, rew, done, _ = env.step(actions)
        total_cost = sum(env.cum_holding_cost)+sum(env.cum_stockout_cost)

        self.assertEqual(total_cost, 2049)

    def test_feedback_calculation(self):
        feedback = calculate_feedback([[1, 2, 3, 4], [5, 6, 7, 8]], 1, 9)
        self.assertListEqual(feedback, [44.0, 48.0])

    def test_calc_exp_ret(self):
        weights = calculate_expectedReturn([1,2,3,4,5], 0.5)
        self.assertListEqual(weights, [3.5625, 5.125, 6.25, 6.5, 5.])

if __name__=='__main__':
    unittest.main()
