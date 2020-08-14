import unittest
from beer_game.envs.bg_env import BeerGame
from action_policies import AgentSimulator

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

        self.assertEqual(total_cost, 1428.5)


if __name__=='__main__':
    unittest.main()
