from beer_game.envs.bg_env import BeerGame
from action_policies import AgentSimulator
import numpy as np
import random

if __name__ == "__main__":


    N_TURNS_PER_GAME = 36
    DEMAND_DIST = "uniform_0_8"
    POLICY_OTHERS = "base_stock"
    CONTROLLED_AGENT = 1
    POLICY_AGENT = "base_stock"
    DISCRETE = True
    seed = 123

    random.seed(seed)
    np.random.seed(seed)

    env = BeerGame(demand_dist=DEMAND_DIST, n_observed_periods=1, n_turns_per_game=N_TURNS_PER_GAME)
    obs = env.reset()
    done = False
    agent_sim_others = AgentSimulator(policy=POLICY_OTHERS, discrete=DISCRETE)
    agent_sim_agent = AgentSimulator(policy=POLICY_AGENT, discrete=DISCRETE)


    returns = []
    for i in range(1000):

        while not done:

            actions = agent_sim_others.get_other_actions(obs, env.demand_dist, env.turn, env.orders,
                                                  env.demands, env.shipments, original=True)
            actions_2 = agent_sim_agent.get_other_actions(obs, env.demand_dist, env.turn, env.orders,
                                                  env.demands, env.shipments, original=True)
            actions[CONTROLLED_AGENT] = actions_2[CONTROLLED_AGENT]
            obs, rew, done, _ = env.step(actions)
            #env.render()
        returns.append(sum(env.cum_holding_cost)+sum(env.cum_stockout_cost))
        obs, done = env.reset(), False
        agent_sim_others.reset()
        agent_sim_agent.reset()

    print(f" The average mean is {np.mean(returns)}")






