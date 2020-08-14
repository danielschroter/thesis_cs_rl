from beer_game.envs.bg_env_cont import BeerGame
from action_policies import AgentSimulator

if __name__ == "__main__":
    env = BeerGame(demand_dist="classical", n_observed_periods=1, n_turns_per_game=20, discrete=False)
    obs = env.reset()
    env.render()
    done = False
    agent_sim = AgentSimulator(policy="random")
    returns = []
    for i in range(1):

        while not done:
            # env.render()

            actions = agent_sim.get_other_actions(obs, env.demand_dist, env.turn, env.orders,
                                                  env.demands, env.shipments, original=True, discrete=False)
            #env.render()
            #actions = agent_sim.get_other_actions_2("sterman", obs, env.demand_dist, env.turn, env.orders,
            #                                      env.demands, env.shipments)
            obs, rew, done, _ = env.step(actions)
            env.render()
        returns.append(sum(env.cum_holding_cost)+sum(env.cum_stockout_cost))
        obs, done = env.reset(), False
        agent_sim.reset()

    print(returns)






