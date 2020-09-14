import numpy as np
import gym
from algorithms.pg.reinforce_agent import PolicyGradientAgent
import matplotlib.pyplot as plt
from utils import plotLearning
from beer_game.envs.bg_env_cont_2 import BeerGame
from action_policies import AgentSimulator, calculate_feedback
from gym import wrappers

if __name__ == '__main__':

    N_OBSERVED_PERIODS = 10
    LR = 1e-4
    CONTROLLED_AGENT = 1
    N_TURNS_PER_GAME = 36
    POLICY_OTHERS = "base_stock"
    DEMAND_DIST = "classical"
    HIDDEN_SIZES = [32, 32]
    EPOCHS = 1200
    ALGORITHM = "policy_gradient"
    MODE = 'discrete'
    DISCRETE = True

    env = BeerGame(demand_dist=DEMAND_DIST, n_observed_periods=N_OBSERVED_PERIODS, n_turns_per_game=N_TURNS_PER_GAME,
               discrete=DISCRETE)
    agent_sim = AgentSimulator(policy=POLICY_OTHERS, discrete=DISCRETE)


    obs_dim = env.observation_space.shape[1]
    n_acts = env.action_space.nvec[CONTROLLED_AGENT]

    agent = PolicyGradientAgent(ALPHA=0.0001, input_dims=[obs_dim], GAMMA=0.99,
                                n_actions=n_acts, layer1_size=128, layer2_size=128)

    score_history = []
    score = 0
    num_episodes = 2500
    #env = wrappers.Monitor(env, "tmp/lunar-lander",
    #                        video_callable=lambda episode_id: True, force=True)


    for i in range(num_episodes):
        print('episode: ', i,'score: ', score)
        done = False
        score = 0
        obs_total = env.reset()
        obs = obs_total[CONTROLLED_AGENT].flatten()
        obs = np.array(obs, dtype=np.float32)
        total_reward = []
        while not done:
            act = agent.choose_action(obs)
            actions = agent_sim.get_other_actions(obs_total, env.demand_dist, env.turn, env.orders,
                                              env.demands, env.shipments)
            actions[CONTROLLED_AGENT] = act
            observation_, reward, done, info = env.step(actions)
            new_state, rew = observation_[CONTROLLED_AGENT].flatten(), reward[CONTROLLED_AGENT]
            #agent.store_rewards(rew)
            total_reward.append(reward)
            obs = new_state
            score += sum(reward)
        score_history.append(score)
        fb = calculate_feedback(total_reward, CONTROLLED_AGENT, 20)
        for elem in fb:
            agent.store_rewards(elem)
        agent.learn()
        #agent.save_checkpoint()
    filename = 'lunar-lander-alpha001-128x128fc-newG.png'
    plotLearning(score_history, filename=filename, window=25)
