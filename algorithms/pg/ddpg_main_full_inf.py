import gym
import numpy as np
import copy
from algorithms.pg.ddpg_agent import Agent
from algorithms.pg.utils import plotLearning
from action_policies import calculate_feedback, total_feedback, AgentSimulator
from beer_game.envs.bg_env_cont_2 import BeerGame

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
DISCRETE = False

env = BeerGame(demand_dist=DEMAND_DIST, n_observed_periods=N_OBSERVED_PERIODS, n_turns_per_game=N_TURNS_PER_GAME,
               discrete=DISCRETE)
agent_sim = AgentSimulator(policy=POLICY_OTHERS, discrete=DISCRETE)

# env = gym.make("LunarLanderContinuous-v2")
# obs_dim = 8
# n_acts = 2

obs_dim = env.observation_space.shape[1]
obs_dim = env.observation_space.shape[0]*obs_dim
n_acts = 1

agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[obs_dim], tau=0.001, env=env,
              batch_size=32, layer1_size=400, layer2_size=300, n_actions=n_acts)

# np.random.seed(0)

controlled_agent = 1
score_history = []
for i in range(10000):
    done = False
    score = 0
    local_cost = 0
    obs_total = env.reset()
    obs = obs_total.flatten()
    obs = np.array(obs, dtype=np.float32)

    g_states = [obs]
    g_rewards = []
    g_rewards_total = []
    g_done = []
    g_actions = []

    while not done:
        act = agent.choose_action(obs)

        actions = agent_sim.get_other_actions(copy.deepcopy(obs_total), env.demand_dist, env.turn, env.orders,
                                              env.demands, env.shipments)
        actions[controlled_agent] = act[0]
        obs_total, rew_total, done, info = env.step(actions)
        #new_state, reward = obs_total.flatten(), rew_total[controlled_agent]
        new_state, reward = obs_total.flatten(), sum(rew_total)

        g_actions.append(act)
        g_states.append(new_state)
        g_rewards.append(reward)
        g_rewards_total.append(rew_total)
        g_done.append(done)
        # reward = -reward

        # agent.remember(obs, act, rew_total, new_state, int(done))
        # we learn after each action, because it is a temporal difference method
        # instead of a monte carlo method where we learn at the end of an episode
        if i>20:
            agent.learn()
        score += sum(rew_total)
        local_cost += reward

        obs = new_state

    #fb = total_feedback(g_rewards_total, controlled_agent, 50)
    #fb_mean = np.mean(fb[controlled_agent])
    #feedback = fb[controlled_agent]
#    feedback = fb[controlled_agent]

    for t in range(len(g_rewards)):
        agent.remember(state=g_states[t], action=g_actions[t], reward=g_rewards[t], new_state=g_states[t + 1],
                       done=int(g_done[t]))

    #for rew,a, f in zip(g_rewards,g_actions, feedback):
    #    print(f'rew: {rew}, action: {a}, feeback: {f}')

    score_history.append(score)
    print('episode ', i,  'score %.2f' % score, 'local cost %.2f' % local_cost,
          ' trailing 100 game average %.2f' % np.mean(score_history[-100:]))
    if i % 500 == 0:
        agent.save_models()

filename = 'lunar-lander.png'
plotLearning(score_history, filename, window=100)
