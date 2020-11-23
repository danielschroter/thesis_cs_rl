import copy
from algorithms.pg.ddpg.ddpg_agent import Agent
from utils import plotLearning
from utils import total_feedback
from algorithms.pg.reinforce.reinforce_main import *
from beer_game.envs.beergame import BeerGame
import random


def evaluate(env, agent, agent_sim, controlled_agent, n_games=50):
    batch_actions = []
    batch_returns = []
    epoch = 0

    start_state = env.reset()
    #env.render()
    done = False
    agent.load_models()

    for i in range(n_games):
        ret, batch_acts = run_one_time(env, agent, agent_sim, controlled_agent=controlled_agent, mode="eval")
        batch_actions.append(batch_acts)
        batch_returns.append(ret[0])

        print(f'epoch: {i} \t return: {ret} \t orders: {batch_acts}')

    print(f'mean return of {n_games} games played is {(sum(batch_returns)/len(batch_returns))}')

    return batch_returns, batch_actions


N_OBSERVED_PERIODS = 10
LR = 1e-4
CONTROLLED_AGENT = 1
N_TURNS_PER_GAME = 36
POLICY_OTHERS = "base_stock"
DEMAND_DIST = "classical"
HIDDEN_SIZES = [32, 32]
EPOCHS = 1200
ALGORITHM = "DDPG"
MODE = 'continuous'
seed = 0
DISCRETE = False

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

env = BeerGame(demand_dist=DEMAND_DIST, n_observed_periods=N_OBSERVED_PERIODS, n_turns_per_game=N_TURNS_PER_GAME,
               discrete=DISCRETE)
agent_sim = AgentSimulator(policy=POLICY_OTHERS, discrete=DISCRETE)

obs_dim = env.observation_space.shape[1]
n_acts = 1 # for one action

agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[obs_dim], tau=0.001, env=env,
              batch_size=32, layer1_size=400, layer2_size=300, n_actions=n_acts)
agent.load_models()


controlled_agent = 1
score_history = []
for i in range(1500):
    done = False
    score = 0
    local_cost = 0
    obs_total = env.reset()
    obs = obs_total[controlled_agent].flatten()
    obs = np.array(obs, dtype=np.float32)
    g_states = [obs]
    g_rewards = []
    g_rewards_total = []
    g_done = []
    g_actions = []

    while not done:
        if i<1200:
            act = np.array([np.random.uniform(low=0, high=16)], dtype=np.float32)
        else:
            act = agent.get_action(obs)

        actions = agent_sim.get_other_actions(copy.deepcopy(obs_total), env.demand_dist, env.turn, env.orders,
                                              env.demands, env.shipments)
        actions[controlled_agent] = act[0]
        obs_total, rew_total, done, info = env.step(actions)
        new_state, reward = obs_total[controlled_agent].flatten(), rew_total[controlled_agent]

        g_actions.append(act)
        g_states.append(new_state)
        g_rewards.append(reward)
        g_rewards_total.append(rew_total)
        g_done.append(done)

        # agent.remember(obs, act, rew_total, new_state, int(done))

        # we learn after each action, because it is a temporal difference method
        # instead of a monte carlo method where we learn at the end of an episode

        # A small warm up phase to ensure that there are already some samples in the replay buffer

        if i>1200:
            agent.learn()
        score += sum(rew_total)
        local_cost += reward

        obs = new_state

    fb = total_feedback(g_rewards_total, controlled_agent, 50)
    fb_mean = np.mean(fb[controlled_agent])
    feedback = calculate_feedback(g_rewards_total, controlled_agent)
    agent_sim.reset()
#    feedback = fb[controlled_agent]

    # remember function is moved outside of the loop, so we can calculate the feedback scheme according to the DQN Paper
    for t in range(len(g_rewards)):
        agent.remember(state=g_states[t], action=g_actions[t], reward=feedback[t], new_state=g_states[t + 1],
                       done=int(g_done[t]))

    #for rew,a, f in zip(g_rewards,g_actions, feedback):
    #    print(f'rew: {rew}, action: {a}, feeback: {f}')

    score_history.append(score)
    print('episode ', i, 'feedback: %.4f' % (sum(feedback)),  'score %.2f' % score, 'local cost %.2f' % local_cost,
          ' trailing 100 game average %.2f' % np.mean(score_history[-100:]))
    if i % 500 == 0:
        agent.save_models()

import os
dir = os.path.dirname(__file__)
PATH = os.path.join(dir, 'logs', '_'+DEMAND_DIST+'_'+POLICY_OTHERS)

torch.save({
    'Mode' : MODE,
    'Algorithm': ALGORITHM,
    'Demand_Distribution' : DEMAND_DIST,
    'Policy_Others' : POLICY_OTHERS,
    'Controlled_Agent' : CONTROLLED_AGENT,
    'N_Turns_Per_Game' : N_TURNS_PER_GAME,
    'N_Observed_Periods' : N_OBSERVED_PERIODS,
    'score_history': score_history,
}, (PATH + '_summary.pt'))
plotLearning(score_history, PATH+'_plot.png', window=100)

evaluate(env, agent, agent_sim, n_games=1000, controlled_agent=CONTROLLED_AGENT)

print("Done")

