from algorithms.pg.ddpg_agent import Agent
import gym
import numpy as np
from algorithms.pg.utils import plotLearning

env = gym.make('LunarLanderContinuous-v2')
agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[8], tau=0.001, env=env,
              batch_size=64,  layer1_size=400, layer2_size=300, n_actions=2)

#agent.load_models()
np.random.seed(0)

score_history = []
for i in range(1000):
    obs = env.reset()
    done = False
    score = 0
    g_states = [obs]
    g_rewards = []
    #g_rewards_total = []
    g_done = []
    g_actions = []
    while not done:
        act = agent.choose_action(obs)
        new_state, reward, done, info = env.step(act)

        g_actions.append(act)
        g_states.append(new_state)
        g_rewards.append(reward)
        g_done.append(done)

        #agent.remember(obs, act, reward, new_state, int(done))
        agent.learn()
        score += reward
        obs = new_state
        env.render()

    for t in range(len(g_rewards)):
        agent.remember(state=g_states[t], action=g_actions[t], reward= g_rewards[t], new_state=g_states[t+1], done=int(g_done[t]))

    score_history.append(score)

    if i % 25 == 0:
        agent.save_models()

    print('episode ', i, 'score %.2f' % score,
          'trailing 100 games avg %.3f' % np.mean(score_history[-100:]))

filename = 'LunarLander-alpha000025-beta00025-400-300.png'
plotLearning(score_history, filename, window=100)
