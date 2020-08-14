import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
from beer_game.envs.bg_env import BeerGame
from action_policies import AgentSimulator
from action_policies import calculate_feedback
import pandas as pd
import matplotlib.pyplot as plt

global logits_net

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1], act())]
    return nn.Sequential(*layers)


def get_policy(obs):
    logits = logits_net(obs)
    return Categorical(logits=logits)


# make action selection function
def get_action(obs):
    return get_policy(obs).sample().item()


# make loss function, whose gradient, for the right data is policy gradient
def compute_loss(obs, act, weights):
    logp = get_policy(obs).log_prob(act)
    return -(logp * weights).mean()


def train(env_name="BeerGame-v0", policy_others="sterman", hidden_sizes=[32], lr=1e-2, epochs=1000, batch_size=5000,
          render=False, continuation=False, PATH=None, n_observed_periods=1):
    agent = 1
    action_means = []
    return_means = []
    epoch = 0
    agent_sim = AgentSimulator()

    env = BeerGame("classical", n_observed_periods=n_observed_periods, n_turns_per_game=20)
    start_state = env.reset()
    env.render()
    done = False

    obs_dim = env.observation_space.shape[1]
    n_acts = env.action_space.nvec[agent]

    global logits_net
    logits_net = mlp(sizes=[obs_dim] + hidden_sizes + [n_acts])

    # make optimizer
    optimizer = Adam(logits_net.parameters(), lr=lr)

    if continuation:
        checkpoint = torch.load(PATH)
        logits_net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        action_means = checkpoint['action_means']
        return_means = checkpoint['return_means']
        logits_net.train()

    # make function to compute action distribution:

    # for training policy
    def train_one_epoch():
        # make some empty lists for logging
        batch_obs = []
        batch_acts = []
        batch_weights = []
        batch_rets = []

        # reset episode specific variables
        obs_total = env.reset()
        obs = obs_total[agent].flatten()
        done = False
        ep_rews = []
        ep_rews_total = []

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy
        while True:
            # rendering
            if (not finished_rendering_this_epoch) and render:
                env.render()

            batch_obs.append(obs.copy())

            # act in environment
            act = get_action(torch.as_tensor(obs, dtype=torch.float32))
            actions = agent_sim.get_other_actions(policy_others, obs_total, env.demand_dist, env.turn, env.orders,
                                                  env.demands, env.shipments)
            actions[agent] = act
            obs_total, rew_total, done, _ = env.step(actions)
            obs, rew = obs_total[agent].flatten(), rew_total[agent]

            batch_acts.append(act)
            ep_rews.append(rew)
            ep_rews_total.append(rew_total)

            if done:
                # ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                ep_ret = sum([sum(elem) for elem in ep_rews_total])

                batch_rets.append(ep_ret)
                feedback = calculate_feedback(ep_rews_total, agent, 10)

                # batch_weights += [ep_ret]*ep_len
                batch_weights += feedback

                obs_total, done, ep_rews, ep_rews_total = env.reset(), False, [], [],
                agent_sim.reset()
                obs = obs_total[agent].flatten()
                finished_rendering_this_epoch = True

                # end experience loope if we have enough of it
                if len(batch_obs) > batch_size:
                    break
        # take a single policy gradient update step
        optimizer.zero_grad()
        batch_loss = compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                                  act=torch.as_tensor(batch_acts, dtype=torch.int32),
                                  weights=torch.as_tensor(batch_weights, dtype=torch.float32))

        batch_loss.backward()
        optimizer.step()
        return batch_loss, batch_rets, batch_acts

    # training loop

    # Create Plots
    plt.ion()
    f = plt.figure(0)
    ax1 = f.add_subplot(2, 1, 1)
    ax2 = f.add_subplot(2, 1, 2)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("Mean Orders in epoch")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("Mean Cost in epoch")

    for i in range(epoch, epochs):
        batch_loss, batch_rets, batch_acts = train_one_epoch()
        action_means.append(sum(batch_acts) / len(batch_acts))
        return_means.append(sum(batch_rets) / len(batch_rets))

        df_1 = pd.Series(action_means)
        df_2 = pd.Series(return_means)

        ax1.plot(df_1)
        ax2.plot(df_2)
        plt.pause(0.0001)

        print('epoch: %3d \t loss: %.3f \t return: %.3f' %
              (i, batch_loss, np.mean(batch_rets)))

        if (i + 1) % 10 == 0:
            # PATH = "checkpoint.pt"
            torch.save({
                'epoch': i,
                'policy_others' : policy_others,
                'model_state_dict': logits_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': batch_loss,
                'action_means': action_means,
                'return_means': return_means
            }, PATH)

    f.show()


def evaluate(PATH=None, n_games=10, hidden_sizes=[32], policy_others="sterman", render=False, epochs=10,
             n_observed_periods=1):
    agent = 1
    batch_actions = []
    batch_returns = []
    epoch = 0
    agent_sim = AgentSimulator()

    env = BeerGame("classical", n_observed_periods=n_observed_periods, n_turns_per_game=20)
    start_state = env.reset()
    env.render()
    done = False

    obs_dim = env.observation_space.shape[1]
    n_acts = env.action_space.nvec[agent]

    global logits_net

    logits_net = mlp(sizes=[obs_dim] + hidden_sizes + [n_acts])
    if PATH:
        checkpoint = torch.load(PATH)
        logits_net.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        # action_means = checkpoint['action_means']
        # return_means = checkpoint['return_means']
    logits_net.eval()

    # for training policy
    def eval_one_round():
        # make some empty lists for logging
        batch_obs = []
        batch_acts = []
        batch_weights = []
        batch_rets = []

        # reset episode specific variables
        obs_total = env.reset()
        obs = obs_total[agent].flatten()
        done = False
        ep_rews = []
        ep_rews_total = []

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy
        while True:
            # rendering
            if (not finished_rendering_this_epoch) and render:
                env.render()

            batch_obs.append(obs.copy())

            # act in environment
            with torch.no_grad():
                act = get_action(torch.as_tensor(obs, dtype=torch.float32))

            actions = agent_sim.get_other_actions(policy_others, obs_total, env.demand_dist, env.turn, env.orders,
                                                  env.demands, env.shipments)
            # actions_2 = agent_sim.get_other_actions("sterman", obs_total, env.demand_dist, env.turn, env.orders, env.demands, env.shipments)
            actions[agent] = act
            obs_total, rew_total, done, _ = env.step(actions)
            obs, rew = obs_total[agent].flatten(), rew_total[agent]

            batch_acts.append(act)
            ep_rews.append(rew)
            ep_rews_total.append(rew_total)

            if done:
                # ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                ep_ret = sum([sum(elem) for elem in ep_rews_total])

                batch_rets.append(ep_ret)
                # feedback = calculate_feedback(ep_rews_total, agent, 10)

                # batch_weights += [ep_ret]*ep_len
                # batch_weights += feedback

                obs_total, done, ep_rews, ep_rews_total = env.reset(), False, [], []
                agent_sim.reset()

                obs = obs_total[agent].flatten()
                finished_rendering_this_epoch = True

                # end experience loope if we have enough of it
                # if len(batch_obs) > batch_size:
                break

        # take a single policy gradient update step

        return batch_rets, batch_acts

    # Create Plots
    plt.ion()
    f = plt.figure(1)
    ax1 = f.add_subplot(2, 1, 1)
    ax2 = f.add_subplot(2, 1, 2)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("Cost in game")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("Cost in game")

    for i in range(n_games):
        ret, batch_acts = eval_one_round()
        batch_actions.append(batch_acts)
        batch_returns.append(ret[0])

        # df_1 = pd.Series(batch_actions)
        df_2 = pd.Series(batch_returns)

        # ax1.plot(df_1)
        ax2.plot(df_2)
        plt.pause(0.0001)

        print(f'epoch: {i} \t return: {ret} \t orders: {batch_acts}')

    f.show()

    return batch_returns, batch_actions


if __name__ == "__main__":
    n_observed_periods = 5
    returns = []
    actions = []

import os
from datetime import datetime

dir = os.path.dirname(__file__)
date = str(datetime.date(datetime.now()))
policy_others = "sterman"
PATH = os.path.join(dir, 'logs', 'checkpoint_' + date + '_' + policy_others + '.pt')

for i in range(20, 30):
    if i == 1:
        continuation = False
    else:
        continuation = True
    train(continuation=continuation, policy_others="sterman", PATH=PATH, epochs=i * 10, lr=1e-4, n_observed_periods=n_observed_periods)

    ret, acts = evaluate(PATH=PATH, n_observed_periods=n_observed_periods)
    returns.append(ret)
    actions.append(acts)

returns_flat = sum(returns, [])

means = [(sum(x) / len(x)) for x in returns]
print(f"mean of 50 test = {sum(means) / len(means)}")

plt.ion()
f = plt.figure(3)
# ax1 = f.add_subplot(2,1,1)
# ax2 = f.add_subplot(2,1,2)
plt.xlabel("game")
plt.ylabel("Cost in game")
plt.plot(returns_flat)
f.show()

PATH = os.path.join(dir, 'logs', 'checkpoint_' + date + '_' + policy_others + '_eval.pt')
torch.save({
    'policy_others' : policy_others,
    'returns': returns,
    'actions': actions
}, PATH)
