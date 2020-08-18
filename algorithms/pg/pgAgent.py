import torch as T
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np
from beer_game.envs.bg_env import BeerGame
from action_policies import AgentSimulator
from action_policies import calculate_feedback
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
import gym


def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1], act())]
    return nn.Sequential(*layers)

obs_dim = 25
n_acts = 16
hidden_sizes = [32]
logits_net = mlp(sizes=[obs_dim] + hidden_sizes + [n_acts])


class GenericNetwork(nn.Module):
    """
    Linear NN with two hidden layers
    """
    def __init__(self, sizes, out_size, alpha = 0.0001,  activation=nn.Tanh, output_activation=nn.Identity):
        super(GenericNetwork, self).__init__()
        self.activation = nn.Tanh()
        self.output_activation = nn.Identity()
        self.out_size = out_size
        self.sizes = sizes
        self.hidden = nn.ModuleList()
        for j in range(len(sizes) - 1):
            self.hidden.append(nn.Linear(sizes[j], sizes[j + 1]))
        self.out = nn.Linear(sizes[-1], out_size)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        #self.device = T.device('cuda:0' if T.cuda.is_available() else 'cuda:1')
        #self.to(self.device)

    def forward(self, observation):
        #x = T.Tensor(observation).to(self.device)
        x = observation
        for layer in self.hidden:
            x = self.activation(layer(x))
        output = self.output_activation(self.out(x))
        return output


model = GenericNetwork(sizes=[obs_dim] + hidden_sizes, out_size = n_acts)


class PGAgent():
    def __init__(self, alpha, sizes, n_actions):
        # self.mlp = GenericNetwork(sizes, out_size=n_actions, alpha=alpha)
        self.mlp = mlp(sizes=sizes+[n_actions])
        self.optimizer = optim.Adam(self.mlp.parameters(), lr=alpha)
        self.log_probs = None
        self.controlled_agent = 1
        self.policy_others = "sterman"

    def choose_action(self, obs):
        logits = self.mlp.forward(obs)
        action_probs = T.distributions.Categorical(logits=logits)
        action = action_probs.sample()
        # self.log_probs = action_probs.log_prob(action)
        return action.item()


    def get_policy(self, obs):
        logits = self.mlp.forward(obs)
        return Categorical(logits=logits)

    # make action selection function
    def get_action(self, obs):
        return self.get_policy(obs).sample().item()


# make loss function, whose gradient, for the right data is policy gradient
    def compute_loss(self, obs, act, weights, n_traj):
        logp = self.get_policy(obs).log_prob(act)
        #return -(logp * weights).mean()
        return -(sum(logp * weights)/n_traj)

    def learn(self, obs, acts, weights, n_traj):
        # self.mlp.optimizer.zero_grad()
        self.optimizer.zero_grad()
        batch_loss = self.compute_loss(obs=T.as_tensor(obs, dtype=T.float32),
                                  act=T.as_tensor(acts, dtype=T.int32),
                                  weights=T.as_tensor(weights, dtype=T.float32),
                                       n_traj = n_traj)

        batch_loss.backward()
        #self.mlp.optimizer.step()
        self.optimizer.step()
        return batch_loss



