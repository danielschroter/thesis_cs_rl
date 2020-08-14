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
import math
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
        #self.activation = nn.Tanh()
        #self.output_activation = nn.Identity()
        self.out_size = out_size
        self.sizes = sizes

        layers = []
        for j in range(len(sizes) - 1):
            layers += [nn.Linear(sizes[j], sizes[j + 1], activation())]
        self.base = nn.Sequential(*layers)

        #for multidimensional case
        self.mu = nn.Sequential(
            nn.Linear(self.sizes[-1], 1),
            output_activation(),
        )
        self.var = nn.Sequential(
            nn.Linear(self.sizes[-1], 1),
            nn.Softplus(),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cuda:1')
        self.to(self.device)

    def forward(self, observation):
        x = T.Tensor(observation).to(self.device)
        #x = observation
        for layer in self.base:
            x = layer(x)
        mu_value = self.mu(x)
        sigma = self.var(x)
        return mu_value, sigma

class PGAgent_cont():
    def __init__(self, alpha, sizes, n_actions):
        #self.mlp = GenericNetwork(sizes, out_size=1, alpha=alpha)
        self.mlp = mlp(sizes=sizes+[n_actions])
        self.optimizer = optim.Adam(self.mlp.parameters(), lr=alpha)
        self.log_probs = None
        self.controlled_agent = 1
        self.policy_others = "sterman"
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cuda:1')
        #self.to(self.device)

    def choose_action(self, obs):
        logits = self.mlp.forward(obs)
        action_probs = T.distributions.Categorical(logits=logits)
        action = action_probs.sample()
        # self.log_probs = action_probs.log_prob(action)
        return action.item()

    def get_policy(self, obs):
        sigma, mu = self.mlp.forward(obs)
        m = T.nn.Softplus()
        sigma = m(sigma)
        return T.distributions.Normal(mu, sigma)

    """ def get_policy(self, obs):

        mu, sigma = self.mlp.forward(obs)
        return T.distributions.Normal(mu, sigma)"""

    # make action selection function
    def get_action(self, obs):
        pol = self.get_policy(obs)
        act = pol.sample()
        #self.log_probs = pol.log_prob(act)
        action = np.clip(act.item(),0,50)
        return action

# make loss function, whose gradient, for the right data is policy gradient
    def compute_loss(self, obs, act, weights):
        """l = []
        for o,a,w in zip(obs,act,weights):

            pol = self.get_policy(o)
            logp = pol.log_prob(a)
            res = logp*w
            res = T.clamp(res, 0, 300)
            l.append(res)
        acc = sum(l)
        mean = -acc/len(act)
        """

        """

        out = self.mlp.forward(obs)
        mu = out[:,0]
        sigma = out[:,1]
        w = T.Tensor(weights).to(self.device)
        a = T.Tensor(act).to(self.device)

        mu, sigma = self.mlp.forward(obs)

        # https://github.com/colinskow/move37/blob/master/actor_critic/a2c_continuous.py
        # https://medium.com/@vittoriolabarbera/continuous-control-with-a2c-and-gaussian-policies-mujoco-pytorch-and-c-4221ec8ba024

        p1 = -((mu - a)**2)/(2*((sigma**2).clamp(min=1e-3)))
        p2 = - T.log(T.sqrt(2*math.pi*(sigma**2)))
        return -((p1+p2)*w).mean()"""



        #From Tutorial of spinning up
        #logp = self.get_policy(obs).log_prob(act).sum(axis=-1)
        #return -(logp * weights).mean()
        #return mean

        out = self.mlp.forward(obs)
        mu = out[:,1]
        sigma = out[:,0]

        # https://github.com/colinskow/move37/blob/master/actor_critic/a2c_continuous.py
        # https://medium.com/@vittoriolabarbera/continuous-control-with-a2c-and-gaussian-policies-mujoco-pytorch-and-c-4221ec8ba024

        p1 = -((mu - act)**2)/(2*((sigma**2).clamp(min=1e-3)))
        p2 = - T.log(T.sqrt(2*math.pi*(sigma**2)))
        return -((p1+p2)*weights).mean()



    def learn(self, obs, acts, weights):
        self.optimizer.zero_grad()
        batch_loss = self.compute_loss(obs=T.as_tensor(obs, dtype=T.float32),
                                  act=T.as_tensor(acts, dtype=T.float32),
                                  weights=T.as_tensor(weights, dtype=T.float32))

        batch_loss.backward()
        self.optimizer.step()
        return batch_loss


