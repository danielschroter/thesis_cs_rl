import torch
from algorithms.pg.reinforce.pgAgent_cont import PGAgent_cont
from algorithms.pg.reinforce.pgAgent import PGAgent

import numpy as np
from beer_game.envs.beergame import BeerGame
from Agent_Simulator import AgentSimulator
from utils import calculate_feedback, calculate_expectedReturn
import pandas as pd
import matplotlib.pyplot as plt


# for training policy
def run_one_time(env, agent, agent_sim, batch_size = 5000, controlled_agent = 1, mode="train", discount = 1, beta = 10):
    train = True
    if mode == "train":
        train = True
    elif mode == "eval":
        train = False
    else:
        raise ValueError('run_mode must be "train' or "eval")

    # make some empty lists for logging
    batch_obs = []
    batch_acts = []
    batch_weights = []
    batch_rets = []
    n_rounds = 0

    # reset episode specific variables
    obs_total = env.reset()
    obs = obs_total[controlled_agent].flatten()
    obs = np.array(obs, dtype=np.float32)
    done = False
    ep_rews = []
    ep_rews_total = []


    # collect experience by acting in the environment with current policy
    while True:

        batch_obs.append(obs.copy())

        # act in environment, obs for only local information (obs_total contains information of all actors)
        if train:
            act = agent.get_action(torch.as_tensor(obs, dtype=torch.float32))
        else:
            with torch.no_grad():
                #act = agent.get_action(torch.as_tensor(obs, dtype=torch.float32), train=False)
                act = agent.get_action(torch.as_tensor(obs, dtype=torch.float32))
                if isinstance(act, np.ndarray):
                    act = act[0]



        actions = agent_sim.get_other_actions(obs_total, env.demand_dist, env.turn, env.orders,
                                              env.demands, env.shipments)

        actions[controlled_agent] = act
        obs_total, rew_total, done, _ = env.step(actions)
        obs, rew = obs_total[controlled_agent].flatten(), rew_total[controlled_agent]

        batch_acts.append(act)
        ep_rews.append(rew)
        ep_rews_total.append(rew_total)

        if done:
            ep_ret = sum([sum(elem) for elem in ep_rews_total])
            n_rounds = len(ep_rews_total)
            batch_rets.append(ep_ret)
            if train:
                feedback = calculate_feedback(ep_rews_total, controlled_agent, beta)
                exp_Ret = calculate_expectedReturn(feedback, discount)
                batch_weights += exp_Ret


            obs_total, done, ep_rews, ep_rews_total = env.reset(), False, [], [],
            agent_sim.reset()
            obs = obs_total[controlled_agent].flatten()
            finished_rendering_this_epoch = True

            if not train:
                break

            # end experience loope if we have enough of it
            if len(batch_obs) > batch_size:
                break

    # calculate batch_loss and take a single policy gradient update step
    if train:
        n_traj = len(batch_obs)/n_rounds
        for _ in range(7):
            batch_loss = agent.learn(batch_obs, batch_acts, batch_weights, n_traj)
        return batch_loss, batch_rets, batch_acts

    # return batch returns and batch actions if in evaluation mode
    return batch_rets, batch_acts


def train(env,agent, agent_sim, controlled_agent, epochs=1000, batch_size=5000,
          continuation=False, PATH=None, plot=False, discount=1, beta = 10):

    action_means = []
    return_means = []
    losses = []
    epoch = 0

    if continuation:
        checkpoint = torch.load(PATH)
        agent.mlp.load_state_dict(checkpoint['model_state_dict'])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        action_means = checkpoint['action_means']
        return_means = checkpoint['return_means']

    # set the network into train mode
    agent.mlp.train()

    # Create Plots
    plt.ion()
    f = plt.figure(0)
    ax1 = f.add_subplot(2, 1, 1)
    ax2 = f.add_subplot(2, 1, 2)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("Mean Orders in epoch")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("Mean Cost in epoch")

    #training loop
    for i in range(epoch, epochs):
        batch_loss, batch_rets, batch_acts = run_one_time(env, agent, agent_sim, batch_size, controlled_agent,
                                                          mode="train", discount=discount, beta=beta)
        action_means.append(sum(batch_acts) / len(batch_acts))
        return_means.append(sum(batch_rets) / len(batch_rets))
        losses.append(batch_loss)

        df_1 = pd.Series(action_means)
        df_2 = pd.Series(return_means)

        ax1.plot(df_1)
        ax2.plot(df_2)
        plt.pause(0.0001)

        print('epoch: %3d \t loss: %.3f \t return: %.3f' %
              (i, batch_loss, np.mean(batch_rets)))

        # save regulary
        if (i + 1) % 50 == 0:
            # PATH = "checkpoint.pt"
            torch.save({
                'epoch': i,
                'policy_others' : agent_sim.policy,
                'model_state_dict': agent.mlp.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'loss': losses,
                'action_means': action_means,
                'return_means': return_means
            }, PATH)

    f.show()
    return action_means, return_means


def evaluate(env, agent, agent_sim, controlled_agent, PATH=None, n_games=50):
    batch_actions = []
    batch_returns = []
    epoch = 0

    start_state = env.reset()
    #env.render()
    done = False

    if PATH:
        checkpoint = torch.load(PATH)
        agent.mlp.load_state_dict(checkpoint['model_state_dict'])
    agent.mlp.eval()


    for i in range(n_games):
        ret, batch_acts = run_one_time(env, agent, agent_sim, controlled_agent=controlled_agent, mode="eval")
        batch_actions.append(batch_acts)
        batch_returns.append(ret[0])

        print(f'epoch: {i} \t return: {ret} \t orders: {batch_acts}')

    print(f'mean return of {n_games} games played is {(sum(batch_returns)/len(batch_returns))}')

    return batch_returns, batch_actions


def plot_returns(return_means, PATH):
    f = plt.figure(1)
    plt.plot(return_means)
    plt.xlabel('epochs')
    plt.ylabel('mean cost per game')
    plt.title('Cost reduction')
    f.show()
    f.savefig(PATH)

def plot_action_means(action_means, PATH):
    g = plt.figure(2)
    plt.plot(action_means)
    plt.xlabel('epochs')
    plt.ylabel('mean order quantity per game')
    plt.title('Mean of Orders')
    g.show()
    g.savefig(PATH)

def grid_search(n_observed_preiods, discount, beta):
    LR = 1e-4
    CONTROLLED_AGENT = 1
    N_TURNS_PER_GAME = 36
    POLICY_OTHERS = "sterman"
    DEMAND_DIST = "classical"
    HIDDEN_SIZES = [32, 32]
    EPOCHS = 1600
    ALGORITHM = "policy_gradient"
    MODE = 'discrete'
    DISCRETE=True
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    CHECKPOINT = 'tmp/grid/pg/discrete'
    for p in n_observed_preiods:
        for d in discount:
            for b in beta:
                print(f"train: observed periods = {p}, discount = {d}, beta = {b}")
                PATH = os.path.join(CHECKPOINT, "periods_"+ str(p) + "_dis_" + str(d) + "_beta_" + str(b))
                env = BeerGame(demand_dist=DEMAND_DIST,  n_observed_periods=p, n_turns_per_game=36, discrete=DISCRETE)
                obs_dim = env.observation_space.shape[1]
                #n_acts = env.action_space.high[CONTROLLED_AGENT]
                n_acts = env.action_space.nvec[CONTROLLED_AGENT]
                pgAgent = PGAgent(LR, sizes=[obs_dim] + HIDDEN_SIZES, n_actions=n_acts)
                agentSim = AgentSimulator(policy=POLICY_OTHERS, discrete=DISCRETE)
                action_means, return_means = train(env, pgAgent, agentSim, 1, epochs=EPOCHS, PATH=(PATH+'.pt'), continuation=False, discount=d,
                                                   beta=b)
                plot_returns(return_means, PATH = (PATH+".png"))
                ret, acts = evaluate(env, pgAgent, agentSim, n_games=1000, controlled_agent=CONTROLLED_AGENT, PATH=(PATH+'.pt'))




if __name__ == "__main__":

    import os
    from datetime import datetime
    import random

    #grid_search([5,10], [1, 0.95], [10,100])
    N_OBSERVED_PERIODS = 5
    LR = 1e-4
    CONTROLLED_AGENT = 1
    N_TURNS_PER_GAME = 36
    POLICY_OTHERS = "sterman"
    DEMAND_DIST = "classical"
    HIDDEN_SIZES = [32, 32]
    EPOCHS = 3600
    ALGORITHM = "policy_gradient"
    DISCRETE=False
    MODE = "discrete" if DISCRETE else "continuous"
    CONTINUATION=False
    BETA = 100
    beta = str(BETA)
    seed = 0
    returns = []
    actions = []
    dir = os.path.dirname(__file__)
    date = str(datetime.date(datetime.now()))
    PATH = os.path.join(dir, 'logs', 'checkpoint_' + date + '_' + MODE + "_" + DEMAND_DIST + '_' + POLICY_OTHERS + '_' + beta + 'rounded')
    #PATH = os.path.join(dir, 'logs', 'checkpoint_2020-10-12_continuous_classical_sterman_100rounded')

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = BeerGame(demand_dist=DEMAND_DIST,  n_observed_periods=N_OBSERVED_PERIODS, n_turns_per_game=N_TURNS_PER_GAME, discrete=DISCRETE)

    obs_dim = env.observation_space.shape[CONTROLLED_AGENT]

    if DISCRETE:
        n_acts = env.action_space.nvec[CONTROLLED_AGENT]
        pgAgent = PGAgent(LR, sizes=[obs_dim] + HIDDEN_SIZES, n_actions=n_acts)
    else:
        n_acts = env.action_space.high[CONTROLLED_AGENT]
        pgAgent = PGAgent_cont(LR, sizes=[obs_dim] + HIDDEN_SIZES, n_actions=2)

    agentSim = AgentSimulator(policy=POLICY_OTHERS, discrete=DISCRETE)
    action_means, return_means = train(env, pgAgent, agentSim, 1, epochs=EPOCHS, PATH=(PATH+'.pt'), continuation=CONTINUATION, beta=BETA)

    torch.save({
        'Mode' : MODE,
        'Algorithm': ALGORITHM,
        'Demand_Distribution' : DEMAND_DIST,
        'Policy_Others' : POLICY_OTHERS,
        'Controlled_Agent' : CONTROLLED_AGENT,
        'N_Turns_Per_Game' : N_TURNS_PER_GAME,
        'N_Observed_Periods' : N_OBSERVED_PERIODS,
        'Model_Summary' : str(pgAgent.mlp),
        'model_state_dict': pgAgent.mlp.state_dict(),
        'optimizer_state_dict': pgAgent.optimizer.state_dict(),
        #'loss': batch_loss,
        'action_means': action_means,
        'return_means': return_means,
    }, (PATH + '_summary.pt'))

    plot_returns(return_means, PATH = (PATH+'_plot.png'))
    plot_action_means(action_means, PATH = (PATH+'_action_mean_plot.png'))
    ret, acts = evaluate(env, pgAgent, agentSim, n_games=1000, controlled_agent=CONTROLLED_AGENT, PATH=(PATH+'.pt'))


