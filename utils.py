import matplotlib.pyplot as plt
import numpy as np

def plotLearning(scores, filename, x=None, window=5):
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-window):(t+1)])
    if x is None:
        x = [i for i in range(N)]
    plt.ylabel('Score')
    plt.xlabel('Game')
    plt.plot(x, running_avg)
    plt.savefig(filename)

# Feedback Scheme DQN Paper p. 14
def calculate_feedback(rews_total, agent, beta = 10):
    T = len(rews_total)
    w = 0
    tau_aux = [[] for i in range(len(rews_total[0]))]
    for i in range(4):
        for t in range(T):
            tau_aux[i].append(rews_total[t][i])
    tau_i = [((1 / T) * sum(elem)) for elem in tau_aux]
    w = sum(tau_i)

    feedback = []
    for t in range(T):
        r_modified = (rews_total[t][agent] + (beta / 3) * (w - tau_i[agent]))
        feedback.append(r_modified)
    return feedback

def total_feedback(rews_total, agent, beta = 10):
    fb = []
    for j in range(4):
        T = len(rews_total)
        w = 0
        tau_aux = [[] for i in range(len(rews_total[0]))]
        for i in range(4):
            for t in range(T):
                tau_aux[i].append(rews_total[t][i])
        tau_i = [((1 / T) * sum(elem)) for elem in tau_aux]
        w = sum(tau_i)

        feedback = []
        for t in range(T):
            r_modified = (rews_total[t][j] + (beta / 3) * (w - tau_i[j]))
            feedback.append(r_modified)
        fb.append(feedback)
    return fb

def calculate_expectedReturn(rew, discount=1):
    G = [None]*len(rew)
    for t in range(len(rew)):
            G_sum = 0
            gamma = 1
            for k in range(t, len(rew)):
                G_sum += rew[k] * gamma
                gamma *= discount
            G[t] = G_sum
    #mean = sum(G)/len(G)
    #G = G - np.mean(G)
    #G = G / np.std(G)
    #G = G.tolist()
    """weights = [None] * len(rew)
    for t in range(len(rew)):
        discounted_rew = []
        gamma = discount
        for i in range(0, len(rew)-t):
            discounted_rew.append(discount**i * rew[i])
        weights[t] = sum(discounted_rew)"""
    return G
    #return  weights - np.mean(weights)
