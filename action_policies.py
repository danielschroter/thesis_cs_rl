import numpy as np


def get_other_actions(policy, obs_total, demand_dist, turn):
    if policy not in ["random", "base_stock", "sterman"]:
        raise NotImplementedError("policy of other agents must be in: [random, base_stock, sterman]")

    if policy == "random":
        actions = np.random.uniform(0, 16, size=4)
        actions = actions.astype(int)
    elif policy == "base_stock" :
        if demand_dist == "classical" or demand_dist == "test":
            base_stock_level = [24,32,32,32] #according to dqn paper page 24
            if turn == 0:
                return [8,8,8,4]

            else:
                curr_obs = [x[-1] for x in obs_total]
                arr_order = [x[2] for x in curr_obs]
                return arr_order

        elif demand_dist == 'uniform_0_2':
            base_stock_level = [8,8,0,0] #according to dqn paper page 20
        elif demand_dist == 'uniform_0_8':
            base_stock_level = [19,20,20,14]
        elif demand_dist == 'normal_10_4':
            base_stock_level = [48,43,41,30] #according to dqn paper page 24
        else:
            raise ValueError("Demand_dist must be out of [classical, uniform_0_2, normal_10_4]."
                             "Other optimal base stock levels are not known")

        # pretend for the first 4 periods, that incoming orders are 8

        """
        curr_obs = [x[-1] for x in obs_total]
        inventory_level = [x[0] for x in curr_obs]
        on_order = [x[1] for x in curr_obs]
        arr_order = [x[2] for x in curr_obs]
        # if demand_dist == "classical" and turn in range(4):


        inventory_position = [(x+y-z) for x, y, z in zip(inventory_level,on_order, arr_order)]
        actions = [(x-y) for x, y in zip(base_stock_level, inventory_position)]
        actions = [max(0, a) for a in actions]
        """

    else:
        raise NotImplementedError("Only random or base_stock are currently implemented")
    return actions


#Feedback Scheme DQN Paper p. 14
def calculate_feedback(rews_total, agent, beta):
    T = len(rews_total)
    w = 0
    tau_aux = [[] for i in range(len(rews_total[0]))]
    for i in range(4):
        for t in range(T):
            tau_aux[i].append(rews_total[t][i])
    tau_i = [((1/T)*sum(elem)) for elem in tau_aux]
    w = sum(tau_i)

    feedback = []
    for t in range(T):
        r_modified = rews_total[t][agent] + (beta/3)*(w-tau_i[agent])
        feedback.append(r_modified)
    return feedback


feedback = calculate_feedback([[1,2,3,4], [5,6,7,8]], 1, 10)
