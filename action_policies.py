import numpy as np
import math
import copy


class AgentSimulator():

    def __init__(self, policy, n_agents=4, discrete = True):
        self.expected_demands = [4] * n_agents
        self.n_agents = n_agents
        self.policy = policy
        self.arr_orders = [[4] for i in range(n_agents)]
        self.discrete = discrete

    def reset(self):
        self.expected_demands = [4] * self.n_agents
        self.arr_orders = [[4] for i in range(self.n_agents)]

    def get_other_actions(self, obs_total, demand_dist, turn, orders, demand, shipments, original = False):
        if self.policy not in ["random", "base_stock", "sterman"]:
            raise NotImplementedError("policy of other agents must be in: [random, base_stock, sterman]")

        curr_obs = [x[-1] for x in obs_total]
        orders = copy.deepcopy(orders)
        shipments = copy.deepcopy(shipments)
        n_agents = len(obs_total)
        inventory_levels = [x[0] for x in curr_obs]
        arriving_orders = [x[2] for x in curr_obs]
        arriving_shipments = [x[3] for x in curr_obs]

        # Calculate outgoing shipment
        shipments_out = [None] * self.n_agents
        for i in range(self.n_agents - 1):
            max_possible_shipment = max(0, inventory_levels[i + 1]) + arriving_shipments[
                i + 1]  # stock + incoming shipment
            order = arriving_orders[i + 1] + max(0, -inventory_levels[i + 1])  # incoming order + stockout
            shipments_out[i] = min(order, max_possible_shipment)
        shipments_out[-1] = orders[-1][0]

        # Update inventory level, state variables, and execute shipments
        for i in range(self.n_agents):
            inventory_levels[i] = inventory_levels[i] + arriving_shipments[i] - arriving_orders[i]
            shipments[i].append(shipments_out[i])

            # Remove processed shipments and orders
            orders[i].popleft()
            shipments[i].popleft()

        # select actions according to policy

        if self.policy == "random":
            actions = None
            if demand_dist == "classical":
                actions = np.random.uniform(0, 16, size=4)
            elif demand_dist == 'uniform_0_2':
                actions = np.random.uniform(0, 2, size=4)
            elif demand_dist == 'uniform_0_8':
                actions = np.random.uniform(0, 8, size=4)
            elif demand_dist == 'normal_10_4':
                actions = np.random.uniform(0, 20, size=4)
            else:
                raise ValueError("Demand_dist must be out of [classical, uniform_0_8, uniform_0_2, normal_10_4]."
                                 "Other optimal base stock levels are not known")
            if self.discrete:
                actions = actions.astype(int)
        elif self.policy == "base_stock":
            if demand_dist == "classical" or demand_dist == "test":
                base_stock_level = [32, 32, 32, 24]  # according to dqn paper page 24
            elif demand_dist == 'uniform_0_2':
                base_stock_level = [8, 8, 0, 0]  # according to dqn paper page 20
            elif demand_dist == 'uniform_0_8':
                base_stock_level = [19, 20, 20, 14]
            elif demand_dist == 'normal_10_4':
                base_stock_level = [48, 43, 41, 30]  # according to dqn paper page 24
            else:
                raise ValueError("Demand_dist must be out of [classical, uniform_0_8, uniform_0_2, normal_10_4]."
                                 "Other optimal base stock levels are not known")

            orders_sum = [sum(x) for x in orders]
            shipments_sum = [sum(x) for x in shipments]
            inventory_position = [(x + y + z) for x, y, z in zip(inventory_levels, orders_sum, shipments_sum)]
            actions = [(x - y) for x, y in zip(base_stock_level, inventory_position)]
            actions = [max(0, a) for a in actions]

        elif self.policy == "sterman":

            # Parameters and initial values
            mdt = [1 for i in range(n_agents - 1)]  # mailing delay time
            sat = [1 for i in range(n_agents)]  # stock adjustment time (corrsp. to 1/alphas in Sterman 1989)
            st = [2 for i in range(n_agents - 1)]  # shipment_time
            plt = 2  # production lead time
            dsl = [None] * n_agents
            sl = [None] * n_agents
            beta = [1] * n_agents
            theta = [0.2] * n_agents

            for i in range(n_agents):
                self.arr_orders[i].append(arriving_orders[i])

            ## Expectation Formation
            for i in range(n_agents):
                if i == 0:
                    self.expected_demands[i] = self.expected_demands[i] + \
                                               theta[i] * (demand[turn] - self.expected_demands[i])
                else:
                    self.expected_demands[i] = self.expected_demands[i] + \
                                               theta[i] * (arriving_orders[i] - self.expected_demands[i])

            if turn == len(demand) - 1:
                arriving_orders = [demand[turn]] + [x[0] for x in orders[:-1]]

            else:
                arriving_orders = [demand[turn + 1]] + [x[0] for x in orders[:-1]]

            # calculate desired inventory level
            for i in range(n_agents):
                if i == 0:
                    dsl[i] = self.expected_demands[i] * (mdt[i] + st[i])
                elif i == n_agents - 1:
                    dsl[i] = self.expected_demands[i] * plt
                else:
                    dsl[i] = self.expected_demands[i] * (mdt[i] + st[i])

            shipments_sum = [sum(x) for x in shipments]



            if turn == 0:
                d_mean = [4] * n_agents
            else:
                d_mean = [sum(elem) / len(elem) for elem in self.arr_orders]

            # According to dqn paper using the mean for the desired inventory model. Better suits other demand ditr.
             # In A Matheatical Model of the Beer Game by Edali and Yasarcan the desired inventory is 0

            if original:
                desired_inventory = [0]*n_agents
            else:
                desired_inventory = d_mean
            #

            sl = [(arriving_orders[i + 1] + shipments_sum[i] + max(0, -inventory_levels[i + 1])) for i in
                  range(n_agents - 1)] + [shipments_sum[-1]]
            sla = [(beta[i] * (dsl[i] - sl[i]) / sat[i]) for i in range(n_agents)]
            ia = [(desired_inventory[i] - inventory_levels[i]) / sat[i] for i in range(n_agents)]

            new_orders = [None] * n_agents
            for i in range(n_agents):
                if turn < 4:
                    return [4] * n_agents
                else:
                    new_orders[i] = max(0, self.expected_demands[i] + ia[i] + sla[i])
            if self.discrete:
                new_orders = [math.floor(elem + 0.5) for elem in new_orders]
            return new_orders

        else:
            raise NotImplementedError("Only random, sterman or base_stock are currently implemented")
        return actions

        # pretend for the first 4 periods, that incoming orders are 8


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
        r_modified = rews_total[t][agent] + (beta / 3) * (w - tau_i[agent])
        feedback.append(r_modified)
    return feedback


feedback = calculate_feedback([[1, 2, 3, 4], [5, 6, 7, 8]], 1, 10)
