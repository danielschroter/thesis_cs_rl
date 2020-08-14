import gym
from gym import error, spaces
from gym.utils import seeding
from collections import deque
import numpy as np


def transform_obs(x: dict):
    """
    transform dict of observations (one step) to an array
    :param x: dict
    :rtype: np.array
    """
    return np.array((x['inventory_level'], x['on_order'], x['arriving_order'], x['arriving_shipments']))


def state_dict_to_array(states):
    """
    transform dict of observations (current step and previous steps) to an array
    The returned state is an 3 dim array:
    For each agent, there are the observations (state variables) of the last observed states.
    The state variables are defined in transform_obs.

    Example:
    [[[12,16,4,4]],
    [[12,16,4,4]]
    [[12,16,4,4]]
    [[12,16,4,4]]]

    The state of each agent consist only out of 1 observation (current):
    Each observation is defined according to tranform_obs:
    1. inventory balance
    2. on_order items
    3. arriving_order
    4. arriving_shipments

    :param state_dict:
    :rtype: np.array
    """
    flatten = [[],[],[],[]]
    for i in range(len(states[0])):
        for t in range(len(states)):
            flatten[i].append(transform_obs(states[t][i]))
        flatten[i] = np.asarray(flatten[i])
    flatten = np.asarray(flatten)
    return flatten


class BeerGame(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, demand_dist = "classical", n_turns_per_game=40, n_discrete_actions=7,
                 n_observed_periods=5, discrete=True):
        super().__init__()
        self.discrete = discrete
        self.orders = []
        self.shipments = []
        self.arriving_orders = []
        self.arriving_shipments = []
        self.on_order = []
        self.inventory_levels = []
        self.stock_limit = 50
        self.holding_cost = None
        self.stockout_cost = None
        self.cum_holding_cost = None
        self.cum_stockout_cost = None
        self.demands = None
        self.cost_weights = None
        self.turn = None
        self.done = True
        self.prev_states = None
        self.n_discrete_actions = n_discrete_actions
        self.n_agents = 4
        self.demand_dist = demand_dist
        self.n_observed_periods = n_observed_periods


        if self.demand_dist not in ['classical', 'uniform_0_2', 'uniform_0_8', 'normal_10_4', 'test']:
            raise NotImplementedError("env_type must be in ['classical', 'uniform_0_2', 'normal_10_4', 'test']")

        """
        if self.policy_others not in ['random', 'base_stock', 'sterman']:
            raise NotImplementedError("policy_others must be in ['random', 'base_stock', 'sterman'")
        if self.controlled_agent not in [0, 1, 2, 3]:
            raise ValueError("controlled agent must be in [0, 1, 2, 3]")
        """

        self.state = None
        self.n_turns = n_turns_per_game
        self.seed()

        """
        Action Spaces of one player, example retailer, because only one player is observed!
        d + x rule means that action space depends on demand: ordered quantity = demand + {-n/2,..., 0, ..., n/2)
        Player               Min          Max
        1	                 0            n_discrete_actions
        
        where 0 = -n_discrete_actions/2, n_discrete_actions/2 = 0, n_discrete_actions=n_discrete_actions/2
        
        """
        self.action_space = None
        if self.discrete:
            self.action_space = spaces.MultiDiscrete([16]*self.n_agents)
        else:
            if (demand_dist == "uniform_0_8"):
                self.action_space = spaces.Box(low=0.0, high=8.0, dtype=np.float64, shape=(4,))
            else:
                self.action_space = spaces.Box(low=0.0, high=30.0, dtype=np.float64, shape=(4,))
        """
        Observation space
        Num	Observation               Min             Max
        0	Inventory Level           -Inf            Inf
        1	On order item             -Inf            Inf
        2	Arriving Order            -Inf            Inf
        3	Arriving Shipment         -Inf            Inf
        """

        self.observation_space = spaces.Box(low=-np.finfo(np.float32).max, high=np.finfo(np.float32).max,
                                            dtype=np.float64, shape=(4, 4*n_observed_periods))

    def _get_observations(self):
        observations = [None] * self.n_agents
        for i in range(self.n_agents):
            observations[i] = {'inventory_level': self.inventory_levels[i],
                               #'orders': list(self.orders[i]),
                               'on_order': self.on_order[i],
                               'arriving_order': self.arriving_orders[i],
                               #'shipments': list(self.shipments[i]),
                               'arriving_shipments': self.arriving_shipments[i]}
        return observations

    def _get_rewards(self):
        return -(self.holding_cost + self.stockout_cost)

    def _get_demand(self):
        return self.demands[self.turn]

    def _get_state(self):
        return state_dict_to_array(self.prev_states)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.done = False

        if self.demand_dist == 'classical':
            temp_orders = [[4, 4]] * (self.n_agents - 1) + [[4]]
            temp_shipments = [[4, 4]] * self.n_agents
            self.arriving_orders = [4] * self.n_agents
            self.inventory_levels = [12] * self.n_agents
            self.demands = [4] * 4 + [8] * (self.n_turns - 4)
            self.cost_weights = [[0.5] * self.n_agents, [1] * self.n_agents]

        elif self.demand_dist == 'uniform_0_2':
            temp_orders = [[1, 1]] * (self.n_agents - 1) + [[1]]
            temp_shipments = [[1, 1]] * self.n_agents
            self.arriving_orders = [1] * self.n_agents
            self.inventory_levels = [4] * self.n_agents

            # uniform [0, 2]
            self.demands = self.np_random.uniform(low=0, high=3, size=self.n_turns).astype(np.int)
            self.cost_weights = [[0.5] * self.n_agents, [1] * self.n_agents]

        elif self.demand_dist == 'uniform_0_8':
            ## what are the orders?
            temp_orders = [[4, 4]] * (self.n_agents - 1) + [[4]]
            temp_shipments = [[4, 4]] * self.n_agents
            self.arriving_orders = [4] * self.n_agents
            self.inventory_levels = [12] * self.n_agents

            # uniform [0, 8]
            self.demands = self.np_random.uniform(low=0, high=8, size=self.n_turns).astype(np.int)
            self.cost_weights = [[0.5] * self.n_agents, [1.0] * self.n_agents]

        elif self.demand_dist == 'normal_10_4':
            temp_orders = [[10, 10]] * (self.n_agents - 1) + [[10]]
            temp_shipments = [[10, 10]] * self.n_agents
            self.arriving_orders = [10] * self.n_agents
            self.inventory_levels = [40] * self.n_agents

            self.demands = self.np_random.normal(loc=10, scale=4, size=self.n_turns)
            self.demands = np.clip(self.demands, 0, 1000).astype(np.int)
            # dqn paper page 24
            self.cost_weights = [[1.0, 0.75, 0.5, 0.25] * self.n_agents, [10.0] + [0.0] * (self.n_agents - 1)]

        elif self.demand_dist == 'test':
            temp_orders = [[8, 8]] * (self.n_agents - 1) + [[4]]
            temp_shipments = [[4, 4]] * self.n_agents
            self.arriving_orders = [8] * self.n_agents
            self.inventory_levels = [12] * self.n_agents
            self.demands = [8] * 4 + [8] * (self.n_turns - 4)
            self.cost_weights = [[0.5] * self.n_agents, [1] * self.n_agents]

        else:
            raise ValueError('wrong env_type')

        # initialize other variables
        # Good Coding shouldnt depend on order of initialization!
        self.holding_cost = np.zeros(self.n_agents, dtype=np.float)
        self.stockout_cost = np.zeros(self.n_agents, dtype=np.float)
        self.cum_holding_cost = np.zeros(self.n_agents, dtype=np.float)
        self.cum_stockout_cost = np.zeros(self.n_agents, dtype=np.float)
        self.orders = [deque(x) for x in temp_orders]
        self.shipments = [deque(x) for x in temp_shipments]
        self.arriving_orders = [self.demands[0]] + [x[0] for x in temp_orders[:-1]]
        orders_sum = [(x+y) for x,y in temp_orders[:-1]]+temp_orders[-1]
        shipments_sum = [(x+y) for x,y in temp_shipments]
        self.on_order = [(x+y) for x,y in zip(orders_sum,shipments_sum)]
        self.arriving_shipments = [x[0] for x in temp_shipments]
        self.turn = 0
        self.done = False

        self.prev_states = deque([self._get_observations()] * (self.n_observed_periods))
        self.state = self._get_state()
        return self.state

    def render(self, mode='human'):
        if mode != 'human':
            raise NotImplementedError(f'Render mode {mode} is not implemented yet')
        print('\n' + '=' * 20)
        print('Turn:     ', self.turn+1)
        print('Inventory_Level:   ', ", ".join([str(x) for x in self.inventory_levels]))
        print('Orders:   ', [list(x) for x in self.orders])
        print('Shipments:', [list(x) for x in self.shipments])
        print('On_Order:', ", ".join([str(x) for x in self.on_order]))
        print('Arriving_orders:', ", ".join([str(x) for x in self.arriving_orders]))
        print('Arriving_shipments:  ', ", ".join([str(x) for x in self.arriving_shipments]))
        print('Cum holding cost:  ', self.cum_holding_cost)
        print('Cum stockout cost: ', self.cum_stockout_cost)
        print('Total Cost: ', self.cum_stockout_cost+self.cum_holding_cost)
        print('Last holding cost: ', self.holding_cost)
        print('Last stockout cost:', self.stockout_cost)

    def step(self, action):
        # sanity checks
        if self.done:
            raise error.ResetNeeded('Environment is finished, please run env.reset() before taking actions')
        if len(action) != self.n_agents:
            raise error.InvalidAction(f'Length of action array must be same as n_agents({self.n_agents})')
        if any(np.array(action) < 0):
            raise error.InvalidAction(f"You can't order negative amount. You agents actions are: {action}")

        # Make incoming step (See Sterman 1989)

        # 1. Learn demand (inbound quantity) = self.arriving_orders

        # 2. Choose outbound order quantity = action

        # 3. Receive inbound shipment = self.arriving_shipments

        # 4. Ship your outbound shipment
        # calculate outgoing shipments respecting orders and stock levels, i+1 da retailer kein out_shipment
        shipments_out = [None] * self.n_agents
        for i in range(self.n_agents - 1):
            max_possible_shipment = max(0, self.inventory_levels[i + 1]) + self.arriving_shipments[i + 1]  # stock + incoming shipment
            order = self.arriving_orders[i+1] + max(0, -self.inventory_levels[i + 1])  # incoming order + stockout
            shipments_out[i] = min(order, max_possible_shipment)
        shipments_out[-1] = self.orders[-1][0]

        # Update inventory level, state variables, and execute shipments

        for i in range(self.n_agents):
            self.inventory_levels[i] = self.inventory_levels[i] + self.arriving_shipments[i] - self.arriving_orders[i]
            self.orders[i].append(action[i])
            self.shipments[i].append(shipments_out[i])

            # Remove processed shipments and orders
            self.orders[i].popleft()
            self.shipments[i].popleft()

        #Reset & Calculate cost

        self.holding_cost = np.zeros(self.n_agents, dtype=np.float)
        self.stockout_cost = np.zeros(self.n_agents, dtype=np.float)

        for i in range(self.n_agents):
            if self.inventory_levels[i] >= 0:
                self.holding_cost[i] = self.inventory_levels[i] * self.cost_weights[0][i]
            else:
                self.stockout_cost[i] = -self.inventory_levels[i] * self.cost_weights[1][i]

        self.cum_holding_cost += self.holding_cost
        self.cum_stockout_cost += self.stockout_cost

        # calculate reward
        rewards = self._get_rewards()

        # check if done
        if self.turn == self.n_turns - 1:
            self.done = True
        else:
            self.turn += 1

        # Update state variables
        self.arriving_orders = [self._get_demand()] + [x[0] for x in self.orders[:-1]]
        self.arriving_shipments = [x[0] for x in self.shipments]
        self.on_order = [(sum(order)+sum(ship)) for order, ship in zip(self.orders, self.shipments)]

         # concatenate previous states, self.prev_states in an queue of previous states
        self.prev_states.popleft()
        self.prev_states.append(self._get_observations())
        state = self._get_state()
        return state, rewards, self.done, {}


if __name__ == '__main__':
    env = BeerGame("test", n_observed_periods=1)
    start_state = env.reset()
    env.render()
    done = False
    while not done:
        actions = np.random.uniform(0, 16, size=4)
        actions = actions.astype(int)
        step_state, step_rewards, done, _ = env.step(actions)
        # print("state: ", step_state)
        env.render()
