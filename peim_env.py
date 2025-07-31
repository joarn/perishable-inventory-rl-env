import numpy as np
from numpy.typing import NDArray
from gymnasium import spaces

class PeimEnv:
    """
        Perishable Inventory Management Environment
        v0.1 created 11. Jul. 2025 by J.A.
    """

    # TODO: implement case lead_time == 0
    # TODO: implement aging of depot resources in the same way as the transient resources

    def __init__(self
            , num_markets
            , time_to_live
            , market_lambdas : NDArray
            , c_production=1.0
            , c_stock_keeping=0.01
            , c_transportation=0.1
            , c_waste=0.0, p_selling=1.5
            , max_timesteps=100
            , production_mu=10
            , production_sigma=2
            , lead_time=0):
        assert market_lambdas.shape == (num_markets, 1), f'market_lambdas must have shape (num_markets, 1), got {market_lambdas.shape}'

        self.num_markets = num_markets
        self.time_to_live = time_to_live

        self.state = None
        self.done = False

        self.c_production = c_production
        self.c_stock_keeping = c_stock_keeping
        self.c_transportation = c_transportation
        self.c_waste = c_waste

        self.p_selling = p_selling

        self.max_timesteps = max_timesteps
        self.current_timestep = 0
        
        self.market_lambdas = market_lambdas

        self.production_mu = production_mu
        self.production_sigma = production_sigma

        self.allocation_buffer = None
        self.lead_time = lead_time
        if self.lead_time > 0:
            self.allocation_buffer = np.zeros((self.num_markets, self.time_to_live, self.lead_time)) # our convention: newest action is added as slice (:,:,0) while action/ allocation to actually arrive this time step is at (:,:,-1)

        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.num_markets, self.time_to_live),
            dtype=np.float32
        )

    def reset(self):
        self.state = np.zeros((self.num_markets + 1, self.time_to_live))
        self.done = False

        self.current_timestep = 0

        if self.lead_time > 0:
            self.allocation_buffer = np.zeros((self.num_markets, self.time_to_live, self.lead_time))

        return self.state

    def shift_columns_right(self, array_2d, new_first_column):
            return np.hstack([new_first_column.reshape(-1, 1), array_2d[:, :-1]])


    def adapt_action(self, action : NDArray):
        available_resources = self.state[0,:]

        total_demand_per_resource = np.sum(action, axis=0)

        is_valid = np.all(total_demand_per_resource <= available_resources)

        if is_valid:
            return action.copy(), True

        # Else: action is invalid. We need to adapt it!
        adapted_action = np.zeros_like(action)
        remaining_resources = available_resources.copy()

        for market_idx in range(self.num_markets):
            market_demand = action[market_idx,:]

            # Allocate as much as possible for this market
            allocated = np.minimum(market_demand, remaining_resources)
            adapted_action[market_idx,:] = allocated

            remaining_resources -= allocated

            # if no resources left at all, break early
            if np.all(remaining_resources == 0):
                break

        return adapted_action, False


    def distribute_demand(self, demand_per_market : NDArray):
        selloff_matrix = np.zeros_like(self.state[1:,:], dtype=int)

        for market_idx in range(self.num_markets):
            # get supply and demand for this market
            market_supply = self.state[(market_idx+1)] # skip the first row (depot)
            market_demand = demand_per_market[market_idx]

            # Nothing happens if no demand or no supply
            if market_demand == 0 or np.sum(market_supply) == 0:
                continue

            # Calculate distribution proportional to supply
            total_supply = np.sum(market_supply)
            probabilities = market_supply / total_supply

            # Sample distributed demands for this market using multinomial
            selloff = np.random.multinomial(market_demand, probabilities)

            # We cannot sell more than is available
            selloff = np.minimum(selloff, market_supply)

            selloff_matrix[market_idx] = selloff

        return selloff_matrix

    def ageing_allocation_buffer(self, action):
        # Take note of wasted resources
        waste = self.allocation_buffer[:,-1,:]
        
        # Age all of the transient actions (individually)
        self.allocation_buffer[:,1:,:] = self.allocation_buffer[:,:-1,:]
         
        # Take note of currently arriving action
        currently_arriving_action = self.allocation_buffer[:,:,-1]

        # Then shift the actions: the last one in the buffer is the one that arrives now
        self.allocation_buffer[:,:,1:] = self.allocation_buffer[:,:,:-1]

        # The new action is added to the front of the buffer
        self.allocation_buffer[:,:,0] = action

        # We return the action that just arrived and the resources wasted in transit this timestep
        return currently_arriving_action, waste

    def get_current_supply(self):
        # Get current supply levels plus the resources that are on their way
        return self.state[1:,:] + self.allocation_buffer.sum(axis=2)

    def step(self, action):
        action = action.reshape(self.num_markets, self.time_to_live)
        # assert action.shape == (self.num_markets, self.time_to_live)
        # Get resource inflow at depot
        inflow_t = np.ceil(np.random.normal(self.production_mu, self.production_sigma)).astype(int)
        
        freshest_resources_t = np.zeros((self.num_markets+1, 1))
        freshest_resources_t[0] = inflow_t

        # Save wasted resources before performing shift operation
        location_waste_t = self.state[:,-1] # Current oldest supplies in each location, including the depot.
        
        # Time shift operation
        self.state = self.shift_columns_right(self.state, freshest_resources_t)

        # Action adaptation (such that we do not try to allocate more than available)
        action, bool_valid_action = self.adapt_action(action)

        # Age allocation buffer + Add new action to buffer and get current action from buffer
        currently_arriving_action, transportation_waste = self.ageing_allocation_buffer(action)

        # Action execution
        self.state[1:,:] += currently_arriving_action

        # Adjust the current state not according to the action that just arrived, but according to the action that was just setn.
        self.state[0,:] -= action.sum(axis=0)

        # Sample location demands
        market_demands = np.random.poisson(self.market_lambdas)

        # Sample precise location demands using requested resources and uniformly from available resources at each location
        selloff_matrix = self.distribute_demand(market_demands)

        # Sell off goods
        self.state[1:,:] -= selloff_matrix

        # Reward calculation
        reward_t = -self.c_production * inflow_t - self.c_stock_keeping * self.state.sum() - self.c_transportation * action.sum() + self.p_selling * selloff_matrix.sum()

        # Check if episode is done
        self.current_timestep += 1
        if self.current_timestep >= self.max_timesteps:
            self.done = True

        info = {'inflow_t' : inflow_t
                , 'location_waste_t' : location_waste_t
                , 'allocation_waste_t' : transportation_waste
                , 'market_demands' : market_demands
                , 'selloff' : selloff_matrix
                # , 'allocation_buffer' : self.allocation_buffer
                , 'action' : action
                , 'currently_arriving_action' : currently_arriving_action
                }
                
        return self.state, reward_t, self.done, info


