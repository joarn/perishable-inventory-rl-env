import numpy as np
from numpy.typing import NDArray

try:
    from gymnasium import spaces
except ImportError:  # Keep the core simulator usable without the optional RL dependency.
    class _Box:
        def __init__(self, low, high, shape, dtype):
            self.low = np.full(shape, low, dtype=dtype)
            self.high = np.full(shape, high, dtype=dtype)
            self.shape = shape
            self.dtype = dtype

        def sample(self):
            return np.random.uniform(self.low, self.high, self.shape).astype(self.dtype)

    class _Spaces:
        Box = _Box

    spaces = _Spaces()

class LegacyPeimEnv:
    """
        Perishable Inventory Management Environment
        Revised experiment environment. Originally created 11 Jul. 2025 by J.A.

        Inventory age classes are ordered from freshest (column 0) to oldest
        (the final column). ``issuance_rule`` controls which age classes are
        used to satisfy market demand:

        - ``random`` samples physical units without replacement;
        - ``fifo`` sells the oldest available units first;
        - ``lifo`` sells the freshest available units first.
    """

    ISSUANCE_RULES = frozenset({"random", "fifo", "lifo"})

    # This model supports positive integer transportation lead times.

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
            , lead_time=1
            , issuance_rule="random"
            , seed=None):
        if isinstance(lead_time, (bool, np.bool_)) or not isinstance(lead_time, (int, np.integer)) or lead_time < 1:
            raise ValueError('lead_time must be a positive integer (zero lead time is not supported)')

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

        self.issuance_rule = issuance_rule
        self._set_seed(seed)

        self.allocation_buffer = None
        self.current_inflow = 0
        self.lead_time = lead_time
        if self.lead_time > 0:
            self.allocation_buffer = np.zeros((self.num_markets, self.time_to_live, self.lead_time)) # our convention: newest action is added as slice (:,:,0) while action/ allocation to actually arrive this time step is at (:,:,-1)

        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.num_markets, self.time_to_live),
            dtype=np.float32
        )

    def _set_seed(self, seed):
        """Create independent streams for dynamics, issuance, and projection."""
        seed_sequence = np.random.SeedSequence(seed)
        dynamics_seed, issuance_seed, projection_seed = seed_sequence.spawn(3)
        self._dynamics_rng = np.random.default_rng(dynamics_seed)
        self._issuance_rng = np.random.default_rng(issuance_seed)
        self._projection_rng = np.random.default_rng(projection_seed)

    def reset(self, seed=None):
        if seed is not None:
            self._set_seed(seed)

        self.state = np.zeros((self.num_markets + 1, self.time_to_live))
        self.done = False

        self.current_timestep = 0

        if self.lead_time > 0:
            self.allocation_buffer = np.zeros((self.num_markets, self.time_to_live, self.lead_time))

        # Opening-of-period convention: the state shown to the policy already
        # contains the production available for allocation in this period.
        self.current_inflow = self._sample_production()
        self.state[0, 0] = self.current_inflow

        return self.state

    def _sample_production(self):
        return int(np.ceil(
            self._dynamics_rng.normal(self.production_mu, self.production_sigma)
        ))

    def shift_columns_right(self, array_2d, new_first_column):
            return np.hstack([new_first_column.reshape(-1, 1), array_2d[:, :-1]])

    @property
    def issuance_rule(self):
        return self._issuance_rule

    @issuance_rule.setter
    def issuance_rule(self, value):
        if not isinstance(value, str):
            raise ValueError(
                f'issuance_rule must be one of {sorted(self.ISSUANCE_RULES)}, got {value!r}'
            )

        normalized_value = value.lower()
        if normalized_value not in self.ISSUANCE_RULES:
            raise ValueError(
                f'issuance_rule must be one of {sorted(self.ISSUANCE_RULES)}, got {value!r}'
            )

        self._issuance_rule = normalized_value


    def adapt_action(self, action : NDArray):
        """Project integer requests proportionally onto depot capacity.

        Each age class is handled independently. When requests exceed the
        available stock, Hamilton's largest-remainder rule converts the
        proportional quotas to integers. Exact fractional ties are randomized
        with a seeded, policy-independent stream so market indices do not
        receive systematic priority.
        """
        requested = np.asarray(action, dtype=float)
        expected_shape = (self.num_markets, self.time_to_live)
        if requested.shape != expected_shape:
            raise ValueError(
                f'action must have shape {expected_shape}, got {requested.shape}'
            )
        if not np.all(np.isfinite(requested)):
            raise ValueError('action requests must be finite')
        if np.any(requested < 0):
            raise ValueError('action requests cannot be negative')
        if not np.allclose(requested, np.rint(requested)):
            raise ValueError('environment actions must contain integer quantities')

        requested = np.rint(requested).astype(int)
        available_resources = np.floor(self.state[0, :]).astype(int)
        totals = requested.sum(axis=0)
        is_valid = bool(np.all(totals <= available_resources))
        if is_valid:
            return requested.copy(), True

        projected = np.zeros_like(requested)
        for age_idx, available in enumerate(available_resources):
            age_requests = requested[:, age_idx]
            total_requested = int(age_requests.sum())
            if total_requested <= available:
                projected[:, age_idx] = age_requests
                continue
            if available <= 0 or total_requested == 0:
                continue

            quotas = age_requests * (available / total_requested)
            allocation = np.floor(quotas).astype(int)
            remainder = int(available - allocation.sum())
            if remainder:
                fractions = quotas - allocation
                tie_order = self._projection_rng.permutation(self.num_markets)
                ranked = tie_order[
                    np.argsort(-fractions[tie_order], kind='stable')
                ]
                allocation[ranked[:remainder]] += 1

            projected[:, age_idx] = allocation

        return projected, False


    @staticmethod
    def _ordered_selloff(market_supply, units_to_sell, age_indices):
        """Allocate sales greedily over age classes in the supplied order."""
        selloff = np.zeros_like(market_supply, dtype=int)
        remaining_demand = units_to_sell

        for age_idx in age_indices:
            if remaining_demand == 0:
                break
            sold = min(remaining_demand, market_supply[age_idx])
            selloff[age_idx] = sold
            remaining_demand -= sold

        return selloff

    def _random_selloff(self, market_supply, units_to_sell):
        """Sample physical units uniformly without replacement across ages."""
        selloff = np.zeros_like(market_supply, dtype=int)
        remaining_demand = units_to_sell
        remaining_supply = int(np.sum(market_supply))

        for age_idx, available_units in enumerate(market_supply[:-1]):
            if remaining_demand == 0:
                break
            sold = self._issuance_rng.hypergeometric(
                ngood=available_units,
                nbad=remaining_supply - available_units,
                nsample=remaining_demand,
            )
            selloff[age_idx] = sold
            remaining_demand -= sold
            remaining_supply -= available_units

        if remaining_demand > 0:
            selloff[-1] = remaining_demand

        return selloff

    def distribute_demand(self, demand_per_market : NDArray):
        """Allocate market demand according to the configured issuance rule."""
        selloff_matrix = np.zeros_like(self.state[1:,:], dtype=int)

        demands = np.asarray(demand_per_market).reshape(-1)
        if demands.size != self.num_markets:
            raise ValueError(
                f'demand_per_market must contain {self.num_markets} values, got {demands.size}'
            )

        for market_idx in range(self.num_markets):
            # get supply and demand for this market
            raw_market_supply = self.state[(market_idx+1)] # skip the first row (depot)
            market_supply = np.floor(raw_market_supply).astype(int)
            market_demand = int(demands[market_idx])

            if market_demand < 0:
                raise ValueError('market demand cannot be negative')
            if np.any(raw_market_supply < 0):
                raise ValueError('market inventory cannot be negative')

            # Nothing happens if no demand or no supply
            if market_demand == 0 or np.sum(market_supply) == 0:
                continue

            units_to_sell = min(market_demand, int(np.sum(market_supply)))

            if self.issuance_rule == "fifo":
                selloff = self._ordered_selloff(
                    market_supply,
                    units_to_sell,
                    range(self.time_to_live - 1, -1, -1),
                )
            elif self.issuance_rule == "lifo":
                selloff = self._ordered_selloff(
                    market_supply,
                    units_to_sell,
                    range(self.time_to_live),
                )
            else:
                selloff = self._random_selloff(market_supply, units_to_sell)

            selloff_matrix[market_idx] = selloff

        return selloff_matrix

    def receive_due_shipments(self):
        """Remove and return shipments with one period of lead time remaining."""
        currently_arriving_action = self.allocation_buffer[:, :, -1].copy()
        self.allocation_buffer[:, :, -1] = 0
        return currently_arriving_action

    def advance_allocation_buffer(self, action):
        """Age dispatched goods and advance all non-arriving shipments."""
        next_buffer = np.zeros_like(self.allocation_buffer)
        transportation_waste = np.zeros(
            (self.num_markets, self.lead_time), dtype=self.allocation_buffer.dtype
        )

        # Goods dispatched in period t appear in s_{t+1} one age class older
        # and at the start of the transport pipeline. Goods already in transit
        # also age and move one step closer to their destination.
        next_buffer[:, 1:, 0] = action[:, :-1]
        transportation_waste[:, 0] += action[:, -1]
        if self.lead_time > 1:
            next_buffer[:, 1:, 1:] = self.allocation_buffer[:, :-1, :-1]
            transportation_waste[:, 1:] += self.allocation_buffer[:, -1, :-1]

        self.allocation_buffer = next_buffer
        return transportation_waste

    def get_current_supply(self):
        # Get current supply levels plus the resources that are on their way
        return self.state[1:,:] + self.allocation_buffer.sum(axis=2)

    def step(self, action):
        # The published SAC notebook emits flattened actions, whereas the
        # heuristic policies use matrices. Support both representations.
        action = np.asarray(action).reshape(self.num_markets, self.time_to_live)
        requested_action = action.copy()

        # The action is checked against exactly the opening state observed by
        # the policy. No unobserved production can be used to make it feasible.
        action, bool_valid_action = self.adapt_action(action)

        # Dispatched goods leave the depot immediately but cannot satisfy this
        # period's demand. Only shipments already due in s_t arrive now.
        self.state[0, :] -= action.sum(axis=0)
        currently_arriving_action = self.receive_due_shipments()
        self.state[1:, :] += currently_arriving_action

        # Demand is realized and served after due shipments have arrived.
        market_demands = self._dynamics_rng.poisson(self.market_lambdas)
        selloff_matrix = self.distribute_demand(market_demands)

        self.state[1:, :] -= selloff_matrix

        # Reward components refer to period t, before the end-of-period ageing
        # operation. Production in s_t was sampled before the policy observed it.
        inflow_t = self.current_inflow
        production_cost = self.c_production * inflow_t
        stock_keeping_cost = self.c_stock_keeping * self.state.sum()
        transportation_cost = self.c_transportation * action.sum()
        sales_revenue = self.p_selling * selloff_matrix.sum()

        # Demand is served before expiry. Remaining oldest goods now expire;
        # dispatched and in-transit goods advance alongside on-hand inventory.
        location_waste_t = self.state[:, -1].copy()
        transportation_waste = self.advance_allocation_buffer(action)
        waste_units = location_waste_t.sum() + transportation_waste.sum()
        waste_cost = self.c_waste * waste_units

        reward_t = (
            sales_revenue
            - production_cost
            - stock_keeping_cost
            - transportation_cost
            - waste_cost
        )

        # Construct the opening state for t+1: remaining on-hand inventory
        # ages, expired units disappear, and new production becomes observable.
        self.current_inflow = self._sample_production()
        freshest_resources_t = np.zeros(self.num_markets + 1)
        freshest_resources_t[0] = self.current_inflow
        self.state = self.shift_columns_right(self.state, freshest_resources_t)

        # Check if episode is done
        self.current_timestep += 1
        if self.current_timestep >= self.max_timesteps:
            self.done = True

        info = {'inflow_t' : inflow_t
                , 'location_waste_t' : location_waste_t
                , 'allocation_waste_t' : transportation_waste
                , 'market_demands' : market_demands
                , 'selloff' : selloff_matrix
                , 'issuance_rule' : self.issuance_rule
                , 'requested_action' : requested_action
                # , 'allocation_buffer' : self.allocation_buffer
                , 'action' : action
                , 'valid_action' : bool_valid_action
                , 'currently_arriving_action' : currently_arriving_action
                , 'reward_components' : {
                    'production_cost' : production_cost
                    , 'stock_keeping_cost' : stock_keeping_cost
                    , 'transportation_cost' : transportation_cost
                    , 'sales_revenue' : sales_revenue
                    , 'waste_cost' : waste_cost
                    , 'waste_units' : waste_units
                }
                }

        return self.state, reward_t, self.done, info
