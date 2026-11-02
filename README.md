# perishable-inventory-rl-env

**Official implementation of the reinforcement learning environment for the paper:**

> **A Markov Decision Process Framework for the Allocation of Perishable Goods**  
> Joshua Arnold, Rudy Milani, Matthias Schilling, Stefan Pickl, Maximilian Moll
## Overview

This repository contains a Gymnasium environment and the simulator used by the
revised experiments. It models a depot supplying multiple markets with stochastic
production and demand, transportation delays, aging inventory, and expiry.
It contains the simulator and its tests. The SAC/AVI training code and complete
experiment runner are not included in this repository.

## Installation and example

Use Python 3.10 or newer. From the repository directory:

```sh
python -m pip install -r requirements.txt
```

```python
import gymnasium as gym
import numpy as np
import peim_env  # Registers PerishableInventory-v0.

env = gym.make(
    "PerishableInventory-v0",
    num_markets=5,
    time_to_live=7,
    market_lambdas=np.array([1.0, 2.0, 3.0, 1.5, 2.5]).reshape(-1, 1),
    lead_time=1,
    max_timesteps=30,
    issuance_rule="random",  # Also supports "fifo" and "lifo".
)
observation, info = env.reset(seed=123)
env.action_space.seed(456)
episode_return = 0.0
done = False
while not done:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    episode_return += reward
print(f"Episode return: {episode_return:.2f}")
env.close()
```

## Gymnasium interface

`PeimEnv` subclasses `gymnasium.Env`. You can instantiate it directly or use
`gym.make` after importing `peim_env` from this repository.

- `reset(seed=..., options=None)` returns `(observation, info)`.
- `step(action)` returns `(observation, reward, terminated, truncated, info)`.
- Observations use a `spaces.Dict` with three independent array snapshots:
  - `inventory`: on-hand stock, shape `(num_markets + 1, time_to_live)`.
    Row 0 is the depot; other rows are markets.
  - `pipeline`: transit stock, shape `(num_markets, time_to_live, lead_time)`.
    The final slice contains shipments due during the current period.
  - `remaining_steps`: a one-element integer array containing the remaining
    planning horizon. Together these arrays expose the full inventory state
    and the time needed for finite-horizon decisions.
- `time_to_live` is the number of usable age columns. Column 0 is freshest,
  and column `time_to_live - 1` is oldest. Seven columns mean ages 0 through 6.
- Actions use a `Box(0, 1)` of shape `(num_markets, time_to_live)`. Entries
  request shares of the observed depot stock of each age. Shares are scaled
  by `min(1, 1 / (column_sum + 1e-8))`, multiplied by available integer stock,
  and floored to whole-unit requests. This matches the experiment's SAC
  conversion, including its small numerical offset. Flooring can leave stock
  at the depot; leftover units are not redistributed by this conversion.
  Samples from `action_space` can be passed directly to `step`.
- The configured `max_timesteps` is the task's finite planning horizon.
  Reaching it sets `terminated=True`; the environment itself never truncates.
  An external `gymnasium.wrappers.TimeLimit` may set `truncated=True` instead
  when imposing a shorter limit. Reset after either signal.
- `render_mode="ansi"` makes `render()` return an inventory and pipeline text
  summary. `close()` releases the environment (no external resources are held).

Inventory has no fixed capacity, so the observation spaces have infinite upper
bounds. `gymnasium.wrappers.FlattenObservation(env)` can turn the dictionary into
a vector for algorithms requiring vector observations. Choose an algorithm or
policy that supports the declared observation and action spaces.

## Compatibility with existing experiment code

The previous integer-request, four-value API remains available as
`LegacyPeimEnv`. Existing experiment scripts should change their import to:

```python
from peim_env import LegacyPeimEnv as PeimEnv
```

For that interface, `reset(seed=...)` returns only on-hand inventory and
`step(requests)` returns `(state, reward, done, info)`. Requests are nonnegative
integer quantities, either shaped `(num_markets, time_to_live)` or flattened.
Oversubscribed requests are projected proportionally using largest remainders
and seeded random tie-breaking. `allocation_buffer` exposes the pipeline and
`get_current_supply()` sums market on-hand and pipeline inventory by age.
This compatibility class is not a Gymnasium `Env`; its historical Box describes
policy outputs rather than accepted integer requests. Use the new `PeimEnv`
for Gymnasium integration. Existing trained models are not automatically
compatible with the new, expanded observation format.

## Within-period timing and rewards

1. Observe opening inventory, including current production, and select requests.
2. Project requests against that observed depot inventory and dispatch goods.
3. Receive previously dispatched shipments due this period, at their current age.
4. Draw market demand and sell available goods under the selected issuance rule.
5. Charge production and transportation costs and holding costs on post-sales
   on-hand inventory. Count on-hand expiry, advance and age the pipeline, and
   count expiry in transit, including newly dispatched goods.
6. Compute revenue minus all costs, including the waste charge. Draw the next
   period's production, age surviving on-hand inventory and remove expired
   on-hand goods to construct the next opening state.

New shipments cannot serve demand in the dispatch period. The default lead
time is one period. Only positive integer lead times are supported. Zero lead
time raises `ValueError`.

The Gymnasium interface uses
`max(0, ceil(Normal(production_mu, production_sigma)))` for production, where
`production_sigma` is the standard deviation. Clamping prevents negative
inventory from rare negative Gaussian draws. `LegacyPeimEnv` retains the
experiment's untruncated draw for reproducibility; this is the one intentional
dynamics difference between the two interfaces. Demand is Poisson with the supplied market rates.
Random issuance samples units without replacement and always serves
`min(demand, available stock)`. FIFO sells oldest units first, while LIFO sells
freshest units first. Demand exceeding inventory is lost.

`info` includes requested and executed actions, feasibility, arrivals, demand,
sales by market and age, expiry by location and in transit, and a
`reward_components` dictionary containing production, holding, transportation
and waste costs, sales revenue, and total expired units. Production costs refer
to the production in the observed opening state, not the next period's draw.
Unexpired terminal inventory is not automatically counted as waste.

## Reproducibility

The environment has separate seeded streams for production/demand, issuance
and shortage projection. Reusing a scenario seed gives common production and
demand draws across policies, even when their sales and allocation decisions
differ. `reset(seed=...)` restarts the streams. `reset()` starts a new episode
while continuing the existing streams. Any policy randomness must be seeded
separately.

## Changes from the original public environment

- Actions use the exact opening inventory seen by the policy.
- Shipments respect lead times. Arrival and waste arrays no longer alias buffer
  memory that is subsequently overwritten.
- Shortage handling uses a shared proportional rule instead of market ordering.
- Random issuance respects inventory capacity. FIFO and LIFO are available.
- Expired on-hand and in-transit goods contribute to waste costs in the reward.
- Detailed operational information and seeded simulation streams are provided.

The dynamics and reward calculation are ported from the revised experiment
implementation. The public interface additionally defaults to `lead_time=1`
and rejects unsupported lead times early. These changes do not alter the
positive-lead-time experiment configuration.

## Tests

From the repository directory:

```sh
python -m unittest discover -s tests -v
```

Tests cover opening production, feasibility, issuance rules, shipment delays,
expiry costs, holding costs, reproducibility, and shortage tie-breaking.
They also run Gymnasium's official environment checker for all three issuance
rules and check action conversion, complete observation snapshots, registration,
rendering, flattening and the distinction between termination and truncation.

