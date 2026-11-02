"""Gymnasium interface for perishable inventory allocation.

Importing this module registers ``PerishableInventory-v0``. The original
integer-request simulator is available as ``LegacyPeimEnv``.
"""

import gymnasium as gym
from gymnasium import spaces
from gymnasium.error import ResetNeeded
from gymnasium.envs.registration import register, registry
import numpy as np

from peim_core import LegacyPeimEnv


class _NonnegativeProductionCore(LegacyPeimEnv):
    def _sample_production(self):
        # A Gaussian can produce negative draws. Keep the public RL observation
        # physically valid, while the legacy simulator preserves the experiment.
        return max(0, super()._sample_production())


class PeimEnv(gym.Env):
    """Allocate stock with continuous shares and observe the full inventory state.

    Actions have shape (markets, age classes), with entries in [0, 1]. Each
    entry requests a fraction of the depot stock of that age. If a column's
    shares sum to more than one, it is normalized before flooring to units.
    This is the continuous-action conversion used by the experiment's SAC.

    Observations are dictionaries with ``inventory``, ``pipeline``, and
    ``remaining_steps`` arrays. The specified planning horizon is part of
    the task, so reaching it terminates the episode (it is not truncation).
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 1}

    def __init__(self, num_markets, time_to_live, market_lambdas, *,
                 render_mode=None, seed=None, **kwargs):
        super().__init__()
        if render_mode not in (None, "ansi"):
            raise ValueError("render_mode must be None or 'ansi'")
        for name, value in [("num_markets", num_markets),
                            ("time_to_live", time_to_live),
                            ("max_timesteps", kwargs.get("max_timesteps", 100))]:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        rates = np.asarray(market_lambdas, dtype=float).copy()
        if rates.shape != (num_markets, 1) or not np.all(np.isfinite(rates)) or np.any(rates < 0):
            raise ValueError("market_lambdas must be finite, nonnegative and have shape (num_markets, 1)")
        for name in ["production_mu", "production_sigma", "c_production",
                     "c_stock_keeping", "c_transportation", "c_waste", "p_selling"]:
            if name in kwargs and not np.isfinite(kwargs[name]):
                raise ValueError(f"{name} must be finite")
        if kwargs.get("production_sigma", 2) < 0:
            raise ValueError("production_sigma cannot be negative")
        self.render_mode = render_mode
        self.simulator = _NonnegativeProductionCore(
            num_markets=num_markets, time_to_live=time_to_live,
            market_lambdas=rates, seed=seed, **kwargs,
        )
        self.action_space = spaces.Box(0.0, 1.0, (num_markets, time_to_live), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "inventory": spaces.Box(0.0, np.inf, (num_markets + 1, time_to_live), dtype=np.float64),
            "pipeline": spaces.Box(0.0, np.inf, (num_markets, time_to_live, self.simulator.lead_time), dtype=np.float64),
            "remaining_steps": spaces.Box(0, self.simulator.max_timesteps, (1,), dtype=np.int64),
        })
        self._has_reset = False

    def _observation(self):
        # Return snapshots so rollout buffers cannot be changed by later steps.
        return {
            "inventory": self.simulator.state.copy(),
            "pipeline": self.simulator.allocation_buffer.copy(),
            "remaining_steps": np.array([
                self.simulator.max_timesteps - self.simulator.current_timestep
            ], dtype=np.int64),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options:
            raise ValueError("No custom reset options are supported")
        self.simulator.reset(seed=seed)
        self._has_reset = True
        return self._observation(), {"issuance_rule": self.simulator.issuance_rule}

    def action_to_requests(self, action):
        """Convert continuous shares to integer requests using opening stock."""
        if not self._has_reset or self.simulator.done:
            raise ResetNeeded("Call reset() before acting or after episode termination")
        values = np.asarray(action, dtype=float)
        if values.shape != self.action_space.shape:
            raise ValueError(f"action must have shape {self.action_space.shape}")
        if not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values > 1):
            raise ValueError("action entries must be finite and in [0, 1]")
        scales = np.minimum(1.0, 1.0 / (values.sum(axis=0) + 1e-8))
        shares = values * scales.reshape(1, -1)
        available = np.floor(self.simulator.state[0]).astype(int)
        return np.floor(shares * available.reshape(1, -1)).astype(int)

    def step(self, action):
        requests = self.action_to_requests(action)
        _, reward, done, info = self.simulator.step(requests)
        info["continuous_action"] = np.asarray(action, dtype=float).copy()
        # End of the modeled finite horizon is a true terminal transition.
        return self._observation(), float(reward), bool(done), False, info

    def render(self):
        if self.render_mode is None:
            return None
        if not self._has_reset:
            raise ResetNeeded("Call reset() before render()")
        return (f"Period {self.simulator.current_timestep}/{self.simulator.max_timesteps}\n"
                f"Inventory:\n{self.simulator.state}\n"
                f"Pipeline:\n{self.simulator.allocation_buffer}")

    def close(self):
        pass  # No external resources are held.


if "PerishableInventory-v0" not in registry:
    register(id="PerishableInventory-v0", entry_point="peim_env:PeimEnv")
