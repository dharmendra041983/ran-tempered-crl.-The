"""CMARS-inspired sequential cooperative PPO baseline for CMARS baseline.

This is an *adaptation*, not an exact reproduction of Zangooei et al. CMARS.
The original paper uses cooperative multi-agent PPO-Lagrangian for sequential
inter-slice PRB allocation, shuffled agent order, local + aggregate state, and
constraint-aware training. This environment has a different system model
and jointly controls admission and resource allocation.

To obtain a fair common-environment baseline, this wrapper preserves the main
CMARS structural ideas while adapting each slice-agent action to:
    [fraction_of_remaining_resource_intent, admission_fraction]

A single parameter-shared PPO policy is invoked sequentially for all slice
agents. Slice identity, SLA budget, local queue/demand/channel summaries,
remaining resource intent, and global network observations are included in the
state. Agent order is shuffled at every macro decision.

One base-environment macro step requires S agent turns. Intermediate substeps
receive zero reward. On the final turn, the assembled 2S action is applied to
the exact same 7-cell base environment used by the proposed method.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from multicell_env import MultiCellNGRANSlicingEnv


class CMARSSequentialWrapper(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, base_config=None, raw_lambda_lr=0.02, lambda_update_macro=3000):
        super().__init__()
        cfg = dict(base_config or {})
        # Reward shaping is implemented here so the baseline has its own
        # constraint logic while the underlying dynamics remain identical.
        cfg["constraint_mode"] = "none"
        self.base = MultiCellNGRANSlicingEnv(cfg)
        self.B = self.base.B
        self.S = self.base.S

        self.lambda_lr = float(raw_lambda_lr)
        self.lambda_update_macro = int(lambda_update_macro)
        self.lam = np.zeros(self.S, dtype=np.float64)
        self._cost_buf = []
        self._macro_since_dual = 0

        # Shared actor action: allocate a fraction of remaining intent and choose
        # an admission fraction for the currently active slice.
        self.action_space = spaces.Box(
            low=np.zeros(2, dtype=np.float32),
            high=np.ones(2, dtype=np.float32),
            dtype=np.float32,
        )

        # Observation:
        # base global observation
        # + slice one-hot(S)
        # + [eps_s, weight_s, remaining_share, turn_fraction]
        # + local slice summaries [q_norm, arrival_norm, eta_norm, load_mean]
        obs_dim = self.base.observation_space.shape[0] + self.S + 8
        self.observation_space = spaces.Box(
            low=0.0, high=1e3, shape=(obs_dim,), dtype=np.float32
        )

        self.order = np.arange(self.S, dtype=int)
        self.turn = 0
        self.remaining = 1.0
        self.alloc_share = np.zeros(self.S, dtype=np.float64)
        self.admission = np.ones(self.S, dtype=np.float64)
        self._base_obs = None
        self.last_macro_info = None
        self.macro_steps = 0

    def _new_order(self):
        self.order = self.base.rng.permutation(self.S)
        self.turn = 0
        self.remaining = 1.0
        self.alloc_share[:] = 0.0
        self.admission[:] = 1.0

    def _agent_obs(self):
        s = int(self.order[self.turn])
        onehot = np.zeros(self.S, dtype=np.float64)
        onehot[s] = 1.0

        q_norm = float(np.mean(self.base.Q[:, s] / (self.base.Q_th[:, s] + 1e-6)))
        a_scale = max(float(self.base.base_arrival[:, s].mean()), 1e-6)
        a_norm = float(np.mean(self.base.last_A[:, s]) / a_scale)
        eta_norm = float(
            np.mean(self.base._eta_eff_cache[:, s])
            / (np.max(self.base.eta_base) + 1e-6)
        )
        load_mean = float(np.mean(self.base.cell_load))

        extra = np.asarray([
            self.base.eps[s],
            self.base.w[s] / max(float(np.max(self.base.w)), 1e-6),
            self.remaining,
            self.turn / max(self.S - 1, 1),
            q_norm,
            a_norm,
            eta_norm,
            load_mean,
        ], dtype=np.float64)

        out = np.concatenate([self._base_obs, onehot, extra])
        return np.clip(out, 0.0, 1e3).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        # Preserve the dual state across episode boundaries. Episodes are only
        # 300 macro steps, while the dual update interval is 3000 macro steps.
        # Resetting lambda/cost accumulation here would prevent all dual updates.
        self.last_macro_info = None
        self._base_obs, info = self.base.reset(seed=seed, options=options)
        self._new_order()
        return self._agent_obs(), info

    def _update_dual_if_needed(self):
        if self._macro_since_dual < self.lambda_update_macro or not self._cost_buf:
            return None
        mean_cost = np.mean(np.stack(self._cost_buf), axis=0)
        residual = mean_cost - self.base.eps
        dual_step = self.lambda_lr * residual
        self.lam = np.maximum(0.0, self.lam + dual_step)
        rec = {
            "macro_step": self.macro_steps,
            "lambda": self.lam.copy(),
            "mean_cost": mean_cost.copy(),
            "residual": residual.copy(),
            "dual_step": dual_step.copy(),
        }
        self._cost_buf = []
        self._macro_since_dual = 0
        return rec

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), 0.0, 1.0)
        s = int(self.order[self.turn])

        # Sequential sharing: each non-final agent takes a fraction of the
        # currently remaining resource intent. The final agent receives the
        # remainder so the base environment sees a complete allocation vector.
        if self.turn < self.S - 1:
            take = float(a[0]) * self.remaining
        else:
            take = self.remaining

        self.alloc_share[s] = max(take, 0.0)
        self.remaining = max(0.0, self.remaining - take)
        self.admission[s] = float(a[1])

        # Intermediate agent turn: no network transition yet.
        if self.turn < self.S - 1:
            self.turn += 1
            info = {
                "macro_step_complete": False,
                "active_slice": s,
                "agent_order": self.order.copy(),
                "remaining_share": self.remaining,
            }
            return self._agent_obs(), 0.0, False, False, info

        # Final turn -> assemble action and advance exact same base dynamics.
        alloc = self.alloc_share.copy()
        if alloc.sum() <= 1e-9:
            alloc[:] = 1.0 / self.S
        else:
            alloc /= alloc.sum()
        joint_action = np.concatenate([alloc, self.admission])

        self._base_obs, base_reward, term, trunc, info = self.base.step(joint_action)
        c_s = np.asarray(info["cost_per_slice"], dtype=np.float64)

        # Cooperative constrained reward: every agent shares the same network
        # objective, with raw per-slice Lagrangian penalties. This deliberately
        # avoids using the proposed tempered update in the prior-work baseline.
        shaped_reward = float(base_reward - np.dot(self.lam, c_s))

        self._cost_buf.append(c_s)
        self._macro_since_dual += 1
        self.macro_steps += 1
        dual_rec = self._update_dual_if_needed()

        info = dict(info)
        info.update({
            "macro_step_complete": True,
            "agent_order": self.order.copy(),
            "cmars_lambda": self.lam.copy(),
            "cmars_alloc": alloc.copy(),
            "cmars_admission": self.admission.copy(),
            "dual_update_record": dual_rec,
        })
        self.last_macro_info = info

        if term or trunc:
            # Return a valid observation even though the episode is ending.
            next_obs = self._agent_obs()
            return next_obs, shaped_reward, term, trunc, info

        self._new_order()
        return self._agent_obs(), shaped_reward, False, False, info
