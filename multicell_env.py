"""Journal-grade multi-cell NG-RAN slicing environment (Gymnasium API).

J1 design goals
---------------
* Replace the pooled-capacity abstraction with explicit 7-cell dynamics.
* Preserve the original six-dimensional joint action for 3 slices:
  [slice allocation intent alpha scores (S), admission fractions beta (S)].
* Realize the network-level slice intent locally using cell demand/channel state.
* Model inter-cell coupling through load-dependent interference.
* Preserve the original per-slice constrained-RL semantics and SLA budgets.
* Log enough internal state to diagnose dual-variable competition.

This is a system-level management model, not a link-level or full NR protocol stack.
"""
from __future__ import annotations

import copy
import numpy as np
import gymnasium as gym
from gymnasium import spaces


DEFAULT_CFG = dict(
    num_cells=7,
    num_slices=3,                         # 0=eMBB, 1=URLLC, 2=Bursty
    prbs_per_cell=80.0,                   # 560 aggregate PRBs across 7 cells
    episode_len=300,

    # Aggregate offered load retained from the conference model, then split
    # across cells using cell_traffic_weights.
    aggregate_arrival=[190.0, 125.0, 115.0],
    cell_traffic_weights=[1.25, 0.95, 1.05, 0.90, 1.10, 0.85, 0.90],

    # Base service efficiencies by slice. Cell heterogeneity and interference
    # multiply these values at run time.
    eta_slice=[1.0, 0.9, 0.95],
    cell_eta_factor=[1.00, 0.96, 1.04, 0.92, 1.08, 0.94, 1.02],

    # Original queue thresholds are network-level; J1 maps them to each cell.
    q_th_aggregate=[200.0, 80.0, 150.0],
    weights=[1.0, 3.0, 1.5],
    eps=[0.20, 0.05, 0.15],
    reward_weights=dict(tput=1.0, fair=1.0, rej=0.5),

    # Interference: I_b = mean weighted load of neighboring cells, then
    # eta_eff = eta_base / (1 + interference_kappa * I_b).
    interference_kappa=0.18,
    min_eta_factor=0.45,

    # Optional temporally correlated cell shadowing, OFF in J1 pilot.
    channel_fading=False,
    fading_std=0.12,
    fading_corr=0.75,

    # Stress model. Keep configured window immutable; randomized training
    # windows are stored separately per episode.
    stress=False,
    stress_prob=0.0,
    stress_mask=[False, True, True],
    stress_factor=1.6,
    stress_start=100,
    stress_end=180,

    control_mode="joint",                 # joint | admission_only | allocation_only
    constraint_mode="per_slice",          # per_slice | single | none
)


def _seven_cell_adjacency() -> np.ndarray:
    """Center cell 0 plus six-cell ring, symmetric binary adjacency."""
    A = np.zeros((7, 7), dtype=np.float64)
    # center <-> all ring cells
    for j in range(1, 7):
        A[0, j] = A[j, 0] = 1.0
    # ring neighbors
    ring = [1, 2, 3, 4, 5, 6]
    for i, b in enumerate(ring):
        c = ring[(i + 1) % len(ring)]
        A[b, c] = A[c, b] = 1.0
    return A


class MultiCellNGRANSlicingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config=None):
        super().__init__()
        cfg = copy.deepcopy(DEFAULT_CFG)
        if config:
            # shallow override is deliberate for scalar/list experiment config
            cfg.update(config)
        self.cfg = cfg

        self.B = int(cfg["num_cells"])
        self.S = int(cfg["num_slices"])
        if self.B != 7:
            raise ValueError("J1 pilot currently defines an explicit 7-cell topology.")
        if self.S != 3:
            raise ValueError("J1 pilot preserves the original 3-slice / 6-D action design.")

        self.Nb = np.full(self.B, float(cfg["prbs_per_cell"]), dtype=np.float64)
        self.T = int(cfg["episode_len"])

        agg_arr = np.asarray(cfg["aggregate_arrival"], dtype=np.float64)
        tw = np.asarray(cfg["cell_traffic_weights"], dtype=np.float64)
        tw = tw / tw.sum()
        # B x S mean arrivals, with common cell heterogeneity across slices.
        self.base_arrival = tw[:, None] * agg_arr[None, :]

        eta_slice = np.asarray(cfg["eta_slice"], dtype=np.float64)
        cell_eta = np.asarray(cfg["cell_eta_factor"], dtype=np.float64)
        self.eta_base = cell_eta[:, None] * eta_slice[None, :]

        q_agg = np.asarray(cfg["q_th_aggregate"], dtype=np.float64)
        self.Q_th = np.tile((q_agg / self.B)[None, :], (self.B, 1))
        self.w = np.asarray(cfg["weights"], dtype=np.float64)
        self.eps = np.asarray(cfg["eps"], dtype=np.float64)
        self.rw = cfg["reward_weights"]

        self.interference_kappa = float(cfg["interference_kappa"])
        self.min_eta_factor = float(cfg["min_eta_factor"])
        self.A = _seven_cell_adjacency()
        self.degree = np.maximum(self.A.sum(axis=1), 1.0)

        self.channel_fading = bool(cfg["channel_fading"])
        self.fading_std = float(cfg["fading_std"])
        self.fading_corr = float(cfg["fading_corr"])

        self.stress = bool(cfg["stress"])
        self.stress_prob = float(cfg["stress_prob"])
        self.stress_mask = np.asarray(cfg["stress_mask"], dtype=bool)
        self.stress_factor = float(cfg["stress_factor"])
        self.default_stress_start = int(cfg["stress_start"])
        self.default_stress_end = int(cfg["stress_end"])
        self._ep_stress = self.stress
        self._ep_stress_start = self.default_stress_start
        self._ep_stress_end = self.default_stress_end

        self.control_mode = cfg["control_mode"]
        self.constraint_mode = cfg["constraint_mode"]
        self.fixed_alloc = np.ones(self.S, dtype=np.float64) / self.S

        self.lam = np.zeros(self.S, dtype=np.float64)
        self.rng = np.random.default_rng()

        # Observation = q_norm(BS) + arrival_norm(BS) + eta_eff(BS)
        #             + previous cell load(B) + time + stress flag.
        obs_dim = 3 * self.B * self.S + self.B + 2
        self.observation_space = spaces.Box(
            low=0.0, high=1e3, shape=(obs_dim,), dtype=np.float32
        )
        # Preserve original 2S action dimension: 6 for S=3.
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(2 * self.S,), dtype=np.float32
        )
        self._reset_state()

    # ---------- multiplier interface ----------
    def set_lambda(self, lam):
        lam = np.asarray(lam, dtype=np.float64).reshape(-1)
        if lam.shape[0] == 1:
            self.lam = np.full(self.S, lam[0], dtype=np.float64)
        elif lam.shape[0] == self.S:
            self.lam = lam.copy()
        else:
            raise ValueError(f"lambda length must be 1 or {self.S}, got {len(lam)}")

    def get_eps(self):
        return self.eps.copy()

    # ---------- state / dynamics ----------
    def _reset_state(self):
        self.t = 0
        self.Q = np.zeros((self.B, self.S), dtype=np.float64)
        self.last_A = self.base_arrival.copy()
        self.cell_load = np.zeros(self.B, dtype=np.float64)
        self.interference = np.zeros(self.B, dtype=np.float64)
        self._fade = np.ones(self.B, dtype=np.float64)
        self._eta_eff_cache = self.eta_base.copy()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Episode-specific stress variables; never mutate configured defaults.
        self._ep_stress_start = self.default_stress_start
        self._ep_stress_end = self.default_stress_end
        if self.stress:
            self._ep_stress = True
        elif self.stress_prob > 0.0 and self.rng.random() < self.stress_prob:
            self._ep_stress = True
            lo = 40
            hi = max(lo + 1, self.T - 90)
            self._ep_stress_start = int(self.rng.integers(lo, hi))
            self._ep_stress_end = min(
                self.T,
                self._ep_stress_start + int(self.rng.integers(40, 80)),
            )
        else:
            self._ep_stress = False

        self._reset_state()
        return self._obs(), {}

    def _stress_active(self):
        return self._ep_stress and (
            self._ep_stress_start <= self.t < self._ep_stress_end
        )

    def _arrivals(self):
        mult = np.ones(self.S, dtype=np.float64)
        if self._stress_active():
            mult = np.where(self.stress_mask, self.stress_factor, 1.0)
        mean = np.maximum(self.base_arrival * mult[None, :], 1e-9)
        return self.rng.poisson(mean).astype(np.float64)

    def _neighbor_interference(self):
        # weighted mean neighbor load from prior step, avoids an algebraic loop.
        return (self.A @ self.cell_load) / self.degree

    def _effective_eta(self):
        self.interference = self._neighbor_interference()
        interf_factor = 1.0 / (1.0 + self.interference_kappa * self.interference)
        interf_factor = np.clip(interf_factor, self.min_eta_factor, 1.0)

        if self.channel_fading:
            z = self.rng.normal(0.0, self.fading_std, size=self.B)
            self._fade = np.exp(
                self.fading_corr * np.log(self._fade)
                + np.sqrt(1.0 - self.fading_corr**2) * z
            )
            self._fade = np.clip(self._fade, 0.65, 1.35)
        else:
            self._fade[:] = 1.0

        eta_eff = self.eta_base * interf_factor[:, None] * self._fade[:, None]
        self._eta_eff_cache = eta_eff
        return eta_eff

    def _obs(self):
        q_norm = self.Q / (self.Q_th + 1e-6)
        # normalize each slice by its aggregate mean / B to preserve scale.
        a_scale = np.maximum(self.base_arrival.mean(axis=0), 1e-6)
        a_norm = self.last_A / a_scale[None, :]
        eta_norm = self._eta_eff_cache / (np.max(self.eta_base) + 1e-6)
        stress_flag = 1.0 if self._stress_active() else 0.0
        obs = np.concatenate([
            q_norm.ravel(),
            a_norm.ravel(),
            eta_norm.ravel(),
            self.cell_load,
            np.asarray([self.t / self.T, stress_flag]),
        ])
        return np.clip(obs, 0.0, 1e3).astype(np.float32)

    def _decode_action(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), 0.0, 1.0)
        raw_alloc = action[: self.S]
        beta = action[self.S :]
        s = raw_alloc.sum()
        alpha_global = (
            raw_alloc / s if s > 1e-6 else np.ones(self.S, dtype=np.float64) / self.S
        )
        if self.control_mode == "admission_only":
            alpha_global = self.fixed_alloc.copy()
        elif self.control_mode == "allocation_only":
            beta = np.ones(self.S, dtype=np.float64)
        return alpha_global, beta

    def _localize_allocation(self, alpha_global, admitted_demand, eta_eff):
        """Map global slice intent to cell-specific slice shares.

        The policy still controls only S allocation intents. Within each cell,
        allocation is demand- and channel-aware, preventing the 6->42 action
        explosion in the first multi-cell experiment.
        """
        # Required-resource proxy: queue+admitted arrivals divided by service
        # efficiency. This exposes local congestion/channel differences.
        req_prb = admitted_demand / np.maximum(eta_eff, 1e-6)
        scores = alpha_global[None, :] * (req_prb + 1e-3)
        denom = scores.sum(axis=1, keepdims=True)
        alpha_local = np.divide(
            scores,
            denom,
            out=np.tile((np.ones(self.S) / self.S)[None, :], (self.B, 1)),
            where=denom > 1e-9,
        )
        return alpha_local

    def step(self, action):
        alpha_global, beta = self._decode_action(action)
        A = self._arrivals()
        A_adm = beta[None, :] * A
        eta_eff = self._effective_eta()

        demand = self.Q + A_adm
        alpha_local = self._localize_allocation(alpha_global, demand, eta_eff)
        mu = alpha_local * self.Nb[:, None] * eta_eff
        served = np.minimum(demand, mu)
        self.Q = np.maximum(demand - mu, 0.0)

        throughput = float(served.sum())
        rejection = float(((1.0 - beta[None, :]) * A).sum())

        # Cell load is the PRB demand ratio before clipping to capacity.
        required_prb = (demand / np.maximum(eta_eff, 1e-6)).sum(axis=1)
        self.cell_load = np.clip(required_prb / self.Nb, 0.0, 1.0)

        viol_cell = np.clip(
            (self.Q - self.Q_th) / (self.Q_th + 1e-6), 0.0, 1.0
        )
        # Per-slice management cost = mean cell violation * slice priority.
        c_s = self.w * viol_cell.mean(axis=0)
        C_weighted = float(c_s.sum() / self.w.sum())

        served_slice = served.sum(axis=0)
        jain = float(
            (served_slice.sum() ** 2)
            / (self.S * (served_slice ** 2).sum() + 1e-9)
        )

        capacity_norm = float(np.sum(self.Nb) * np.max(self.eta_base))
        tput_norm = throughput / (capacity_norm + 1e-9)
        rej_norm = rejection / (np.sum(self.base_arrival) + 1e-9)
        base_reward = (
            self.rw["tput"] * tput_norm
            + self.rw["fair"] * jain
            - self.rw["rej"] * rej_norm
        )
        if self.constraint_mode == "per_slice":
            penalty = float(np.dot(self.lam, c_s))
        elif self.constraint_mode == "single":
            penalty = float(self.lam.mean() * C_weighted)
        else:
            penalty = 0.0
        reward = base_reward - penalty

        # Record current-step diagnostics before advancing time.
        info = dict(
            throughput=throughput,
            rejection=rejection,
            sla_cost=C_weighted,
            cost_per_slice=c_s.copy(),
            jain=jain,
            stress=self._stress_active(),
            lambda_per_slice=self.lam.copy(),
            queue_per_cell=self.Q.copy(),
            queue_per_slice=self.Q.mean(axis=0),
            arrivals_per_cell=A.copy(),
            arrivals_per_slice=A.sum(axis=0),
            admission=beta.copy(),
            alpha_global=alpha_global.copy(),
            alpha_per_cell=alpha_local.copy(),
            prb_share_per_slice=alpha_local.mean(axis=0),
            cell_load=self.cell_load.copy(),
            interference=self.interference.copy(),
            eta_eff=eta_eff.copy(),
            episode_stress_start=self._ep_stress_start,
            episode_stress_end=self._ep_stress_end,
        )

        self.last_A = A
        self.t += 1
        terminated = False
        truncated = self.t >= self.T
        return self._obs(), float(reward), terminated, truncated, info
