"""Heuristics and primal-dual callback for the J1 multi-cell experiment."""
from __future__ import annotations

import csv
from pathlib import Path
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


def _aggregate_demand(env):
    return (env.last_A + env.Q).sum(axis=0)


def static_policy(env, obs=None):
    S = env.S
    return np.concatenate([np.ones(S) / S, np.ones(S)])


def proportional_policy(env, obs=None):
    S = env.S
    demand = _aggregate_demand(env)
    alpha = demand / demand.sum() if demand.sum() > 1e-6 else np.ones(S) / S
    return np.concatenate([alpha, np.ones(S)])


def threshold_policy(env, obs=None):
    S = env.S
    demand = _aggregate_demand(env)
    alpha = demand / demand.sum() if demand.sum() > 1e-6 else np.ones(S) / S
    # Network-level admission intent from fraction of cells above threshold.
    congested_fraction = (env.Q > env.Q_th).mean(axis=0)
    beta = np.where(congested_fraction >= 0.5, 0.0, 1.0)
    return np.concatenate([alpha, beta])


HEURISTICS = {
    "Static": static_policy,
    "Proportional": proportional_policy,
    "Threshold": threshold_policy,
}


class DiagnosticLagrangianCallback(BaseCallback):
    """Projected dual ascent plus dual-update dynamics logging.

    One row is written per multiplier update. Window-mean queues, admissions,
    resource shares, costs, cell loads, and interference are stored alongside
    lambda and budget residuals so the final study can test the hypothesized
    dual-variable competition mechanism.
    """

    def __init__(
        self,
        eps,
        lr_lambda=0.02,
        update_every=3000,
        constraint_mode="per_slice",
        dual_update_mode="raw",
        normalized_lr_scale="mean_budget",
        normalization_alpha=1.0,
        eps_floor=1e-6,
        lambda_cap=None,
        log_path=None,
        verbose=0,
    ):
        super().__init__(verbose)
        self.eps = np.asarray(eps, dtype=np.float64)
        self.S = len(self.eps)
        self.lr_lambda = float(lr_lambda)
        self.update_every = int(update_every)
        self.constraint_mode = constraint_mode
        self.dual_update_mode = str(dual_update_mode)
        if self.dual_update_mode not in {"raw", "budget_normalized", "tempered"}:
            raise ValueError("dual_update_mode must be raw, budget_normalized, or tempered")
        self.normalized_lr_scale = normalized_lr_scale
        self.normalization_alpha = float(normalization_alpha)
        self.eps_floor = float(eps_floor)
        self.lambda_cap = None if lambda_cap is None else float(lambda_cap)
        self.lam = (
            np.zeros(self.S, dtype=np.float64)
            if constraint_mode == "per_slice"
            else np.zeros(1, dtype=np.float64)
        )
        self.log_path = Path(log_path) if log_path else None
        self._buf = []
        self._since = 0
        self.records = []

    def _push(self):
        self.training_env.env_method("set_lambda", self.lam)

    def _on_training_start(self):
        self._push()

    @staticmethod
    def _mean_array(infos, key):
        vals = [np.asarray(x[key], dtype=np.float64) for x in infos if key in x]
        return np.mean(np.stack(vals), axis=0) if vals else None

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "cost_per_slice" in info:
                self._buf.append(info)
        self._since += 1

        if self._since >= self.update_every and self._buf:
            mean_cost = self._mean_array(self._buf, "cost_per_slice")
            residual = mean_cost - self.eps

            den = np.maximum(self.eps, self.eps_floor)
            normalized_residual = residual / den
            tempered_residual = residual / np.power(den, self.normalization_alpha)

            if self.constraint_mode == "per_slice":
                if self.dual_update_mode == "raw":
                    dual_step = self.lr_lambda * residual
                elif self.dual_update_mode == "budget_normalized":
                    if self.normalized_lr_scale == "mean_budget":
                        eta_norm = self.lr_lambda * float(np.mean(self.eps))
                    else:
                        eta_norm = self.lr_lambda
                    dual_step = eta_norm * normalized_residual
                else:  # tempered
                    if self.normalized_lr_scale == "mean_budget":
                        eta_alpha = (
                            self.lr_lambda
                            * float(np.mean(self.eps)) ** self.normalization_alpha
                        )
                    else:
                        eta_alpha = self.lr_lambda
                    dual_step = eta_alpha * tempered_residual

                self.lam = np.maximum(0.0, self.lam + dual_step)
                if self.lambda_cap is not None:
                    self.lam = np.minimum(self.lam, self.lambda_cap)
            elif self.constraint_mode == "single":
                agg = float(mean_cost.mean())
                dual_step = np.asarray([
                    self.lr_lambda * (agg - float(self.eps.mean()))
                ], dtype=np.float64)
                self.lam = np.maximum(0.0, self.lam + dual_step)
                if self.lambda_cap is not None:
                    self.lam = np.minimum(self.lam, self.lambda_cap)
            else:
                dual_step = np.zeros_like(self.lam)

            self._push()

            q = self._mean_array(self._buf, "queue_per_slice")
            admission = self._mean_array(self._buf, "admission")
            prb = self._mean_array(self._buf, "prb_share_per_slice")
            loads = self._mean_array(self._buf, "cell_load")
            interf = self._mean_array(self._buf, "interference")

            lam_full = (
                self.lam
                if self.lam.size == self.S
                else np.repeat(self.lam[0], self.S)
            )
            rec = {
                "step": int(self.num_timesteps),
                "lambda_embb": float(lam_full[0]),
                "lambda_urllc": float(lam_full[1]),
                "lambda_bursty": float(lam_full[2]),
                "cost_embb": float(mean_cost[0]),
                "cost_urllc": float(mean_cost[1]),
                "cost_bursty": float(mean_cost[2]),
                "residual_embb": float(residual[0]),
                "residual_urllc": float(residual[1]),
                "residual_bursty": float(residual[2]),
                "normalized_residual_embb": float(normalized_residual[0]),
                "normalized_residual_urllc": float(normalized_residual[1]),
                "normalized_residual_bursty": float(normalized_residual[2]),
                "tempered_residual_embb": float(tempered_residual[0]),
                "tempered_residual_urllc": float(tempered_residual[1]),
                "tempered_residual_bursty": float(tempered_residual[2]),
                "dual_step_embb": float(dual_step[0] if len(dual_step) > 1 else dual_step[0]),
                "dual_step_urllc": float(dual_step[1] if len(dual_step) > 1 else dual_step[0]),
                "dual_step_bursty": float(dual_step[2] if len(dual_step) > 1 else dual_step[0]),
                "queue_embb": float(q[0]),
                "queue_urllc": float(q[1]),
                "queue_bursty": float(q[2]),
                "admit_embb": float(admission[0]),
                "admit_urllc": float(admission[1]),
                "admit_bursty": float(admission[2]),
                "prb_embb": float(prb[0]),
                "prb_urllc": float(prb[1]),
                "prb_bursty": float(prb[2]),
                "mean_cell_load": float(np.mean(loads)),
                "max_cell_load": float(np.max(loads)),
                "mean_interference": float(np.mean(interf)),
                "max_interference": float(np.max(interf)),
            }
            for b, v in enumerate(loads):
                rec[f"cell{b}_load"] = float(v)
            for b, v in enumerate(interf):
                rec[f"cell{b}_interference"] = float(v)
            self.records.append(rec)

            if self.verbose:
                print(
                    f"[lambda] step={self.num_timesteps} "
                    f"lam={np.round(lam_full, 3)} "
                    f"cost={np.round(mean_cost, 3)}"
                )
            self._buf = []
            self._since = 0
        return True

    def _on_training_end(self):
        if self.log_path and self.records:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(self.records[0].keys()))
                writer.writeheader()
                writer.writerows(self.records)
