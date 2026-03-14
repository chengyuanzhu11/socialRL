from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import random
from collections import deque, defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import argparse
import os
import math

try:
    from torch.amp import autocast as torch_autocast
    from torch.amp import GradScaler as TorchGradScaler

    def make_autocast(device_type: str):
        return torch_autocast(device_type=device_type, enabled=(device_type == "cuda"))

    def make_grad_scaler(device_type: str):
        return TorchGradScaler(device=device_type, enabled=(device_type == "cuda"))

except ImportError:
    from torch.cuda.amp import autocast as torch_autocast
    from torch.cuda.amp import GradScaler as TorchGradScaler

    def make_autocast(device_type: str):
        return torch_autocast(enabled=(device_type == "cuda"))

    def make_grad_scaler(device_type: str):
        return TorchGradScaler(enabled=(device_type == "cuda"))

# -------------------------
# Seed / utils
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))

def clamp_float(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))

def ensure_dir(path: str):
    if path is None or path == "":
        return
    os.makedirs(path, exist_ok=True)

# -------------------------
# Actions
# -------------------------
A_LEFT = 0
A_STAY = 1
A_RIGHT = 2
A_OBS = 3  # consumes time; reveals social timing info stochastically
A_EAT = 4

# ============================================================
# Environment
# ============================================================
class SocialLickEnv1D:
    """
    Fast but detailed (paper-aligned):
      dt_s = 0.5s/step
      teacher cue bursts ("lick bouts") with stochastic duration (mean ~ 1s)
      reward window: 3s after last cue step

    Observation gating:
      - social signal (cue + noisy window remaining) only appears when action==OBS
      - detection during cue burst is probabilistic per OBS step

    Condition knobs:
      - social_visibility: 0.0 => "social blind" (OBS yields no social info)
      - detect_memory_steps: if set, detected_bout_id expires after N steps
      - familiarity_enabled: if True, a latent familiarity variable accumulates across episodes and
        increases effective detect prob and reduces window-remaining noise

    NEW (requested) familiarity effects (ONLY active when familiarity_enabled=True):
      - reduces observe_cost as familiarity increases (quicker/less costly observing)
      - boosts attend_bonus as familiarity increases (more quick observing)
      - adds movement shaping during open window after detection (go to eat after observing)
      - discourages OBS during open window after detection (stop observing; move/eat)
      - adds small "earlier eat" bonus on valid reward (shorter response latency)

    Empathy knob (separate from familiarity):
      - empathy_enabled: gives intrinsic reward when OBS successfully sees the lick cue
        (lets you compare empathy on/off without touching palatability).
    """
    def __init__(
        self,
        grid_size=15,
        dt_s=0.5,
        max_time_s=120.0,
        teacher_period_s=30.0,
        teacher_jitter_s=6.0,
        lick_mean_s=1.0,
        lick_dist="gamma",  # "gamma" or "lognormal"
        lick_cv=0.8,        # for lognormal
        window_s=3.0,
        eat_cooldown_s=2.0,
        observe_cost=0.002,
        eat_cost=0.005,
        attend_bonus=0.004,
        p_detect_per_obs=0.70,
        win_rem_noise_steps=1,

        # perception / cognition constraints
        social_visibility: float = 1.0,             # 1.0 normal, 0.0 blind
        detect_memory_steps: Optional[int] = None,  # None => perfect retention

        # familiarity dynamics
        familiarity_enabled: bool = False,
        familiarity_init: float = 0.0,              # persistent across episodes
        fam_decay_ep: float = 0.00,                 # applied at reset (episode boundary)
        fam_gain_obs: float = 0.00,                 # gain per OBS during teacher lick
        fam_gain_detect: float = 0.00,              # extra gain when OBS successfully detects lick
        p_detect_fam_boost: float = 0.00,           # added detect prob at fam=1
        noise_fam_reduction: float = 0.00,          # fraction reduction of noise_steps at fam=1

        # familiarity -> behavior shaping (ONLY when familiarity_enabled=True)
        fam_obs_cost_reduction: float = 0.50,       # at fam=1, OBS cost *= (1-0.50)
        fam_attend_bonus_boost: float = 0.50,       # at fam=1, attend_bonus *= (1+0.50)
        fam_move_shaping: float = 0.010,            # per-step reward * fam for moving closer to food (during detected window)
        fam_obs_during_window_penalty: float = 0.002,  # penalty * fam for OBS during detected open window
        fam_latency_bonus: float = 0.20,            # extra reward on valid eat: + fam * bonus * (remaining/window)

        # empathy (separate condition)
        empathy_enabled: bool = False,
        empathy_seen_reward: float = 0.05,          # intrinsic reward when OBS sees lick (seen_lick==1)
    ):
        self.grid_size = int(grid_size)
        self.dt_s = float(dt_s)
        self.max_steps = int(round(max_time_s / self.dt_s))
        self.teacher_period_steps = int(round(teacher_period_s / self.dt_s))
        self.teacher_jitter_steps = int(round(teacher_jitter_s / self.dt_s))
        self.lick_mean_s = float(lick_mean_s)
        self.lick_dist = str(lick_dist)
        self.lick_cv = float(lick_cv)
        self.window_steps = int(round(window_s / self.dt_s))
        self.eat_cooldown_steps = int(round(eat_cooldown_s / self.dt_s))
        self.observe_cost = float(observe_cost)
        self.eat_cost = float(eat_cost)
        self.attend_bonus = float(attend_bonus)
        self.p_detect_per_obs = float(p_detect_per_obs)
        self.win_rem_noise_steps = int(win_rem_noise_steps)

        self.social_visibility = clamp_float(social_visibility, 0.0, 1.0)
        self.detect_memory_steps = None if detect_memory_steps is None else int(detect_memory_steps)

        self.familiarity_enabled = bool(familiarity_enabled)
        self.familiarity = clamp_float(familiarity_init, 0.0, 1.0)
        self.fam_decay_ep = clamp_float(fam_decay_ep, 0.0, 1.0)
        self.fam_gain_obs = float(fam_gain_obs)
        self.fam_gain_detect = float(fam_gain_detect)
        self.p_detect_fam_boost = float(p_detect_fam_boost)
        self.noise_fam_reduction = clamp_float(noise_fam_reduction, 0.0, 1.0)

        # shaping params (only matter if familiarity_enabled)
        self.fam_obs_cost_reduction = clamp_float(fam_obs_cost_reduction, 0.0, 1.0)
        self.fam_attend_bonus_boost = max(0.0, float(fam_attend_bonus_boost))
        self.fam_move_shaping = float(fam_move_shaping)
        self.fam_obs_during_window_penalty = float(fam_obs_during_window_penalty)
        self.fam_latency_bonus = float(fam_latency_bonus)

        # empathy params
        self.empathy_enabled = bool(empathy_enabled)
        self.empathy_seen_reward = float(empathy_seen_reward)

        assert self.teacher_period_steps > self.window_steps + 1
        assert self.max_steps > self.teacher_period_steps + 5
        self.reset()

    def _sample_lick_duration_steps(self):
        if self.lick_dist == "gamma":
            shape = 2.0
            scale = self.lick_mean_s / shape
            dur_s = float(np.random.gamma(shape, scale))
        elif self.lick_dist == "lognormal":
            cv = max(1e-6, self.lick_cv)
            sigma2 = np.log(1.0 + cv**2)
            sigma = np.sqrt(sigma2)
            mu = np.log(self.lick_mean_s) - 0.5 * sigma2
            dur_s = float(np.random.lognormal(mean=mu, sigma=sigma))
        else:
            raise ValueError(f"Unknown lick_dist={self.lick_dist}")
        steps = int(np.round(dur_s / self.dt_s))
        return max(1, steps), dur_s

    def _effective_detect_prob(self) -> float:
        p = self.p_detect_per_obs
        if self.familiarity_enabled:
            p += self.p_detect_fam_boost * self.familiarity
        return clamp_float(p, 0.0, 1.0)

    def _effective_noise_steps(self) -> int:
        n = int(self.win_rem_noise_steps)
        if self.familiarity_enabled and n > 0:
            n = int(round(n * (1.0 - self.noise_fam_reduction * self.familiarity)))
        return max(0, n)

    def _effective_observe_cost(self) -> float:
        if not self.familiarity_enabled:
            return self.observe_cost
        # cost reduces as familiarity increases
        return float(self.observe_cost * (1.0 - self.fam_obs_cost_reduction * self.familiarity))

    def _effective_attend_bonus(self) -> float:
        if not self.familiarity_enabled:
            return self.attend_bonus
        return float(self.attend_bonus * (1.0 + self.fam_attend_bonus_boost * self.familiarity))

    def reset(self):
        # Episode boundary familiarity decay (persistent across episodes)
        if self.familiarity_enabled and self.fam_decay_ep > 0.0:
            self.familiarity = clamp_float(self.familiarity * (1.0 - self.fam_decay_ep), 0.0, 1.0)

        self.t = 0
        self.teacher_pos = int(np.random.randint(0, self.grid_size))
        candidates = [i for i in range(self.grid_size) if i != self.teacher_pos]
        self.learner_food_pos = int(np.random.choice(candidates))
        self.learner_pos = int(np.random.randint(0, self.grid_size))

        self.eat_cd = 0
        self.last_lick_step = None
        self.detected_bout_id = None
        self._detect_ttl = None
        self.rewarded_bout_ids = set()

        # build cue-burst timeline
        self.bout_id_at_step = -np.ones(self.max_steps + 1, dtype=np.int32)
        self.bout_start_steps = []
        self.bout_end_steps = []
        self.bout_dur_s_list = []
        bout_id = 0
        k = 1
        while True:
            nominal = k * self.teacher_period_steps
            if nominal >= self.max_steps:
                break
            jitter = int(np.random.randint(-self.teacher_jitter_steps, self.teacher_jitter_steps + 1))
            start = clamp_int(nominal + jitter, 1, self.max_steps - 2)
            dur_steps, dur_s = self._sample_lick_duration_steps()
            end = clamp_int(start + dur_steps - 1, start, self.max_steps - 1)
            self.bout_id_at_step[start:end + 1] = bout_id
            self.bout_start_steps.append(start)
            self.bout_end_steps.append(end)
            self.bout_dur_s_list.append(dur_s)
            bout_id += 1
            k += 1
        self.n_bouts = int(bout_id)
        return self._get_obs(lick_sig=0.0, win_rem=0.0)

    def _teacher_lick_now(self, t: int):
        bid = int(self.bout_id_at_step[t]) if (0 <= t <= self.max_steps) else -1
        return (1, bid) if bid >= 0 else (0, -1)

    def _window_open(self, t: int) -> int:
        if self.last_lick_step is None:
            return 0
        dt = t - self.last_lick_step
        return 1 if (0 <= dt < self.window_steps) else 0

    def _window_remaining(self, t: int) -> int:
        if self.last_lick_step is None:
            return 0
        dt = t - self.last_lick_step
        rem = self.window_steps - dt
        return int(max(0, rem))

    def _phase_to_next_nominal(self, t: int) -> int:
        mod = t % self.teacher_period_steps
        return self.teacher_period_steps - mod

    def _get_obs(self, lick_sig: float, win_rem: float):
        lp = float(self.learner_pos) / float(self.grid_size - 1)
        lf = float(self.learner_food_pos) / float(self.grid_size - 1)
        phase = float(self._phase_to_next_nominal(self.t)) / float(self.teacher_period_steps)
        cdn = float(self.eat_cd) / float(max(1, self.eat_cooldown_steps))
        # [pos, food_pos, gated cue, gated window remaining, phase clock, cooldown]
        return np.array([lp, lf, float(lick_sig), float(win_rem), phase, cdn], dtype=np.float32)

    def step(self, action: int):
        self.t += 1
        teacher_lick, bout_id = self._teacher_lick_now(self.t)
        if teacher_lick == 1:
            self.last_lick_step = self.t

        window_open = self._window_open(self.t)
        win_rem_true = self._window_remaining(self.t)

        if self.eat_cd > 0:
            self.eat_cd -= 1

        # optional memory decay for detected bout id
        if self.detect_memory_steps is not None and self._detect_ttl is not None:
            self._detect_ttl -= 1
            if self._detect_ttl <= 0:
                self._detect_ttl = None
                self.detected_bout_id = None

        lick_sig = 0.0
        win_rem = 0.0
        did_observe = 0
        did_eat = 0
        seen_lick = 0

        # distance for shaping (computed before move)
        dist_before = abs(int(self.learner_pos) - int(self.learner_food_pos))

        # movement / observe / eat
        if action == A_LEFT:
            self.learner_pos -= 1
        elif action == A_RIGHT:
            self.learner_pos += 1
        elif action == A_OBS:
            did_observe = 1

            # social blind => OBS reveals nothing
            if self.social_visibility > 0.0:
                p_det = self._effective_detect_prob()
                if teacher_lick == 1 and bout_id >= 0 and (np.random.rand() < p_det):
                    seen_lick = 1
                    self.detected_bout_id = int(bout_id)
                    if self.detect_memory_steps is not None:
                        self._detect_ttl = int(self.detect_memory_steps)

                lick_sig = float(seen_lick)

                if win_rem_true > 0:
                    noise_steps = self._effective_noise_steps()
                    noise = int(np.random.randint(-noise_steps, noise_steps + 1)) if noise_steps > 0 else 0
                    est = clamp_int(win_rem_true + noise, 0, self.window_steps)
                    win_rem = float(est) / float(max(1, self.window_steps))

                # familiarity update
                if self.familiarity_enabled and teacher_lick == 1 and bout_id >= 0:
                    self.familiarity = clamp_float(self.familiarity + self.fam_gain_obs, 0.0, 1.0)
                    if seen_lick == 1:
                        self.familiarity = clamp_float(self.familiarity + self.fam_gain_detect, 0.0, 1.0)

        elif action == A_EAT:
            did_eat = 1
        # A_STAY does nothing

        self.learner_pos = clamp_int(self.learner_pos, 0, self.grid_size - 1)
        dist_after = abs(int(self.learner_pos) - int(self.learner_food_pos))

        # base costs (familiarity can reduce OBS cost)
        reward = 0.0
        reward -= self._effective_observe_cost() * float(did_observe)
        reward -= self.eat_cost * float(did_eat)

        # shaping: reward OBS during cue only if bout not yet detected
        # (kept exactly as your template, except attend_bonus can be boosted by familiarity)
        if did_observe == 1 and teacher_lick == 1 and bout_id >= 0:
            if self.detected_bout_id != int(bout_id):
                reward += self._effective_attend_bonus()

        # empathy: intrinsic reward when OBS successfully sees lick
        if self.empathy_enabled and did_observe == 1 and seen_lick == 1:
            reward += self.empathy_seen_reward

        # eat allowed+/-
        eat_allowed = (did_eat == 1 and self.eat_cd == 0)
        if eat_allowed:
            self.eat_cd = self.eat_cooldown_steps

        # success logic (1 reward per bout)
        eat_valid = 0
        latency_steps = None
        last_bid = -1
        if self.last_lick_step is not None and self.last_lick_step <= self.max_steps:
            last_bid = int(self.bout_id_at_step[self.last_lick_step])

        if eat_allowed and (self.learner_pos == self.learner_food_pos) and (window_open == 1) and (self.last_lick_step is not None):
            if (last_bid >= 0) and (self.detected_bout_id == last_bid) and (last_bid not in self.rewarded_bout_ids):
                eat_valid = 1
                self.rewarded_bout_ids.add(last_bid)
                reward += 1.0
                latency_steps = int(self.t - int(self.last_lick_step))

                # familiarity: reward earlier eating (shorter latency)
                if self.familiarity_enabled and self.fam_latency_bonus > 0.0:
                    # earlier in window => larger win_rem_true
                    reward += float(self.familiarity * self.fam_latency_bonus * (float(win_rem_true) / float(max(1, self.window_steps))))

        # familiarity: push "go to eat after observing" ONLY when a detected bout window is open
        if self.familiarity_enabled and (last_bid >= 0) and (window_open == 1) and (self.detected_bout_id == last_bid) and (last_bid not in self.rewarded_bout_ids):
            fam = float(self.familiarity)
            if fam > 0.0:
                # reward moving closer to food (shorter response latency)
                if action in (A_LEFT, A_RIGHT) and self.fam_move_shaping != 0.0:
                    # dist_before - dist_after: +1 if moved closer, -1 if moved away
                    reward += float(self.fam_move_shaping * fam * float(dist_before - dist_after))
                # penalize continued observing during the open window (stop observing, act!)
                if did_observe == 1 and self.fam_obs_during_window_penalty > 0.0:
                    reward -= float(self.fam_obs_during_window_penalty * fam)

        done = bool(self.t >= self.max_steps)
        info = {
            "t": int(self.t),
            "teacher_lick": int(teacher_lick),
            "bout_id": int(bout_id),
            "window_open": int(window_open),
            "win_rem_true": int(win_rem_true),
            "action": int(action),
            "observe": int(did_observe),
            "eat": int(did_eat),
            "eat_allowed": int(eat_allowed),
            "eat_valid": int(eat_valid),
            "latency_steps": -1 if latency_steps is None else int(latency_steps),
            "last_lick_step": -1 if self.last_lick_step is None else int(self.last_lick_step),
            "learner_pos": int(self.learner_pos),
            "food_pos": int(self.learner_food_pos),
            "seen_lick": int(seen_lick),
            "win_rem_est": float(win_rem),
            "detected_bout_id": -1 if self.detected_bout_id is None else int(self.detected_bout_id),
            "rewarded_bouts": len(self.rewarded_bout_ids),
            "n_bouts": int(self.n_bouts),
            "dt_s": float(self.dt_s),
            "window_steps": int(self.window_steps),
            "social_visibility": float(self.social_visibility),
            "familiarity": float(self.familiarity),
            "p_detect_eff": float(self._effective_detect_prob()) if (self.social_visibility > 0.0) else 0.0,
            "obs_cost_eff": float(self._effective_observe_cost()),
            "attend_bonus_eff": float(self._effective_attend_bonus()),
            "empathy_enabled": int(self.empathy_enabled),
        }
        obs = self._get_obs(lick_sig=lick_sig, win_rem=win_rem)
        return obs, reward, done, info


# ============================================================
# Plot saving
# ============================================================
def _save_fig(fig, out_path_no_ext: str, ext: str, pdf_pages: Optional[PdfPages] = None):
    ensure_dir(os.path.dirname(out_path_no_ext))
    if ext.lower() == "pdf" and pdf_pages is not None:
        pdf_pages.savefig(fig, bbox_inches="tight")
    else:
        fig.savefig(f"{out_path_no_ext}.{ext}", bbox_inches="tight")

def ema_1d(y: np.ndarray, alpha: float = 0.02) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    out = np.zeros_like(y)
    m = 0.0
    for i in range(len(y)):
        if np.isnan(y[i]):
            out[i] = m
            continue
        m = (1 - alpha) * m + alpha * y[i]
        out[i] = m
    return out

def logs_to_array_with_ffill(logs: List[Dict[str, float]], metric: str, target_len: int) -> np.ndarray:
    y = np.full((target_len,), np.nan, dtype=np.float32)
    if len(logs) == 0:
        return y
    vals = [float(d.get(metric, np.nan)) for d in logs]
    L = min(len(vals), target_len)
    y[:L] = np.array(vals[:L], dtype=np.float32)
    if L < target_len and L > 0 and not np.isnan(y[L-1]):
        y[L:] = y[L-1]
    return y

def mean_sem(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(arr, axis=0)
    sem = np.zeros_like(mean, dtype=np.float32)
    for t in range(arr.shape[1]):
        col = arr[:, t]
        col = col[~np.isnan(col)]
        if len(col) <= 1:
            sem[t] = 0.0
        else:
            sem[t] = float(np.std(col, ddof=1) / math.sqrt(len(col)))
    return mean.astype(np.float32), sem.astype(np.float32)

def plot_metric_mean_sem(ep: np.ndarray, mean: np.ndarray, sem: np.ndarray, label: str, alpha_fill: float = 0.20):
    plt.plot(ep, mean, label=label)
    lo = mean - sem
    hi = mean + sem
    plt.fill_between(ep, lo, hi, alpha=alpha_fill)

def plot_comparison_curves_mean_sem(
    results_logs_by_seed: Dict[str, List[List[Dict[str, float]]]],
    metric: str,
    episodes: int,
    save_dir: str,
    ext: str = "pdf",
    pdf_pages: Optional[PdfPages] = None,
    alpha_ema: float = 0.02,
    title: Optional[str] = None,
):
    fig = plt.figure(figsize=(8, 4.8))
    ep = np.arange(1, episodes + 1, dtype=np.int32)

    for label, runs in results_logs_by_seed.items():
        arr = np.stack([logs_to_array_with_ffill(run, metric, episodes) for run in runs], axis=0)
        arr_s = np.stack([ema_1d(arr[i], alpha=alpha_ema) for i in range(arr.shape[0])], axis=0)
        m, s = mean_sem(arr_s)
        plot_metric_mean_sem(ep, m, s, label=label)

    plt.xlabel("episode")
    plt.ylabel(metric)
    plt.title(title or f"{metric}: mean +/- SEM (across seeds)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    _save_fig(fig, os.path.join(save_dir, f"compare_{metric}_mean_sem"), ext=ext, pdf_pages=pdf_pages)

def plot_single_learning_panel_mean_sem(
    runs: List[List[Dict[str, float]]],
    episodes: int,
    save_dir: str,
    ext: str,
    pdf_pages: Optional[PdfPages],
    title_suffix: str = "",
    alpha_ema: float = 0.02,
):
    ep = np.arange(1, episodes + 1, dtype=np.int32)

    def get_ms(metric):
        arr = np.stack([logs_to_array_with_ffill(run, metric, episodes) for run in runs], axis=0)
        arr_s = np.stack([ema_1d(arr[i], alpha=alpha_ema) for i in range(arr.shape[0])], axis=0)
        return mean_sem(arr_s)

    m_succ, s_succ = get_ms("bout_success_rate")
    m_obs, s_obs = get_ms("obs_rate")
    m_seen, s_seen = get_ms("obs_seen_lick_rate")
    m_eatw, s_eatw = get_ms("eat_in_window_rate")
    m_lat, s_lat = get_ms("mean_latency_s")
    m_fam, s_fam = get_ms("familiarity")
    m_lr, s_lr = get_ms("lr")

    fig = plt.figure(figsize=(14, 4))

    plt.subplot(1, 3, 1)
    plot_metric_mean_sem(ep, m_succ, s_succ, label="success")
    plt.title(f"EMA bout success{title_suffix}")
    plt.xlabel("episode"); plt.ylabel("rate")
    plt.legend(fontsize=8)

    plt.subplot(1, 3, 2)
    plot_metric_mean_sem(ep, m_obs, s_obs, label="observe")
    plot_metric_mean_sem(ep, m_seen, s_seen, label="obs sees cue")
    plt.title("EMA observation metrics")
    plt.xlabel("episode"); plt.legend(fontsize=8)

    plt.subplot(1, 3, 3)
    plot_metric_mean_sem(ep, m_eatw, s_eatw, label="eat in window")
    plot_metric_mean_sem(ep, m_lat, s_lat, label="latency (s)")
    if np.nanmax(m_fam) > 1e-6:
        plot_metric_mean_sem(ep, m_fam, s_fam, label="familiarity")
    if np.nanmax(m_lr) > 0:
        plot_metric_mean_sem(ep, m_lr, s_lr, label="lr")
    plt.title("EMA eating + latency + familiarity + lr")
    plt.xlabel("episode"); plt.legend(fontsize=8)

    plt.tight_layout()
    _save_fig(fig, os.path.join(save_dir, "learning_curves_mean_sem"), ext=ext, pdf_pages=pdf_pages)

# ============================================================
# Significance tests (no SciPy)
# ============================================================
def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va = np.var(a, ddof=1)
    vb = np.var(b, ddof=1)
    sp = math.sqrt(((len(a)-1)*va + (len(b)-1)*vb) / max(1, (len(a)+len(b)-2)))
    if sp <= 1e-12:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / sp)

def permutation_test_mean_diff(a: np.ndarray, b: np.ndarray, n_perm: int = 5000, seed: int = 0) -> float:
    rng = np.random.RandomState(seed)
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    obs = abs(np.mean(a) - np.mean(b))
    x = np.concatenate([a, b], axis=0)
    na = len(a)
    count = 0
    for _ in range(int(n_perm)):
        rng.shuffle(x)
        d = abs(np.mean(x[:na]) - np.mean(x[na:]))
        if d >= obs - 1e-12:
            count += 1
    return float((count + 1) / (n_perm + 1))

def last_k_window_scores(runs: List[List[Dict[str, float]]], metric: str, episodes: int, last_k: int) -> np.ndarray:
    last_k = int(min(last_k, episodes))
    scores = []
    for logs in runs:
        y = logs_to_array_with_ffill(logs, metric, episodes)
        tail = y[-last_k:]
        valid = tail[~np.isnan(tail)]
        scores.append(float(np.mean(valid)) if len(valid) > 0 else float("nan"))
    return np.array(scores, dtype=np.float32)

def write_stats_report(
    out_path: str,
    group_key: str,
    metric: str,
    a_label: str,
    b_label: str,
    a_scores: np.ndarray,
    b_scores: np.ndarray,
    pval: float,
    d: float,
):
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(f"\n[{group_key}] metric={metric}\n")
        f.write(f"  {a_label}: n={len(a_scores)} mean={float(np.mean(a_scores)):.4f} std={float(np.std(a_scores, ddof=1) if len(a_scores)>1 else 0.0):.4f}\n")
        f.write(f"  {b_label}: n={len(b_scores)} mean={float(np.mean(b_scores)):.4f} std={float(np.std(b_scores, ddof=1) if len(b_scores)>1 else 0.0):.4f}\n")
        f.write(f"  perm-test p={pval:.6f}  Cohen_d={d:.4f}\n")

# ============================================================
# Latent / "DA-like" simulation (kept; optional)
# ============================================================
def calcium_filter(impulses, dt_s, tau_rise_s=0.2, tau_decay_s=1.2):
    impulses = np.asarray(impulses, dtype=np.float32)
    n = len(impulses)
    L = int(np.ceil(8 * tau_decay_s / dt_s))
    t = np.arange(L, dtype=np.float32) * dt_s
    k = (1 - np.exp(-t / max(1e-6, tau_rise_s))) * np.exp(-t / max(1e-6, tau_decay_s))
    k = k / (k.sum() + 1e-9)
    y = np.convolve(impulses, k, mode="full")[:n]
    return y.astype(np.float32)

def gae_advantages(rewards, values, dones, gamma=0.99, lam=0.95):
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        nonterminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * values[t+1] * nonterminal - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        adv[t] = gae
    return adv

def simulate_latent_models_for_episode(tr, agent, gamma=0.99, ep_frac=0.0, gae_lam=0.95):
    r = np.array(tr["reward"], dtype=np.float32)
    obs = np.array(tr["observe"], dtype=int)
    eat = np.array(tr["eat"], dtype=int)
    seen_lick = np.array(tr["seen_lick"], dtype=int)
    phase = np.array(tr["phase"], dtype=np.float32)
    S = np.array(tr["state"], dtype=np.float32)
    S2 = np.array(tr["next_state"], dtype=np.float32)
    done = np.array(tr["done"], dtype=np.float32)

    V_s = agent.value_batch(S)
    V_s2 = agent.value_batch(S2)
    rpe = r + gamma * V_s2 * (1.0 - done) - V_s

    V_boot = np.concatenate([V_s, V_s2[-1:]], axis=0)
    adv = gae_advantages(r, V_boot[:len(r)+1], done, gamma=gamma, lam=gae_lam)

    S2_prior = S2.copy()
    S2_prior[:, 2] = 0.0
    S2_prior[:, 3] = 0.0
    V_post = V_s2
    V_prior = agent.value_batch(S2_prior)
    deltaV = (V_post - V_prior) * obs.astype(np.float32)

    sal = np.zeros(len(r), dtype=np.float32)
    sal += 1.0 * (obs == 1).astype(np.float32)
    sal += 0.6 * (eat == 1).astype(np.float32)

    sigma = 0.25 * (1.0 - ep_frac) + 0.10 * ep_frac
    p = np.exp(-(phase / max(1e-6, sigma)) ** 2)
    surprise = -np.log(p + 1e-6)
    info_gain = surprise * ((obs == 1) & (seen_lick == 1)).astype(np.float32)

    impulses = {
        "reward_RPE": 0.8 * rpe,
        "gae_adv": 0.7 * adv,
        "social_cue_RPE": 1.2 * deltaV,
        "social_value": 0.6 * deltaV,
        "action_salience": 0.3 * sal,
        "info_gain": 0.25 * info_gain,
    }

    if hasattr(agent, "policy_entropy_batch"):
        ent = agent.policy_entropy_batch(S)
        if ent is not None:
            impulses["policy_entropy"] = 0.15 * ent.astype(np.float32)

    return impulses

def simulate_and_plot_latents(traces, agent, dt_s, gamma=0.99,
                              save_dir="outputs", ext="pdf", pdf_pages=None):
    impulses_by_model = defaultdict(list)
    n = len(traces)
    for i, tr in enumerate(traces):
        ep_frac = i / max(1, n - 1)
        imp = simulate_latent_models_for_episode(tr, agent, gamma=gamma, ep_frac=ep_frac)
        for k, v in imp.items():
            impulses_by_model[k].append(v)

    phot_by_model = {}
    for k, imps in impulses_by_model.items():
        phot_by_model[k] = [calcium_filter(x, dt_s=dt_s, tau_rise_s=0.2, tau_decay_s=1.2) for x in imps]
    return phot_by_model


# ============================================================
# Agents
# ============================================================
class BaseAgent:
    def act(self, s: np.ndarray):
        raise NotImplementedError
    def record_transition(self, s, a, r, s2, done, extra):
        pass
    def step_update(self):
        return None
    def episode_update(self):
        return None
    def value_batch(self, states: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    def get_lr(self) -> float:
        return 0.0

# -------------------------
# DQN (dueling + double + huber + soft target)
# -------------------------
class DuelingQNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.V = nn.Linear(hidden, 1)
        self.A = nn.Linear(hidden, action_dim)

    def forward(self, x):
        h = self.backbone(x)
        v = self.V(h)
        a = self.A(h)
        return v + (a - a.mean(dim=1, keepdim=True))

class TorchReplayBuffer:
    def __init__(self, capacity, state_dim, device):
        self.capacity = int(capacity)
        self.device = device
        self.state = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
        self.action = torch.zeros(capacity, dtype=torch.int64, device=device)
        self.reward = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.next_state = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
        self.done = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.idx = 0
        self.size = 0

    def push(self, s, a, r, s2, done):
        self.state[self.idx] = torch.tensor(s, dtype=torch.float32, device=self.device)
        self.action[self.idx] = int(a)
        self.reward[self.idx] = float(r)
        self.next_state[self.idx] = torch.tensor(s2, dtype=torch.float32, device=self.device)
        self.done[self.idx] = float(done)
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.state[indices],
            self.action[indices].unsqueeze(1),
            self.reward[indices],
            self.next_state[indices],
            self.done[indices],
        )

    def __len__(self):
        return self.size

class DQNAgent(BaseAgent):
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=3e-4,
        gamma=0.99,
        batch_size=256,
        buffer_size=50000,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=0.9995,
        tau=0.02,
        grad_clip=10.0,
        device=None,
        use_compile=False,
    ):
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.tau = float(tau)
        self.grad_clip = float(grad_clip)

        self.eps = float(eps_start)
        self.eps_end = float(eps_end)
        self.eps_decay = float(eps_decay)

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q = DuelingQNet(self.state_dim, self.action_dim).to(self.device)
        self.qt = DuelingQNet(self.state_dim, self.action_dim).to(self.device)
        self.qt.load_state_dict(self.q.state_dict())

        if use_compile and hasattr(torch, "compile"):
            try:
                self.q = torch.compile(self.q)
            except Exception:
                pass

        self.opt = optim.Adam(self.q.parameters(), lr=float(lr))
        self.loss_fn = nn.SmoothL1Loss()
        self.rb = TorchReplayBuffer(buffer_size, state_dim, self.device)
        self.scaler = make_grad_scaler(self.device.type)

    def get_lr(self) -> float:
        return float(self.opt.param_groups[0]["lr"])

    def act(self, s):
        if np.random.rand() < self.eps:
            return int(np.random.randint(self.action_dim)), {}
        st = torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            qv = self.q(st)
        return int(torch.argmax(qv, dim=1).item()), {}

    def record_transition(self, s, a, r, s2, done, extra):
        self.rb.push(s, a, r, s2, done)

    def step_update(self, updates_per_step=1):
        if len(self.rb) < self.batch_size:
            self.eps = max(self.eps_end, self.eps * self.eps_decay)
            return None

        last_loss = None
        for _ in range(int(updates_per_step)):
            s, a, r, s2, d = self.rb.sample(self.batch_size)
            with make_autocast(self.device.type):
                q_sa = self.q(s).gather(1, a).squeeze(1)
                with torch.no_grad():
                    a_star = torch.argmax(self.q(s2), dim=1, keepdim=True)
                    q_next = self.qt(s2).gather(1, a_star).squeeze(1)
                    target = r + self.gamma * q_next * (1.0 - d)
                loss = self.loss_fn(q_sa, target)

            self.opt.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
            nn.utils.clip_grad_norm_(self.q.parameters(), self.grad_clip)
            self.scaler.step(self.opt)
            self.scaler.update()

            with torch.no_grad():
                for p, pt in zip(self.q.parameters(), self.qt.parameters()):
                    pt.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

            last_loss = float(loss.item())

        self.eps = max(self.eps_end, self.eps * self.eps_decay)
        return last_loss

    def value_batch(self, states: np.ndarray) -> np.ndarray:
        st = torch.tensor(states, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            qv = self.q(st)
            v = torch.max(qv, dim=1).values
        return v.detach().cpu().numpy().astype(np.float32)

# -------------------------
# PPO (actor-critic)
# -------------------------
class ActorCriticNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.pi = nn.Linear(hidden, action_dim)
        self.v = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.body(x)
        logits = self.pi(h)
        value = self.v(h).squeeze(-1)
        return logits, value

class PPOAgent(BaseAgent):
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        train_epochs=4,
        minibatch_size=256,
        max_grad_norm=1.0,
        device=None,
    ):
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_ratio = float(clip_ratio)
        self.vf_coef = float(vf_coef)
        self.ent_coef = float(ent_coef)
        self.train_epochs = int(train_epochs)
        self.minibatch_size = int(minibatch_size)
        self.max_grad_norm = float(max_grad_norm)

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = ActorCriticNet(self.state_dim, self.action_dim).to(self.device)
        self.opt = optim.Adam(self.net.parameters(), lr=float(lr))
        self.roll = []

    def get_lr(self) -> float:
        return float(self.opt.param_groups[0]["lr"])

    def act(self, s):
        st = torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, v = self.net(st)
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            logp = dist.log_prob(a)
            ent = dist.entropy()
        return int(a.item()), {"logp": float(logp.item()), "v": float(v.item()), "ent": float(ent.item())}

    def record_transition(self, s, a, r, s2, done, extra):
        self.roll.append({
            "s": np.array(s, dtype=np.float32),
            "a": int(a),
            "r": float(r),
            "done": float(done),
            "logp": float(extra.get("logp", 0.0)),
            "v": float(extra.get("v", 0.0)),
        })

    def episode_update(self):
        if len(self.roll) < 8:
            self.roll.clear()
            return None

        S = np.stack([x["s"] for x in self.roll], axis=0).astype(np.float32)
        A = np.array([x["a"] for x in self.roll], dtype=np.int64)
        R = np.array([x["r"] for x in self.roll], dtype=np.float32)
        D = np.array([x["done"] for x in self.roll], dtype=np.float32)
        LOGP_OLD = np.array([x["logp"] for x in self.roll], dtype=np.float32)
        V = np.array([x["v"] for x in self.roll], dtype=np.float32)

        with torch.no_grad():
            st_last = torch.tensor(S[-1], dtype=torch.float32, device=self.device).unsqueeze(0)
            _, v_last = self.net(st_last)
            v_last = float(v_last.item())
        V_ext = np.concatenate([V, np.array([v_last], dtype=np.float32)], axis=0)

        ADV = gae_advantages(R, V_ext, D, gamma=self.gamma, lam=self.gae_lambda)
        RET = ADV + V
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)

        ts = torch.tensor(S, dtype=torch.float32, device=self.device)
        ta = torch.tensor(A, dtype=torch.int64, device=self.device)
        told = torch.tensor(LOGP_OLD, dtype=torch.float32, device=self.device)
        tadv = torch.tensor(ADV, dtype=torch.float32, device=self.device)
        tret = torch.tensor(RET, dtype=torch.float32, device=self.device)

        n = len(S)
        idx = np.arange(n)
        info = {}

        for _ in range(self.train_epochs):
            np.random.shuffle(idx)
            for start in range(0, n, self.minibatch_size):
                mb = idx[start:start+self.minibatch_size]
                logits, vpred = self.net(ts[mb])
                dist = torch.distributions.Categorical(logits=logits)

                logp = dist.log_prob(ta[mb])
                ratio = torch.exp(logp - told[mb])

                clip = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
                pg_loss = -(torch.min(ratio * tadv[mb], clip * tadv[mb])).mean()

                v_loss = ((vpred - tret[mb]) ** 2).mean()
                ent = dist.entropy().mean()
                loss = pg_loss + self.vf_coef * v_loss - self.ent_coef * ent

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.opt.step()

                info = {
                    "pg_loss": float(pg_loss.item()),
                    "v_loss": float(v_loss.item()),
                    "ent": float(ent.item()),
                    "loss": float(loss.item()),
                }

        self.roll.clear()
        return info

    def value_batch(self, states: np.ndarray) -> np.ndarray:
        st = torch.tensor(states, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            _, v = self.net(st)
        return v.detach().cpu().numpy().astype(np.float32)

    def policy_entropy_batch(self, states: np.ndarray):
        st = torch.tensor(states, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits, _ = self.net(st)
            dist = torch.distributions.Categorical(logits=logits)
            ent = dist.entropy()
        return ent.detach().cpu().numpy().astype(np.float32)

# -------------------------
# Tabular Q-learning
# -------------------------
class TabularQAgent(BaseAgent):
    def __init__(self, action_dim=5, alpha=0.25, gamma=0.99, eps_start=1.0, eps_end=0.05, eps_decay=0.9995,
                 phase_bins=10, win_bins=7, cd_bins=6):
        self.action_dim = int(action_dim)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.eps = float(eps_start)
        self.eps_end = float(eps_end)
        self.eps_decay = float(eps_decay)

        self.phase_bins = int(phase_bins)
        self.win_bins = int(win_bins)
        self.cd_bins = int(cd_bins)
        self.Q = defaultdict(lambda: np.zeros(self.action_dim, dtype=np.float32))

    def _disc(self, s: np.ndarray):
        lp, lf, lick, win, phase, cd = s.tolist()
        p = int(np.round(lp * 14))
        f = int(np.round(lf * 14))
        lickb = int(lick > 0.5)
        winb = int(np.round(win * (self.win_bins - 1)))
        winb = clamp_int(winb, 0, self.win_bins - 1)
        phb = int(np.floor(phase * self.phase_bins))
        phb = clamp_int(phb, 0, self.phase_bins - 1)
        cdb = int(np.round(cd * (self.cd_bins - 1)))
        cdb = clamp_int(cdb, 0, self.cd_bins - 1)
        return (p, f, lickb, winb, phb, cdb)

    def act(self, s):
        key = self._disc(s)
        if np.random.rand() < self.eps:
            a = int(np.random.randint(self.action_dim))
        else:
            a = int(np.argmax(self.Q[key]))
        return a, {"key": key}

    def record_transition(self, s, a, r, s2, done, extra):
        k = extra["key"]
        k2 = self._disc(s2)
        q = self.Q[k]
        target = float(r) + self.gamma * (0.0 if done else float(np.max(self.Q[k2])))
        q[a] = (1 - self.alpha) * q[a] + self.alpha * target
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

    def value_batch(self, states: np.ndarray) -> np.ndarray:
        out = np.zeros(len(states), dtype=np.float32)
        for i, s in enumerate(states):
            out[i] = float(np.max(self.Q[self._disc(s)]))
        return out

# ============================================================
# Training + logging
# ============================================================
def train_social_with_latents(
    env,
    agent: BaseAgent,
    episodes=1200,
    log_every=100,
    updates_per_step=4,
    early_stop_threshold=0.8,
    early_stop_window=100,
    compute_latents: bool = True,
):
    logs = []
    traces = []
    success_history = deque(maxlen=early_stop_window)

    for ep in range(1, episodes + 1):
        s = env.reset()
        done = False

        ep_reward = 0.0
        succ_bouts = 0
        n_bouts = max(1, env.n_bouts)
        obs_steps = 0
        obs_seen = 0
        eat_in_win = 0
        latencies_steps = []

        tr = defaultdict(list)

        while not done:
            a, extra = agent.act(s)
            s2, r, done, info = env.step(a)
            ep_reward += r

            if info["eat_valid"] == 1:
                succ_bouts += 1
                if info.get("latency_steps", -1) >= 0:
                    latencies_steps.append(int(info["latency_steps"]))
            if info["observe"] == 1:
                obs_steps += 1
                if info["seen_lick"] == 1:
                    obs_seen += 1
            if info["eat"] == 1 and info["window_open"] == 1:
                eat_in_win += 1

            if compute_latents:
                tr["state"].append(s.copy())
                tr["next_state"].append(s2.copy())
                tr["reward"].append(float(r))
                tr["done"].append(float(done))

                tr["teacher_lick"].append(int(info["teacher_lick"]))
                tr["window_open"].append(int(info["window_open"]))
                tr["observe"].append(int(info["observe"]))
                tr["eat"].append(int(info["eat"]))
                tr["eat_valid"].append(int(info["eat_valid"]))
                tr["seen_lick"].append(int(info["seen_lick"]))
                tr["win_rem_est"].append(float(info["win_rem_est"]))
                tr["phase"].append(float(s[4]))

            agent.record_transition(s, a, r, s2, done, extra)

            if isinstance(agent, DQNAgent):
                agent.step_update(updates_per_step=updates_per_step)

            s = s2

        if hasattr(agent, "episode_update"):
            agent.episode_update()

        if compute_latents:
            for k in list(tr.keys()):
                tr[k] = np.array(tr[k])
            traces.append(tr)

        mean_latency_s = float(np.mean(latencies_steps) * env.dt_s) if len(latencies_steps) > 0 else np.nan

        logs.append({
            "ep": ep,
            "bout_success_rate": float(succ_bouts / n_bouts),
            "mean_reward": float(ep_reward),
            "obs_rate": float(obs_steps / max(1, env.max_steps)),
            "obs_seen_lick_rate": float(obs_seen / max(1, env.max_steps)),
            "eat_in_window_rate": float(eat_in_win / max(1, env.max_steps)),
            "mean_latency_s": float(mean_latency_s),

            "eps": float(getattr(agent, "eps", np.nan)),
            "familiarity": float(getattr(env, "familiarity", 0.0)),
            "p_detect_eff": float(getattr(env, "_effective_detect_prob")()) if getattr(env, "social_visibility", 1.0) > 0.0 else 0.0,
            "lr": float(agent.get_lr() if hasattr(agent, "get_lr") else 0.0),
        })

        success_history.append(logs[-1]["bout_success_rate"])

        if ep % log_every == 0:
            recent = logs[-log_every:]
            print(
                f"[ep {ep:4d}] "
                f"boutSucc={np.mean([x['bout_success_rate'] for x in recent]):.3f} "
                f"lat={np.nanmean([x['mean_latency_s'] for x in recent]):.3f}s "
                f"obs={np.mean([x['obs_rate'] for x in recent]):.3f} "
                f"obsSeen={np.mean([x['obs_seen_lick_rate'] for x in recent]):.3f} "
                f"eatWin={np.mean([x['eat_in_window_rate'] for x in recent]):.3f} "
                f"R={np.mean([x['mean_reward'] for x in recent]):.3f} "
                f"eps={logs[-1]['eps']:.3f} "
                f"fam={logs[-1]['familiarity']:.2f} "
                f"lr={logs[-1]['lr']:.2e}"
            )

        if len(success_history) == early_stop_window and np.mean(success_history) > early_stop_threshold:
            print(f"Early stopping at episode {ep}: Avg success rate {np.mean(success_history):.3f} > {early_stop_threshold}")
            break

    return logs, traces


# ============================================================
# Condition builder (teacher palatability UNCHANGED)
# ============================================================
def build_env_for_condition(args, teacher_pal: str, learner_profile: str, familiarity_mode: str, empathy_mode: str):
    lick_mean = args.lick_mean_s
    if teacher_pal == "high":
        lick_mean = args.pal_high_lick_mean_s
    elif teacher_pal == "low":
        lick_mean = args.pal_low_lick_mean_s
    elif teacher_pal == "default":
        lick_mean = args.lick_mean_s

    env_kwargs = dict(
        grid_size=args.grid_size,
        dt_s=args.dt_s,
        max_time_s=args.max_time_s,
        teacher_period_s=args.teacher_period_s,
        teacher_jitter_s=args.teacher_jitter_s,
        lick_mean_s=lick_mean,
        lick_dist=args.lick_dist,
        lick_cv=args.lick_cv,
        window_s=args.window_s,
        eat_cooldown_s=args.eat_cooldown_s,
        observe_cost=args.observe_cost,
        eat_cost=args.eat_cost,
        attend_bonus=args.attend_bonus,
        p_detect_per_obs=args.p_detect_per_obs,
        win_rem_noise_steps=args.win_rem_noise_steps,

        social_visibility=1.0,
        detect_memory_steps=None,

        familiarity_enabled=(familiarity_mode == "on"),
        familiarity_init=args.fam_init,
        fam_decay_ep=args.fam_decay_ep,
        fam_gain_obs=args.fam_gain_obs,
        fam_gain_detect=args.fam_gain_detect,
        p_detect_fam_boost=args.p_detect_fam_boost,
        noise_fam_reduction=args.noise_fam_reduction,

        # familiarity behavior shaping (requested)
        fam_obs_cost_reduction=args.fam_obs_cost_reduction,
        fam_attend_bonus_boost=args.fam_attend_bonus_boost,
        fam_move_shaping=args.fam_move_shaping,
        fam_obs_during_window_penalty=args.fam_obs_during_window_penalty,
        fam_latency_bonus=args.fam_latency_bonus,

        # empathy
        empathy_enabled=(empathy_mode == "on"),
        empathy_seen_reward=args.empathy_seen_reward,
    )

    if learner_profile == "normal":
        pass
    elif learner_profile == "blind":
        env_kwargs["social_visibility"] = 0.0
    elif learner_profile == "autistic":
        env_kwargs["p_detect_per_obs"] = args.aut_p_detect_per_obs
        env_kwargs["win_rem_noise_steps"] = args.aut_win_rem_noise_steps
        env_kwargs["detect_memory_steps"] = args.aut_detect_memory_steps
        env_kwargs["observe_cost"] = args.aut_observe_cost
        if familiarity_mode == "on":
            env_kwargs["fam_gain_obs"] = args.aut_fam_gain_obs
            env_kwargs["fam_gain_detect"] = args.aut_fam_gain_detect
            env_kwargs["p_detect_fam_boost"] = args.aut_p_detect_fam_boost
            env_kwargs["noise_fam_reduction"] = args.aut_noise_fam_reduction
    else:
        raise ValueError(f"Unknown learner_profile={learner_profile}")

    return SocialLickEnv1D(**env_kwargs)

def build_agent(args, state_dim=6, action_dim=5, lr_override: Optional[float] = None):
    lr_use = float(args.lr if lr_override is None else lr_override)
    if args.algo == "dqn":
        return DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=lr_use,
            gamma=args.gamma,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            eps_start=args.eps_start,
            eps_end=args.eps_end,
            eps_decay=args.eps_decay,
            tau=args.tau,
            grad_clip=args.grad_clip,
            use_compile=args.use_compile,
        )
    elif args.algo == "ppo":
        return PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=lr_use,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_ratio=args.clip_ratio,
            vf_coef=args.vf_coef,
            ent_coef=args.ent_coef,
            train_epochs=args.ppo_epochs,
            minibatch_size=args.minibatch_size,
            max_grad_norm=args.max_grad_norm,
        )
    elif args.algo == "tabular":
        return TabularQAgent(
            action_dim=action_dim,
            alpha=args.tab_alpha,
            gamma=args.gamma,
            eps_start=args.eps_start,
            eps_end=args.eps_end,
            eps_decay=args.eps_decay,
        )
    else:
        raise ValueError(f"Unknown algo: {args.algo}")

# ============================================================
# Multi-seed runner
# ============================================================
def run_single_condition_multi_seed(args, teacher_pal: str, learner_profile: str, familiarity_mode: str, empathy_mode: str, out_dir: str):
    ensure_dir(out_dir)

    pdf_pages = None
    if args.plot_ext.lower() == "pdf" and args.multipage_pdf:
        pdf_pages = PdfPages(os.path.join(out_dir, f"all_figures_{args.algo}_mean_sem.pdf"))

    runs_logs: List[List[Dict[str, float]]] = []
    for i in range(args.n_seeds):
        seed_i = args.seed + i * args.seed_stride
        print(f"  - seed {seed_i}")
        set_seed(seed_i)
        env = build_env_for_condition(args, teacher_pal=teacher_pal, learner_profile=learner_profile,
                                      familiarity_mode=familiarity_mode, empathy_mode=empathy_mode)
        agent = build_agent(args, state_dim=6, action_dim=5)

        logs, _traces = train_social_with_latents(
            env, agent,
            episodes=args.episodes,
            log_every=max(50, args.episodes // 12),
            updates_per_step=args.updates_per_step,
            early_stop_threshold=args.early_stop_threshold,
            early_stop_window=args.early_stop_window,
            compute_latents=False,  # off for multi-seed
        )
        runs_logs.append(logs)

    if not args.no_plots:
        title_suffix = f" (T={teacher_pal}, L={learner_profile}, fam={familiarity_mode}, emp={empathy_mode}, n={args.n_seeds})"
        plot_single_learning_panel_mean_sem(
            runs=runs_logs,
            episodes=args.episodes,
            save_dir=out_dir,
            ext=args.plot_ext,
            pdf_pages=pdf_pages,
            title_suffix=title_suffix,
            alpha_ema=args.alpha_ema,
        )

    if pdf_pages is not None:
        pdf_pages.close()

    return runs_logs

# ============================================================
# Comparison suite (supports empathy + familiarity without touching teacher pal)
# ============================================================
def run_comparison_suite_multi_seed(args):
    base_dir = args.save_dir
    ensure_dir(base_dir)

    cmp_pdf = None
    if args.plot_ext.lower() == "pdf" and args.multipage_pdf:
        cmp_pdf = PdfPages(os.path.join(base_dir, f"comparison_suite_{args.algo}_mean_sem.pdf"))

    teacher_pals = ["high", "low"] if args.compare_teacher_pal else [args.teacher_pal]
    profiles = ["normal", "blind", "autistic"] if args.compare_profiles else [args.learner_profile]
    fam_modes = ["off", "on"] if args.compare_familiarity else [args.familiarity]
    emp_modes = ["off", "on"] if args.compare_empathy else [args.empathy]

    results: Dict[str, List[List[Dict[str, float]]]] = {}
    meta: Dict[str, Tuple[str, str, str, str]] = {}

    cond_list = []
    for tp in teacher_pals:
        for pr in profiles:
            for fm in fam_modes:
                for em in emp_modes:
                    cond_list.append((tp, pr, fm, em))

    for (tp, pr, fm, em) in cond_list:
        label = f"T={tp}|L={pr}|F={fm}|E={em}"
        out_dir = os.path.join(base_dir, label.replace("|", "__").replace("=", "_"))
        print("\n" + "=" * 80)
        print(f"Running condition: {label}  (n_seeds={args.n_seeds})")
        runs_logs = run_single_condition_multi_seed(args, tp, pr, fm, em, out_dir=out_dir)
        results[label] = runs_logs
        meta[label] = (tp, pr, fm, em)

    if not args.no_plots:
        plot_comparison_curves_mean_sem(results, metric="bout_success_rate", episodes=args.episodes,
                                        save_dir=base_dir, ext=args.plot_ext, pdf_pages=cmp_pdf,
                                        alpha_ema=args.alpha_ema,
                                        title="Bout success rate (mean +/- SEM)")
        plot_comparison_curves_mean_sem(results, metric="mean_reward", episodes=args.episodes,
                                        save_dir=base_dir, ext=args.plot_ext, pdf_pages=cmp_pdf,
                                        alpha_ema=args.alpha_ema,
                                        title="Episode reward (mean +/- SEM)")
        plot_comparison_curves_mean_sem(results, metric="obs_rate", episodes=args.episodes,
                                        save_dir=base_dir, ext=args.plot_ext, pdf_pages=cmp_pdf,
                                        alpha_ema=args.alpha_ema,
                                        title="Observe rate (mean +/- SEM)")
        plot_comparison_curves_mean_sem(results, metric="obs_seen_lick_rate", episodes=args.episodes,
                                        save_dir=base_dir, ext=args.plot_ext, pdf_pages=cmp_pdf,
                                        alpha_ema=args.alpha_ema,
                                        title="OBS sees cue rate (mean +/- SEM)")
        plot_comparison_curves_mean_sem(results, metric="mean_latency_s", episodes=args.episodes,
                                        save_dir=base_dir, ext=args.plot_ext, pdf_pages=cmp_pdf,
                                        alpha_ema=args.alpha_ema,
                                        title="Mean response latency (s) (mean +/- SEM)")

    if cmp_pdf is not None:
        cmp_pdf.close()

    stats_path = os.path.join(base_dir, "significance_report.txt")
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("Permutation-test significance on LAST-K episode-window means.\n")
        f.write(f"metric_last_k={args.stats_last_k}, n_perm={args.stats_n_perm}, seed={args.seed}\n\n")

    # helper for pairwise tests
    def _pairwise(a_lab, b_lab, group_key):
        a_runs = results[a_lab]; b_runs = results[b_lab]
        for metric in ["bout_success_rate", "mean_reward", "obs_seen_lick_rate", "mean_latency_s"]:
            a_scores = last_k_window_scores(a_runs, metric, args.episodes, args.stats_last_k)
            b_scores = last_k_window_scores(b_runs, metric, args.episodes, args.stats_last_k)
            pval = permutation_test_mean_diff(a_scores, b_scores, n_perm=args.stats_n_perm, seed=args.seed)
            d = cohen_d(a_scores, b_scores)
            write_stats_report(stats_path, group_key, metric, a_lab, b_lab, a_scores, b_scores, pval, d)

    # Compare empathy on/off within each (T,L,F)
    if args.compare_empathy:
        groups = defaultdict(dict)  # (tp,pr,fm) -> {em: label}
        for label, (tp, pr, fm, em) in meta.items():
            groups[(tp, pr, fm)][em] = label
        for (tp, pr, fm), m in groups.items():
            if "off" in m and "on" in m:
                _pairwise(m["off"], m["on"], group_key=f"T={tp},L={pr},F={fm}  (E off vs on)")

    # Compare familiarity on/off within each (T,L,E)
    if args.compare_familiarity:
        groups = defaultdict(dict)  # (tp,pr,em) -> {fm: label}
        for label, (tp, pr, fm, em) in meta.items():
            groups[(tp, pr, em)][fm] = label
        for (tp, pr, em), m in groups.items():
            if "off" in m and "on" in m:
                _pairwise(m["off"], m["on"], group_key=f"T={tp},L={pr},E={em}  (F off vs on)")

    # Compare profiles (normal vs blind/autistic) within each (T,F,E)
    if args.compare_profiles:
        groups = defaultdict(dict)  # (tp,fm,em) -> {profile: label}
        for label, (tp, pr, fm, em) in meta.items():
            groups[(tp, fm, em)][pr] = label
        for (tp, fm, em), m in groups.items():
            if "normal" in m and "blind" in m:
                _pairwise(m["normal"], m["blind"], group_key=f"T={tp},F={fm},E={em}  (normal vs blind)")
            if "normal" in m and "autistic" in m:
                _pairwise(m["normal"], m["autistic"], group_key=f"T={tp},F={fm},E={em}  (normal vs autistic)")

    print("\nSaved significance report:", stats_path)
    return results

# ============================================================
# LR sweep (optional)
# ============================================================
def parse_lr_sweep(s: str) -> List[float]:
    if s is None or s.strip() == "":
        return []
    out = []
    for part in s.split(","):
        part = part.strip()
        if part == "":
            continue
        out.append(float(part))
    return out

def run_lr_sweep(args):
    lrs = parse_lr_sweep(args.lr_sweep)
    if len(lrs) == 0:
        return

    base_dir = args.save_dir
    ensure_dir(base_dir)

    tp = args.teacher_pal
    pr = args.learner_profile
    fm = args.familiarity
    em = args.empathy

    means = []
    sems = []
    raw_means = {}

    for lr in lrs:
        scores = []
        print("\n" + "=" * 80)
        print(f"LR sweep: lr={lr:g} (n_seeds={args.n_seeds})  condition: T={tp}, L={pr}, F={fm}, E={em}")
        for i in range(args.n_seeds):
            seed_i = args.seed + i * args.seed_stride
            set_seed(seed_i)
            env = build_env_for_condition(args, teacher_pal=tp, learner_profile=pr, familiarity_mode=fm, empathy_mode=em)
            agent = build_agent(args, state_dim=6, action_dim=5, lr_override=lr)
            logs, _tr = train_social_with_latents(
                env, agent,
                episodes=args.episodes,
                log_every=max(50, args.episodes // 12),
                updates_per_step=args.updates_per_step,
                early_stop_threshold=args.early_stop_threshold,
                early_stop_window=args.early_stop_window,
                compute_latents=False,
            )
            y = logs_to_array_with_ffill(logs, "bout_success_rate", args.episodes)
            scores.append(float(np.mean(y[-args.stats_last_k:])))
        scores = np.array(scores, dtype=np.float32)
        raw_means[lr] = scores
        m = float(np.mean(scores))
        s = float(np.std(scores, ddof=1) / math.sqrt(len(scores))) if len(scores) > 1 else 0.0
        means.append(m); sems.append(s)

    fig = plt.figure(figsize=(6.8, 4.5))
    x = np.array(lrs, dtype=np.float32)
    y = np.array(means, dtype=np.float32)
    e = np.array(sems, dtype=np.float32)
    plt.errorbar(x, y, yerr=e, fmt="-o", capsize=4)
    if args.lr_sweep_logx:
        plt.xscale("log")
    plt.xlabel("learning rate")
    plt.ylabel(f"mean bout success (last {args.stats_last_k} eps)")
    plt.title("LR sweep: mean +/- SEM")
    plt.tight_layout()
    _save_fig(fig, os.path.join(base_dir, "lr_sweep_bout_success_mean_sem"), ext=args.plot_ext, pdf_pages=None)

    out_txt = os.path.join(base_dir, "lr_sweep_scores.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"LR sweep scores (metric=bout_success_rate, last_k={args.stats_last_k})\n")
        for lr in lrs:
            sc = raw_means[lr]
            f.write(f"lr={lr:g}  n={len(sc)}  mean={float(np.mean(sc)):.4f}  sem={float(np.std(sc, ddof=1)/math.sqrt(len(sc)) if len(sc)>1 else 0.0):.4f}  values={sc.tolist()}\n")
    print("Saved LR sweep plot + scores:", out_txt)

# ============================================================
# run_experiment
# ============================================================
def run_experiment(args):
    ensure_dir(args.save_dir)

    if args.lr_sweep is not None and args.lr_sweep.strip() != "":
        run_lr_sweep(args)

    if args.compare_suite:
        return run_comparison_suite_multi_seed(args)

    out_dir = os.path.join(
        args.save_dir,
        f"single_T_{args.teacher_pal}_L_{args.learner_profile}_F_{args.familiarity}_E_{args.empathy}",
    )
    runs = run_single_condition_multi_seed(
        args, args.teacher_pal, args.learner_profile, args.familiarity, args.empathy, out_dir=out_dir
    )
    return runs

# ============================================================
# Args
# ============================================================
def build_argparser():
    p = argparse.ArgumentParser(description="Active observational learning + empathy/familiarity comparisons + SEM plots + significance + LR sweep")

    # algo
    p.add_argument("--algo", type=str, default="dqn", choices=["dqn", "ppo", "tabular"])

    # run
    p.add_argument("--episodes", type=int, default=1200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seed_stride", type=int, default=1000, help="seed_i = seed + i*seed_stride")
    p.add_argument("--n_seeds", type=int, default=5, help="replicate runs per condition for SEM/statistics")
    p.add_argument("--save_dir", type=str, default="outputs")
    p.add_argument("--no_plots", action="store_true")
    p.add_argument("--plot_ext", type=str, default="pdf", choices=["pdf", "png"])
    p.add_argument("--multipage_pdf", action="store_true")
    p.add_argument("--updates_per_step", type=int, default=4)
    p.add_argument("--alpha_ema", type=float, default=0.02, help="EMA alpha used before mean +/- SEM aggregation")
    p.add_argument("--no_latents", action="store_true", help="skip latent simulation (kept; default multi-seed already skips)")

    # significance settings
    p.add_argument("--stats_last_k", type=int, default=200)
    p.add_argument("--stats_n_perm", type=int, default=5000)

    # environment
    p.add_argument("--grid_size", type=int, default=15)
    p.add_argument("--dt_s", type=float, default=0.5)
    p.add_argument("--max_time_s", type=float, default=120.0)
    p.add_argument("--teacher_period_s", type=float, default=30.0)
    p.add_argument("--teacher_jitter_s", type=float, default=6.0)
    p.add_argument("--lick_mean_s", type=float, default=1.0)
    p.add_argument("--lick_dist", type=str, default="gamma", choices=["gamma", "lognormal"])
    p.add_argument("--lick_cv", type=float, default=0.8)
    p.add_argument("--window_s", type=float, default=3.0)
    p.add_argument("--eat_cooldown_s", type=float, default=2.0)
    p.add_argument("--observe_cost", type=float, default=0.002)
    p.add_argument("--eat_cost", type=float, default=0.005)
    p.add_argument("--attend_bonus", type=float, default=0.004)
    p.add_argument("--p_detect_per_obs", type=float, default=0.70)
    p.add_argument("--win_rem_noise_steps", type=int, default=1)

    # common learning
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eps_start", type=float, default=1.0)
    p.add_argument("--eps_end", type=float, default=0.05)
    p.add_argument("--eps_decay", type=float, default=0.9995)

    # DQN
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--buffer_size", type=int, default=50000)
    p.add_argument("--tau", type=float, default=0.02)
    p.add_argument("--grad_clip", type=float, default=10.0)
    p.add_argument("--use_compile", action="store_true")

    # PPO
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--clip_ratio", type=float, default=0.2)
    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--ppo_epochs", type=int, default=4)
    p.add_argument("--minibatch_size", type=int, default=256)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    # Tabular Q
    p.add_argument("--tab_alpha", type=float, default=0.25)

    # early stopping
    p.add_argument("--early_stop_threshold", type=float, default=0.8)
    p.add_argument("--early_stop_window", type=int, default=100)

    # teacher palatability (kept; DO NOT change unless you turn on compare_teacher_pal)
    p.add_argument("--teacher_pal", type=str, default="default", choices=["default", "high", "low"])
    p.add_argument("--pal_high_lick_mean_s", type=float, default=1.4)
    p.add_argument("--pal_low_lick_mean_s", type=float, default=0.6)

    # learner
    p.add_argument("--learner_profile", type=str, default="normal", choices=["normal", "blind", "autistic"])

    # familiarity toggle
    p.add_argument("--familiarity", type=str, default="off", choices=["off", "on"])

    # empathy toggle
    p.add_argument("--empathy", type=str, default="off", choices=["off", "on"])
    p.add_argument("--empathy_seen_reward", type=float, default=0.05)

    # familiarity baseline (normal)
    p.add_argument("--fam_init", type=float, default=0.0)
    p.add_argument("--fam_decay_ep", type=float, default=0.00)
    p.add_argument("--fam_gain_obs", type=float, default=0.01)
    p.add_argument("--fam_gain_detect", type=float, default=0.04)
    p.add_argument("--p_detect_fam_boost", type=float, default=0.20)
    p.add_argument("--noise_fam_reduction", type=float, default=0.70)

    # familiarity -> behavior shaping (requested)
    p.add_argument("--fam_obs_cost_reduction", type=float, default=0.50)
    p.add_argument("--fam_attend_bonus_boost", type=float, default=0.50)
    p.add_argument("--fam_move_shaping", type=float, default=0.010)
    p.add_argument("--fam_obs_during_window_penalty", type=float, default=0.002)
    p.add_argument("--fam_latency_bonus", type=float, default=0.20)

    # autistic overrides
    p.add_argument("--aut_p_detect_per_obs", type=float, default=0.25)
    p.add_argument("--aut_win_rem_noise_steps", type=int, default=3)
    p.add_argument("--aut_detect_memory_steps", type=int, default=12)
    p.add_argument("--aut_observe_cost", type=float, default=0.004)

    p.add_argument("--aut_fam_gain_obs", type=float, default=0.006)
    p.add_argument("--aut_fam_gain_detect", type=float, default=0.020)
    p.add_argument("--aut_p_detect_fam_boost", type=float, default=0.12)
    p.add_argument("--aut_noise_fam_reduction", type=float, default=0.45)

    # comparison suite switches
    p.add_argument("--compare_suite", action="store_true")
    p.add_argument("--compare_profiles", action="store_true")
    p.add_argument("--compare_teacher_pal", action="store_true")
    p.add_argument("--compare_familiarity", action="store_true")
    p.add_argument("--compare_empathy", action="store_true")

    # LR sweep
    p.add_argument("--lr_sweep", type=str, default="", help="comma-separated learning rates, e.g. '1e-4,3e-4,1e-3'")
    p.add_argument("--lr_sweep_logx", action="store_true")

    return p

def main(cli_args=None):
    parser = build_argparser()
    args = parser.parse_args(cli_args)
    return run_experiment(args)


if __name__ == "__main__":
    main()
