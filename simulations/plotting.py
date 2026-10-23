"""Figures for the simulation gallery."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from simulations.econometrics import iv_versus_ols, ols_sampling, rdd_sample
from simulations.macro import nk_impulse, solow_steady_state, solow_transition, steady_consumption
from simulations.micro import (
    APPLICANTS,
    DEPARTMENTS,
    cournot_dynamics,
    cournot_nash,
    fictitious_play_matching_pennies,
    uniform_optimal_price,
    uniform_seller_profit,
)

INK = "#243044"
RED = "#b6402c"
BLUE = "#1f5f8b"
GREEN = "#2d6a4f"
GOLD = "#c48a1a"
PAPER = "#f7f3ea"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.titlecolor": INK,
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#e3d9c6",
            "grid.linewidth": 0.7,
            "lines.linewidth": 2.1,
            "figure.dpi": 140,
        }
    )


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def write_gallery(directory: Path) -> list[Path]:
    """Write every gallery figure and return the file paths."""
    _style()
    directory.mkdir(parents=True, exist_ok=True)
    writers = [
        posted_price,
        becker_crime,
        matching_pennies,
        cournot,
        solow,
        new_keynesian,
        ols_histogram,
        iv_histogram,
        regression_discontinuity,
    ]
    return [writer(directory) for writer in writers]


def posted_price(directory: Path) -> Path:
    prices = np.linspace(0, 1, 201)
    profit = [uniform_seller_profit(p, 0.0, 1.0, 0.0) for p in prices]
    optimal = uniform_optimal_price(0.0, 1.0, 0.0)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(prices, profit, color=BLUE, label="expected profit")
    ax.scatter([optimal], [uniform_seller_profit(optimal, 0.0, 1.0, 0.0)], color=RED, zorder=3, s=40)
    ax.axvline(optimal, color=RED, lw=1.2, ls="--", label="optimal price 1/2")
    ax.set_xlabel("posted price")
    ax.set_ylabel("expected profit")
    ax.set_title("Uniform values on [0, 1], zero continuation value")
    ax.legend(frameon=False)
    return _save(fig, directory / "micro_posted_price.png")


def becker_crime(directory: Path) -> Path:
    fines = np.linspace(0, 2, 200)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for probability, color in ((0.25, GOLD), (0.5, BLUE), (1.0, RED)):
        rates = [max(0.0, 1.0 - probability * fine) for fine in fines]
        ax.plot(fines, rates, color=color, label=f"detection probability {probability:g}")
    ax.set_xlabel("fine")
    ax.set_ylabel("crime rate")
    ax.set_title("Becker cutoff, net gains uniform on [0, 1]")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(frameon=False)
    return _save(fig, directory / "micro_becker_crime.png")


def matching_pennies(directory: Path) -> Path:
    frequency = fictitious_play_matching_pennies(4_000, seed=7)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(frequency[:, 0], color=BLUE, label="player 0, P(heads)")
    ax.plot(frequency[:, 1], color=RED, label="player 1, P(heads)")
    ax.axhline(0.5, color=INK, lw=1.0, ls="--", label="mixed Nash")
    ax.set_xlabel("round")
    ax.set_ylabel("empirical frequency")
    ax.set_title("Fictitious play in matching pennies")
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False)
    return _save(fig, directory / "micro_matching_pennies.png")


def cournot(directory: Path) -> Path:
    nash = cournot_nash(10.0, 1.0, 1.0)
    path = cournot_dynamics(np.array([0.4, 4.5]), 10.0, 1.0, 1.0, rounds=25)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(path[:, 0], color=BLUE, label="firm 1")
    ax.plot(path[:, 1], color=RED, label="firm 2")
    ax.axhline(nash, color=INK, lw=1.0, ls="--", label="Nash quantity")
    ax.set_xlabel("round")
    ax.set_ylabel("quantity")
    ax.set_title("Cournot best-response dynamics")
    ax.legend(frameon=False)
    return _save(fig, directory / "micro_cournot.png")


def solow(directory: Path) -> Path:
    saving, depreciation, alpha = 0.3, 0.08, 0.35
    steady = solow_steady_state(saving, depreciation, alpha)
    low = solow_transition(0.4 * steady, saving, depreciation, alpha, periods=60)
    high = solow_transition(1.8 * steady, saving, depreciation, alpha, periods=60)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2))
    axes[0].plot(low, color=BLUE, label="below steady state")
    axes[0].plot(high, color=RED, label="above steady state")
    axes[0].axhline(steady, color=INK, lw=1.0, ls="--", label="k*")
    axes[0].set_xlabel("period")
    axes[0].set_ylabel("capital")
    axes[0].set_title("Solow transition")
    axes[0].legend(frameon=False, fontsize=9)
    grid = np.linspace(0.05, 0.9, 80)
    consumption = [steady_consumption(s, depreciation, alpha) for s in grid]
    axes[1].plot(grid, consumption, color=GREEN)
    axes[1].axvline(alpha, color=RED, lw=1.2, ls="--", label="golden rule s = α")
    axes[1].set_xlabel("saving rate")
    axes[1].set_ylabel("steady consumption")
    axes[1].set_title("Golden rule")
    axes[1].legend(frameon=False, fontsize=9)
    return _save(fig, directory / "macro_solow.png")


def new_keynesian(directory: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2), sharey=True)
    for ax, shock, title in (
        (axes[0], "monetary", "Contractionary policy shock"),
        (axes[1], "demand", "Demand shock"),
    ):
        paths = nk_impulse(shock=shock, horizon=20)
        ax.plot(paths["output"], color=BLUE, label="output gap")
        ax.plot(paths["inflation"], color=RED, label="inflation")
        ax.plot(paths["nominal_rate"], color=GOLD, label="nominal rate")
        ax.axhline(0.0, color=INK, lw=0.8)
        ax.set_xlabel("quarter")
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=9)
    axes[0].set_ylabel("deviation")
    return _save(fig, directory / "macro_new_keynesian.png")


def ols_histogram(directory: Path) -> Path:
    slopes = ols_sampling()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.hist(slopes, bins=30, color=BLUE, edgecolor=PAPER)
    ax.axvline(2.0, color=RED, ls="--", label="true slope 2")
    ax.axvline(float(np.mean(slopes)), color=GOLD, label="Monte Carlo mean")
    ax.set_xlabel("OLS slope")
    ax.set_ylabel("replications")
    ax.set_title("Sampling distribution, correctly specified OLS")
    ax.legend(frameon=False)
    return _save(fig, directory / "econometrics_ols.png")


def iv_histogram(directory: Path) -> Path:
    ols, iv = iv_versus_ols()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.hist(ols, bins=28, color=RED, alpha=0.75, edgecolor=PAPER, label="OLS")
    ax.hist(iv, bins=28, color=BLUE, alpha=0.75, edgecolor=PAPER, label="2SLS")
    ax.axvline(2.0, color=INK, ls="--", label="true slope 2")
    ax.set_xlabel("estimated slope")
    ax.set_ylabel("replications")
    ax.set_title("Endogenous regressor: OLS versus 2SLS")
    ax.legend(frameon=False)
    return _save(fig, directory / "econometrics_iv.png")


def regression_discontinuity(directory: Path) -> Path:
    running, outcome = rdd_sample(n=700, seed=3)
    order = np.argsort(running)
    running, outcome = running[order], outcome[order]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.scatter(running, outcome, s=10, color=BLUE, alpha=0.35, linewidths=0)
    edges = np.concatenate([np.linspace(-1.0, 0.0, 8), np.linspace(0.0, 1.0, 8)[1:]])
    for side, label in (("left", "bin means"), ("right", None)):
        centers = []
        means = []
        for left, right in zip(edges[:-1], edges[1:]):
            if side == "left" and right > 0:
                continue
            if side == "right" and left < 0:
                continue
            mask = (running >= left) & (running < right)
            if right == edges[-1]:
                mask = (running >= left) & (running <= right)
            if mask.sum() == 0:
                continue
            centers.append(0.5 * (left + right))
            means.append(float(outcome[mask].mean()))
        ax.plot(centers, means, color=RED, marker="o", ms=4, label=label)
    ax.axvline(0.0, color=INK, lw=1.0, ls="--", label="cutoff")
    ax.set_xlabel("running variable")
    ax.set_ylabel("outcome")
    ax.set_title("Sharp regression discontinuity")
    ax.legend(frameon=False)
    return _save(fig, directory / "econometrics_rdd.png")
