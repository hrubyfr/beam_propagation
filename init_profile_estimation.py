"""
multi_run_estimation.py

Estimates the initial beam profile (σy at beam window exit) for high-energy
runs (> 750 MeV/c) using electron MCS simulation. Produces three plots:

  1. High-energy fit — divergence corrected
  2. High-energy fit — scintillator height correction only (no divergence)
  3. Comparison of both corrections

Usage:
    python multi_run_estimation.py --config /path/to/config.json
"""

import argparse
import io
import json
import math
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

import beam_setup
import layers


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

BEAM_DIVERGENCE_MRAD = 1.0  # Beam divergence in mrad
DISTANCE_WINDOW_TO_T5_M = 6.5306  # Distance from beam window to T5 detector (m)
SCINT_SIZE_Y_CM = 1.625  # Scintillator height (cm)
HIGH_ENERGY_THRESHOLD_MEV_C = 750  # High-energy cut (MeV/c)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_config(config_path: str) -> list:
    path = Path(config_path)
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        raise ValueError(f"ERROR: config file '{path}' not found")


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------


def simulate_mcs(run_number: int, momentum_MeV_c: float) -> float:
    """Run electron MCS simulation and return σ (cm)."""
    detectors_list = layers.return_built_beam(run_number=run_number)
    beamline = beam_setup.beamline_setup(name="mcs_sim")
    part = beam_setup.particle(name="electron", momentum_MeV_c=momentum_MeV_c)
    beamline.set_particle(part)
    beamline.create_setup(detectors_list)
    _stdout = sys.stdout
    sys.stdout = io.StringIO()
    width_variance, _ = beamline.get_final_variance(HC_veto=True)
    sys.stdout = _stdout
    return math.sqrt(width_variance)


def subtract_in_quadrature(a: float, b: float) -> Optional[float]:
    """Return √(a² − b²), or None if the result would be imaginary."""
    diff = a**2 - b**2
    return math.sqrt(diff) if diff >= 0 else None


def divergence_contribution_cm(divergence_mrad: float, distance_m: float) -> float:
    """Beam-size contribution from divergence, in cm."""
    return (divergence_mrad / 1000.0) * distance_m * 100.0


# ---------------------------------------------------------------------------
# Fitting helpers
# ---------------------------------------------------------------------------


def fit_constant_weighted(
    y_values: list[float], y_errors: list[float]
) -> tuple[float, float, float]:
    """
    Weighted least-squares fit of a constant to data.
    Returns (constant, stat_uncertainty, data_uncertainty), where:
      - stat_uncertainty  is derived from the individual error bars (1/√Σw)
      - data_uncertainty  is derived from the scatter of the data points around
                          the fitted constant (RMS residual / √N), which tends
                          to be larger and more conservative.
    Falls back to unweighted mean when all errors are invalid.
    """
    y = np.array(y_values, dtype=float)
    e = np.array(y_errors, dtype=float)
    n = len(y)

    valid = (e > 0) & np.isfinite(e)
    if not np.any(valid):
        print("WARNING: All errors invalid — falling back to unweighted fit.")
        constant = float(np.mean(y))
        stat_unc = float(np.std(y, ddof=1) / np.sqrt(n))
        data_unc = stat_unc
        return constant, stat_unc, data_unc

    w = 1.0 / e[valid] ** 2
    constant = float(np.sum(w * y[valid]) / np.sum(w))
    stat_unc = float(np.sqrt(1.0 / np.sum(w)))
    data_unc = float(np.sqrt(np.sum((y - constant) ** 2) / n))
    return constant, stat_unc, data_unc


def chi_squared(observed: list[float], expected: float, errors: list[float]) -> float:
    return sum(
        ((o - expected) / err) ** 2
        for o, err in zip(observed, errors)
        if err and err > 0
    )



# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _base_axes(ax: plt.Axes, momenta: list[float]) -> list[float]:
    """Apply common grid / tick styling and return the x-range."""
    x_range = [min(momenta) - 50, max(momenta) + 50]
    ax.set_xlim(x_range)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.grid(which="minor", linestyle=":", linewidth=0.3, alpha=0.4)
    ax.set_xlabel("Nominal beam momentum (MeV/c)", fontsize=12)
    ax.set_ylabel(r"Initial $\sigma_y$ (cm)", fontsize=12)
    return x_range


def plot_single_fit(
    momenta: list[float],
    sigma: list[float],
    sigma_errors: list[float],
    fitted: float,
    unc_stat: float,
    unc_data: float,
    chi_sq: float,
    title: str,
    point_label: str,
    color: str = "#e6194b",
) -> plt.Figure:
    """
    Plot a single series with a fitted constant and two uncertainty bands:
      - inner (darker): statistical uncertainty from individual error bars
      - outer (lighter): uncertainty from scatter of data points around the fit
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_ylim(0, 5)

    ax.errorbar(
        momenta,
        sigma,
        yerr=sigma_errors,
        fmt="o",
        color=color,
        markersize=8,
        capsize=5,
        capthick=2,
        linewidth=1.5,
        label=point_label,
    )

    x_range = _base_axes(ax, momenta)
    ax.axhline(
        y=fitted,
        color="black",
        linestyle="-",
        linewidth=2,
        label=f"Fit: {fitted:.4f} ± {unc_data:.4f} cm (data scatter)",
    )
    ax.fill_between(
        x_range,
        fitted - unc_data,
        fitted + unc_data,
        alpha=0.15,
        color="gray",
        label=f"Unc. from data scatter: ±{unc_data:.4f} cm",
    )
    ax.fill_between(
        x_range,
        fitted - unc_stat,
        fitted + unc_stat,
        alpha=0.35,
        color="gray",
        label=f"Unc. from error bars: ±{unc_stat:.4f} cm",
    )
    ax.text(
        0.05,
        0.95,
        f"χ² = {chi_sq:.2f}, NDF = {len(sigma) - 1}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
    )
    ax.set_title(title, fontsize=12, pad=10)
    ax.legend(loc="best", fontsize=10, framealpha=0.9)
    fig.tight_layout()
    return fig


def plot_comparison(
    momenta: list[float],
    sigma_scint: list[float],
    sigma_nodiv: list[float],
    sigma_errors_scint: list[float],
    sigma_errors_nodiv: list[float],
    fitted_scint: float,
    unc_data_scint: float,
    fitted_nodiv: float,
    unc_data_nodiv: float,
    title: str,
) -> plt.Figure:
    """
    Overlay scintillator-only and divergence-corrected series.
    Uncertainty bands show the data-scatter uncertainty.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(
        momenta,
        sigma_scint,
        yerr=sigma_errors_scint,
        fmt="o",
        color="#4363d8",
        markersize=8,
        capsize=5,
        capthick=2,
        linewidth=1.5,
        label="σy,init (scintillator corrected)",
    )
    ax.errorbar(
        momenta,
        sigma_nodiv,
        yerr=sigma_errors_nodiv,
        fmt="s",
        color="#e6194b",
        markersize=8,
        capsize=5,
        capthick=2,
        linewidth=1.5,
        label="σy,init (divergence corrected)",
    )

    x_range = _base_axes(ax, momenta)
    ax.axhline(
        y=fitted_scint,
        color="#4363d8",
        linestyle="--",
        linewidth=2,
        label=f"Fit (scint. only): {fitted_scint:.4f} ± {unc_data_scint:.4f} cm",
    )
    ax.axhline(
        y=fitted_nodiv,
        color="#e6194b",
        linestyle=":",
        linewidth=2,
        label=f"Fit (div. corrected): {fitted_nodiv:.4f} ± {unc_data_nodiv:.4f} cm",
    )
    ax.fill_between(
        x_range,
        fitted_scint - unc_data_scint,
        fitted_scint + unc_data_scint,
        alpha=0.1,
        color="#4363d8",
    )
    ax.fill_between(
        x_range,
        fitted_nodiv - unc_data_nodiv,
        fitted_nodiv + unc_data_nodiv,
        alpha=0.1,
        color="#e6194b",
    )

    ax.set_title(title, fontsize=12, pad=10)
    ax.legend(loc="best", fontsize=10, framealpha=0.9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Estimate initial beam profile — high-energy fit, with and without divergence correction."
    )
    parser.add_argument("--config", required=True, help="Path to config.json")
    parser.add_argument(
        "--output",
        default="initial_beam_profile_fit.png",
        help="Base output path for plots (default: initial_beam_profile_fit.png)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # ------------------------------------------------------------------
    # Filter valid runs
    # ------------------------------------------------------------------
    VALID_MOMENTUM_RANGES = [(0, 450), (750, 10000)]

    valid_runs = []
    for cf_run in config:
        if not cf_run.get("run_number"):
            continue
        rn = int(cf_run["run_number"])

        if "hadron" not in cf_run.get("beam_config", "").lower():
            continue

        beam_cfg = cf_run.get("beam_config", "").lower()
        if "tagged" in beam_cfg and "gamma" in beam_cfg:
            continue

        if "T5_beam_sigma_x" not in cf_run or "T5_beam_sigma_y" not in cf_run:
            print(f"NOTE: run {rn} has no T5 sigma data, skipping.")
            continue

        if "lemb" in cf_run.get("trigger_config", "").lower():
            continue

        if (
            "T5_beam_sigma_x_error" not in cf_run
            or "T5_beam_sigma_y_error" not in cf_run
        ):
            print(f"NOTE: run {rn} has no T5 sigma error data, skipping.")
            continue

        if not cf_run.get("beam_momentum"):
            print(f"NOTE: run {rn} has no beam momentum, skipping.")
            continue

        p = float(cf_run["beam_momentum"])
        if not any(lo <= p <= hi for lo, hi in VALID_MOMENTUM_RANGES):
            continue

        valid_runs.append((rn, p, cf_run))

    print(
        f"Found {len(valid_runs)} runs in valid momentum ranges (≤450 or ≥750 MeV/c)\n"
    )

    # ------------------------------------------------------------------
    # Process runs
    # ------------------------------------------------------------------
    momenta_data = []
    sigma_y_init_from_scint = []
    sigma_y_init_from_scint_errors = []
    sigma_y_init_from_nodiv = []
    sigma_y_init_from_nodiv_errors = []

    div_contrib_cm = divergence_contribution_cm(
        BEAM_DIVERGENCE_MRAD, DISTANCE_WINDOW_TO_T5_M
    )

    for rn, momentum, cf_run in valid_runs:
        if (
            cf_run.get("T5_beam_sigma_y") is None
            or cf_run.get("T5_beam_sigma_y_error") is None
        ):
            print(f"NOTE: run {rn} has None values in T5 sigma data, skipping.")
            continue

        try:
            sy_meas = float(cf_run["T5_beam_sigma_y"]) * 0.1  # mm → cm
            sy_meas_err = float(cf_run["T5_beam_sigma_y_error"]) * 0.1
        except (TypeError, ValueError) as e:
            print(f"NOTE: run {rn} has invalid T5 sigma data ({e}), skipping.")
            continue

        # Scintillator height correction
        sy_scint = subtract_in_quadrature(sy_meas, SCINT_SIZE_Y_CM / math.sqrt(12))
        if sy_scint is None:
            print(
                f"Run {rn}: scintillator correction yields imaginary result, skipping."
            )
            continue
        sy_scint_err = (sy_meas_err * sy_meas / sy_scint) if sy_scint > 0 else None

        # Divergence correction (applied on top of scintillator correction)
        sy_nodiv = subtract_in_quadrature(sy_scint, div_contrib_cm)
        if sy_nodiv is None:
            print(
                f"Run {rn} (p={momentum:.1f} MeV/c): divergence ({div_contrib_cm:.4f} cm) "
                f"exceeds corrected σy ({sy_scint:.4f} cm), skipping."
            )
            continue

        print(
            f"Run {rn} (p={momentum:6.1f} MeV/c) → "
            f"σy_meas={sy_meas:.4f} cm, σy_scint={sy_scint:.4f} cm, σy_nodiv={sy_nodiv:.4f} cm ...",
            flush=True,
        )

        try:
            sigma_mcs_e = simulate_mcs(rn, momentum)
        except Exception as exc:
            print(f"  WARNING: electron simulation failed for run {rn}: {exc}")
            continue

        init_scint = subtract_in_quadrature(sy_scint, sigma_mcs_e)
        init_nodiv = subtract_in_quadrature(sy_nodiv, sigma_mcs_e)

        if init_nodiv is None:
            print(f"  MCS σ exceeds σy_nodiv for run {rn}, skipping.")
            continue

        # Error propagation: σ_init = √(σ_corr² − σ_MCS²)  →  dσ_init = σ_corr/σ_init · dσ_corr
        err_scint = (
            (sy_scint_err * sy_scint / init_scint)
            if (sy_scint_err and init_scint > 0)
            else None
        )
        err_nodiv = (
            (sy_scint_err * sy_scint / init_nodiv)
            if (sy_scint_err and init_nodiv > 0)
            else None
        )

        momenta_data.append(momentum)
        sigma_y_init_from_scint.append(init_scint)
        sigma_y_init_from_scint_errors.append(err_scint)
        sigma_y_init_from_nodiv.append(init_nodiv)
        sigma_y_init_from_nodiv_errors.append(err_nodiv)

        print(
            f"  MCS_e={sigma_mcs_e:.4f} cm → σy_init(scint)={init_scint:.4f} cm, σy_init(no_div)={init_nodiv:.4f} cm"
        )

    # ------------------------------------------------------------------
    # High-energy subset
    # ------------------------------------------------------------------
    mask = [p > HIGH_ENERGY_THRESHOLD_MEV_C for p in momenta_data]
    momenta_he = [p for p, m in zip(momenta_data, mask) if m]
    scint_he = [s for s, m in zip(sigma_y_init_from_scint, mask) if m]
    scint_err_he = [e for e, m in zip(sigma_y_init_from_scint_errors, mask) if m]
    nodiv_he = [s for s, m in zip(sigma_y_init_from_nodiv, mask) if m]
    nodiv_err_he = [e for e, m in zip(sigma_y_init_from_nodiv_errors, mask) if m]

    if not momenta_he:
        print("ERROR: No valid high-energy data points found for fitting!")
        return

    print(
        f"\nFitting to HIGH-ENERGY runs (> {HIGH_ENERGY_THRESHOLD_MEV_C} MeV/c): {len(momenta_he)} runs\n"
    )

    fitted_scint, unc_stat_scint, unc_data_scint = fit_constant_weighted(
        scint_he, scint_err_he
    )
    fitted_nodiv, unc_stat_nodiv, unc_data_nodiv = fit_constant_weighted(
        nodiv_he, nodiv_err_he
    )
    chi_sq_scint = chi_squared(scint_he, fitted_scint, scint_err_he)
    chi_sq_nodiv = chi_squared(nodiv_he, fitted_nodiv, nodiv_err_he)

    print(f"\n{'=' * 60}")
    print(f"FIT RESULTS — HIGH ENERGY (> {HIGH_ENERGY_THRESHOLD_MEV_C} MeV/c)")
    print(f"{'=' * 60}")
    print(f"  Scintillator only:")
    print(f"    Fitted constant:          {fitted_scint:.4f} cm")
    print(f"    Unc. from error bars:   ± {unc_stat_scint:.4f} cm")
    print(
        f"    Unc. from data scatter: ± {unc_data_scint:.4f} cm  (χ²={chi_sq_scint:.2f})"
    )
    print(f"  Divergence corrected:")
    print(f"    Fitted constant:          {fitted_nodiv:.4f} cm")
    print(f"    Unc. from error bars:   ± {unc_stat_nodiv:.4f} cm")
    print(
        f"    Unc. from data scatter: ± {unc_data_nodiv:.4f} cm  (χ²={chi_sq_nodiv:.2f})"
    )
    print(f"  Difference:           {fitted_scint - fitted_nodiv:.4f} cm")
    print(f"  N data points:        {len(momenta_he)}")
    print(f"{'=' * 60}\n")

    # ------------------------------------------------------------------
    # Three plots
    # ------------------------------------------------------------------
    suffix = f"_{HIGH_ENERGY_THRESHOLD_MEV_C}plus_MeV_c"

    # Plot 1: Divergence-corrected fit
    fig = plot_single_fit(
        momenta_he,
        nodiv_he,
        nodiv_err_he,
        fitted_nodiv,
        unc_stat_nodiv,
        unc_data_nodiv,
        chi_sq_nodiv,
        title=f"Initial Beam Profile — Divergence Corrected (> {HIGH_ENERGY_THRESHOLD_MEV_C} MeV/c)",
        point_label=r"$\sigma_{y,\mathrm{init}}$ (divergence corrected)",
        color="#e6194b",
    )
    path1 = args.output.replace(".png", f"{suffix}_divergence_corrected.png")
    fig.savefig(path1, dpi=150)
    print(f"Plot saved: {path1}")
    plt.close(fig)

    # Plot 2: Scintillator-only fit (no divergence correction)
    fig = plot_single_fit(
        momenta_he,
        scint_he,
        scint_err_he,
        fitted_scint,
        unc_stat_scint,
        unc_data_scint,
        chi_sq_scint,
        title=f"Initial Beam Profile — Scintillator Correction Only (> {HIGH_ENERGY_THRESHOLD_MEV_C} MeV/c)",
        point_label=r"$\sigma_{y,\mathrm{init}}$ (scintillator corrected)",
        color="#4363d8",
    )
    path2 = args.output.replace(".png", f"{suffix}_scint_only.png")
    fig.savefig(path2, dpi=150)
    print(f"Plot saved: {path2}")
    plt.close(fig)

    # Plot 3: Comparison
    fig = plot_comparison(
        momenta_he,
        scint_he,
        nodiv_he,
        scint_err_he,
        nodiv_err_he,
        fitted_scint,
        unc_data_scint,
        fitted_nodiv,
        unc_data_nodiv,
        title=f"Divergence Correction Comparison (> {HIGH_ENERGY_THRESHOLD_MEV_C} MeV/c)",
    )
    path3 = args.output.replace(".png", f"{suffix}_comparison.png")
    fig.savefig(path3, dpi=150)
    print(f"Plot saved: {path3}")
    plt.close(fig)


if __name__ == "__main__":
    main()
