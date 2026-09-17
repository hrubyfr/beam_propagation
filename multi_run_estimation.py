"""
band_survey.py

Picks the lowest run number from each 200 MeV/c momentum band
(200-400, 400-600, ..., 1400-1600) and runs the MCS simulation
for electrons, muons and pions, showing how both measured sigma
and simulated MCS width evolve with beam momentum.

Usage:
    python band_survey.py --config /path/to/config.json
"""

import argparse
import io
import csv
import json
import math
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy import optimize

import beam_setup
import layers


MOMENTUM_BANDS = [(lo, lo + 50) for lo in range(200, 650, 50)] + [
    (lo, lo + 100) for lo in range(650, 2000, 100)
]
PARTICLE_NAMES = ["electron", "muon", "pion", "proton"]

# Beam divergence parameters
BEAM_DIVERGENCE_MRAD = 1.0  # Beam divergence in mrad
DISTANCE_WINDOW_TO_T5_M = (
    6.5306  # Distance from beam window (0 m) to T5 detector (6.5306 m)
)


def calculate_chi_squared(
    observed: list[float], expected: list[float], errors: Optional[list[float]] = None
) -> float:
    if errors is None:
        # If no errors provided, assume equal weighting
        return sum((o - e) ** 2 for o, e in zip(observed, expected))
    else:
        return sum(
            ((o - e) / err) ** 2
            for o, e, err in zip(observed, expected, errors)
            if err > 0
        )


def calculate_fit_uncertainty(observed: list[float], expected: float) -> float:
    n_data_points = np.size(observed)
    sigma_constant = math.sqrt(
        1
        / (n_data_points * (n_data_points - 1))
        * sum((o - expected) ** 2 for o in observed)
    )
    return sigma_constant


def load_config(config_path: str) -> list:
    path = Path(config_path)
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        raise ValueError(f"ERROR: config file '{path}' not found")

def get_T5_resolution_cm(T5_config_path: str) -> tuple[float, float]:
    """
    Load the T5 detector configuration and return the resolution and its
    propagated uncertainty in cm.
    """
    path = Path(T5_config_path)
    try:
        with open(path) as f:
            T5_config = json.load(f)
            v_eff = float(T5_config["v_eff"])
            sigma_sipm_ns = float(T5_config["sigma_sipm_7"])

            v_eff_uncertainty = float(
                T5_config.get("v_eff_uncertainty", T5_config.get("v_eff_error", 0.0))
            )
            sigma_sipm_ns_uncertainty = float(
                T5_config.get(
                    "sigma_sipm_7_uncertainty",
                    T5_config.get(
                        "sigma_sipm_ns_uncertainty",
                        T5_config.get("sigma_sipm_uncertainty", 0.0),
                    ),
                )
            )

            # Resolution: R = (v_eff * sigma_sipm_ns) / 2, converted from mm to cm.
            resolution_cm = 0.05 * v_eff * sigma_sipm_ns

            # Propagate uncertainty from both inputs.
            derivative_wrt_v_eff = 0.05 * sigma_sipm_ns
            derivative_wrt_sigma_sipm_ns = 0.05 * v_eff
            resolution_uncertainty_cm = math.sqrt(
                (derivative_wrt_v_eff * v_eff_uncertainty) ** 2
                + (derivative_wrt_sigma_sipm_ns * sigma_sipm_ns_uncertainty) ** 2
            )

            return resolution_cm, resolution_uncertainty_cm
    except FileNotFoundError:
        raise ValueError(f"ERROR: T5 config file '{path}' not found")
    
def simulate_mcs(
    run_number: int, particle_name: str, momentum_MeV_c: float, HC_veto: bool = True
) -> float:
    _stdout = sys.stdout
    sys.stdout = io.StringIO()
    detectors_list = layers.return_built_beam(run_number=run_number)
    beamline = beam_setup.beamline_setup(name="mcs_sim")
    part = beam_setup.particle(name=particle_name, momentum_MeV_c=momentum_MeV_c)
    beamline.set_particle(part)
    beamline.create_setup(detectors_list)
    try:
        # variance = beamline.get_final_variance()
        variance = beamline.get_final_variance(HC_veto=HC_veto)
    finally:
        sys.stdout = _stdout
    return math.sqrt(variance)


def initial_sigma(sigma_meas_cm: float, sigma_mcs_cm: float) -> Optional[float]:
    diff = sigma_meas_cm**2 - sigma_mcs_cm**2
    return math.sqrt(diff) if diff >= 0 else None


def calculate_true_sigma_y(sigmas_y_meas_cm: list[float]) -> list[float]:
    scint_size_y_cm = 1.625
    return [
        math.sqrt(sigma_y_meas_cm**2 - scint_size_y_cm**2 / 12)
        for sigma_y_meas_cm in sigmas_y_meas_cm
    ]


def calculate_divergence_contribution(
    divergence_mrad: float, distance_m: float
) -> float:
    """
    Calculate the beam size contribution from divergence.

    Args:
        divergence_mrad: Beam divergence in milliradians
        distance_m: Distance traveled in meters

    Returns:
        Divergence contribution to beam size in cm
    """
    divergence_rad = divergence_mrad / 1000.0
    return divergence_rad * distance_m * 100.0  # Convert to cm


def fit_constant_least_squares(
    x_values: list[float], y_values: list[float], y_errors: Optional[list[float]] = None
) -> tuple[float, float]:
    """
    Fit a constant (horizontal line) to data using least-squares method.

    Args:
        x_values: Independent variable (not used for fitting, just for reference)
        y_values: Dependent variable values
        y_errors: Optional errors on y values (for weighted least-squares)

    Returns:
        Tuple of (fitted_constant, uncertainty)
    """
    y_array = np.array(y_values)

    if y_errors is None:
        # Unweighted least-squares: constant = mean
        constant = np.mean(y_array)
        # Uncertainty from variance
        variance = np.var(y_array, ddof=1)
        uncertainty = np.sqrt(variance / len(y_array))
    else:
        # Check if all errors are valid (not None, not zero, not NaN)
        y_err_array = np.array(y_errors, dtype=float)

        # Filter out invalid errors (zeros, NaNs, infs)
        valid_mask = (y_err_array > 0) & np.isfinite(y_err_array)

        if not np.any(valid_mask):
            # If no valid errors, fall back to unweighted
            print(
                "WARNING: All error values are invalid (zero, NaN, or inf). Using unweighted fit."
            )
            constant = np.mean(y_array)
            variance = np.var(y_array, ddof=1)
            uncertainty = np.sqrt(variance / len(y_array))
        else:
            # Filter to only valid points
            y_valid = y_array[valid_mask]
            y_err_valid = y_err_array[valid_mask]

            # Weighted least-squares
            weights = 1.0 / (y_err_valid**2)

            # Weighted mean
            constant = np.sum(weights * y_valid) / np.sum(weights)

            # Uncertainty in weighted mean
            uncertainty = np.sqrt(1.0 / np.sum(weights))

    return constant, uncertainty


def calculate_highland_systematic_uncertainty(MCS_estimation: float):
    # assuming the Highland theta uncertainty is 11 %
    return 0.11 * MCS_estimation


def combine_uncertainties(
    sigma_init: float,
    sigma_measured: float,
    sigma_MCS: float,
    statistical_uncertainty: float,
    systematic_uncertainty: float,
) -> float:
    """
    Combine statistical and systematic uncertainties in quadrature.

    Args:
        sigma_init: Initial beam size (σ_init)
        sigma_measured: Measured beam size (σ_measured)
        sigma_MCS: MCS beam size (σ_MCS)
        statistical_uncertainty: Statistical uncertainty (σ_stat)
        systematic_uncertainty: Systematic uncertainty (σ_sys)

    Returns:
        Combined uncertainty (σ_total)
    """
    return (
        1
        / sigma_init
        * math.sqrt(
            statistical_uncertainty**2 * sigma_measured**2
            + systematic_uncertainty**2 * sigma_MCS**2
        )
    )


def fmt_f(val: Optional[float], w: int = 10, d: int = 4) -> str:
    return ("N/A" if val is None else f"{val:.{d}f}").rjust(w)


def fmt_i(val: int, w: int = 7) -> str:
    return str(val).rjust(w)


def fmt_band(lo: int, hi: int, w: int = 11) -> str:
    return f"{lo}-{hi}".rjust(w)


def print_table(rows: list[dict]):
    # Column layout
    # Band | Run | p_nom | sx_meas | sy_meas
    # | mcs_e | mcs_mu | mcs_pi | mcs_p
    # | s0x_e | s0x_mu | s0x_pi | s0x_p

    cols = [
        ("Band[MeV/c]", 11),
        ("Run", 7),
        ("p[MeV/c]", 9),
        ("σx_meas", 10),
        ("σy_meas", 10),
        ("σ_MCS_e", 10),
        ("σ_MCS_μ", 10),
        ("σ_MCS_π", 10),
        ("σ_MCS_p", 10),
        ("σ0x_e", 10),
        ("σ0x_μ", 10),
        ("σ0x_π", 10),
        ("σ0x_p", 10),
    ]
    units_row = [
        ("", 11),
        ("", 7),
        ("", 9),
        ("[cm]", 10),
        ("[cm]", 10),
        ("[cm]", 10),
        ("[cm]", 10),
        ("[cm]", 10),
        ("[cm]", 10),
        ("[cm]", 10),
        ("[cm]", 10),
        ("[cm]", 10),
        ("[cm]", 10),
    ]

    sep = "+" + "+".join("-" * (w + 2) for _, w in cols) + "+"

    def row_line(cells):
        return (
            "|" + "|".join(f" {c.center(w)} " for c, (_, w) in zip(cells, cols)) + "|"
        )

    print(sep)
    print(row_line([label for label, _ in cols]))
    print(row_line([unit for unit, _ in units_row]))
    print(sep)

    for r in rows:
        cells = [
            fmt_band(r["band_lo"], r["band_hi"], 11),
            fmt_i(r["run"], 7),
            fmt_f(r["p_nom"], 9, 1),
            fmt_f(r["sx_meas"], 10),
            fmt_f(r["sy_meas"], 10),
            fmt_f(r["mcs_e"], 10),
            fmt_f(r["mcs_mu"], 10),
            fmt_f(r["mcs_pi"], 10),
            fmt_f(r["mcs_proton"], 10),
            fmt_f(r.get("s0x_e"), 10),
            fmt_f(r.get("s0x_mu"), 10),
            fmt_f(r.get("s0x_pi"), 10),
            fmt_f(r.get("s0x_proton"), 10),
        ]
        print(row_line(cells))

    print(sep)


def plot_MCS_data_comparison(
    rows: list[dict],
    output_path: str = "band_survey_plot.png",
    plot_estimations: bool = True,
    plot_MCS_sim: bool = True,
    plot_bias_line=True,
    plot_sigma_init: bool = False,
    sipm_resolution_cm: Optional[float] = None,
    y_limits: Optional[tuple[float, float]] = (0, 15),
    filename_suffix: str = "",
    mcs_correction_mode: str = "both",
    plot_y_corrected_estimation: bool = False,
    plot_x_corrected_estimation: bool = False,
    plot_sipm_resolution_line: bool = False,
    plot_annotations: bool = False,
):
    """
    Plot σx_meas, σy_meas and σ_MCS for each particle species
    as a function of nominal beam momentum.

    Only momenta in [0, 400] or [700, inf) are plotted, with one
    representative point per 50 MeV/c band (the first run encountered
    in each band).
    """
    # --- validity filter + 50 MeV/c binning ---
    VALID_RANGES = [(0, float("inf"))]
    BAND_WIDTH = 50  # MeV/c

    def in_valid_range(p):
        return any(lo <= p <= hi for lo, hi in VALID_RANGES)

    def band_index(p):
        return int(p // BAND_WIDTH)

    seen_bands: dict[int, dict] = {}
    for r in rows:
        p = r["p_nom"]
        if not in_valid_range(p):
            continue
        b = band_index(p)
        if b not in seen_bands:
            seen_bands[b] = r  # keep the first run in each band

    # Sort by momentum so lines connect left-to-right
    binned_rows = sorted(seen_bands.values(), key=lambda r: r["p_nom"])

    momenta = [r["p_nom"] for r in binned_rows]
    sx_meas = [r["sx_meas"] for r in binned_rows]
    sy_meas = [r["sy_meas"] for r in binned_rows]
    sx_meas_err = [r["sx_meas_err"] for r in binned_rows]
    sy_meas_err = [r["sy_meas_err"] for r in binned_rows]

    sy_corrected = [r["sy_corrected"] for r in binned_rows]
    sy_corrected_err = [r["sy_corrected_err"] for r in binned_rows]

    sx_corrected = [r["sx_corrected"] for r in binned_rows]
    sx_corrected_err = [r["sx_corrected_err"] for r in binned_rows]

    mcs_e = [r.get("mcs_e") for r in binned_rows]
    mcs_mu = [r.get("mcs_mu") for r in binned_rows]
    mcs_pi = [r.get("mcs_pi") for r in binned_rows]
    mcs_proton = [r.get("mcs_proton") for r in binned_rows]
    uncertainty_mcs_e = [r.get("uncertainty_mcs_e") for r in binned_rows]
    uncertainty_mcs_mu = [r.get("uncertainty_mcs_mu") for r in binned_rows]
    uncertainty_mcs_pi = [r.get("uncertainty_mcs_pi") for r in binned_rows]
    uncertainty_mcs_proton = [r.get("uncertainty_mcs_proton") for r in binned_rows]

    mcs_nohc_e = [r.get("mcs_nohc_e") for r in binned_rows]
    mcs_nohc_mu = [r.get("mcs_nohc_mu") for r in binned_rows]
    mcs_nohc_pi = [r.get("mcs_nohc_pi") for r in binned_rows]
    mcs_nohc_proton = [r.get("mcs_nohc_proton") for r in binned_rows]
    uncertainty_mcs_nohc_e = [r.get("uncertainty_mcs_nohc_e") for r in binned_rows]
    uncertainty_mcs_nohc_mu = [r.get("uncertainty_mcs_nohc_mu") for r in binned_rows]
    uncertainty_mcs_nohc_pi = [r.get("uncertainty_mcs_nohc_pi") for r in binned_rows]
    uncertainty_mcs_nohc_proton = [r.get("uncertainty_mcs_nohc_proton") for r in binned_rows]

    # Filter out None values per series for clean line drawing
    def valid(xs, ys, errs=None):
        if errs is None:
            pairs = [(x, y) for x, y in zip(xs, ys) if y is not None]
            return zip(*pairs) if pairs else ([], [])
        else:
            triples = [
                (x, y, e)
                for x, y, e in zip(xs, ys, errs)
                if y is not None and e is not None
            ]
            return zip(*triples) if triples else ([], [], [])

    fig, ax = plt.subplots(figsize=(12, 8))

    if mcs_correction_mode not in {"both", "with_hc", "no_hc"}:
        raise ValueError(
            "mcs_correction_mode must be one of: 'both', 'with_hc', 'no_hc'"
        )


    # --- measured sigmas (markers + line) ---
    ax.errorbar(
        momenta,
        sx_meas,
        yerr=sx_meas_err,
        color="black",
        marker="o",
        linewidth=1.8,
        markersize=6,
        label=r"$\sigma_{x}^{\mathrm{meas}}$",
    )
    ax.errorbar(
        momenta,
        sy_meas,
        yerr=sy_meas_err,
        color="black",
        marker="s",
        linewidth=1.8,
        markersize=6,
        linestyle="--",
        label=r"$\sigma_{y}^{\mathrm{meas}}$",
    )

    if plot_y_corrected_estimation:
        ax.errorbar(
            momenta,
            sy_corrected,
            yerr=sy_corrected_err,
            color="gray",
            marker="^",
            linewidth=1.8,
            markersize=6,
            linestyle="--",
            label=r"$\sigma_{y}^{\mathrm{true}}(estimated)$",
        )

    # corrected values
    if plot_x_corrected_estimation:
        sx_corr_x, sx_corr_y, sx_corr_err = valid(
            momenta, sx_corrected, sx_corrected_err
        )

        if sipm_resolution_cm is not None:
            ax.axhline(
                y=sipm_resolution_cm,
                color="dimgray",
                linestyle=":",
                linewidth=1.5,
                label=rf"SiPM resolution = {sipm_resolution_cm:.4f} cm",
            )


        if plot_x_corrected_estimation and len(sx_corr_x) > 0:
            ax.errorbar(
                list(sx_corr_x),
                list(sx_corr_y),
                yerr=list(sx_corr_err),
                color="gray",
                marker="o",
                linewidth=1.8,
                markersize=6,
                label=r"$\sigma_{x}^{\mathrm{true}}(estimated)$",
            )
        ax.errorbar(
            momenta,
            sy_corrected,
            yerr=sy_corrected_err,
            color="gray",
            marker="^",
            linewidth=1.8,
            markersize=6,
            linestyle="--",
            label=r"$\sigma_{y}^{\mathrm{true}}(estimated)$",
        )


        # ax.plot(
        #     momenta,
        #     sx_meas_corr,
        #     color="green",
        #     marker="v",
        #     linewidth=1.8,
        #     markersize=6,
        #     linestyle="--",
        #     label=r"$\sigma_{x}^{\mathrm{true}}(estimated)$",
        # )

    # --- MCS per particle (coloured, dashed) ---
    if plot_MCS_sim:
        COLOURS = {
            "electron": "#e6194b",
            "muon": "#4363d8",
            "pion": "#f58231",
            "proton": "#3cb44b",
        }
        LABELS_WITH_HC = {
            "electron": r"$\sigma_{\mathrm{MCS}}^{e}$ (sim)",
            "muon": r"$\sigma_{\mathrm{MCS}}^{\mu}$ (sim)",
            "pion": r"$\sigma_{\mathrm{MCS}}^{\pi}$ (sim)",
            "proton": r"$\sigma_{\mathrm{MCS}}^{p}$ (sim)",
        }
        LABELS_NO_HC = {
            "electron": r"$\sigma_{\mathrm{MCS}}^{e}$ (sim)",
            "muon": r"$\sigma_{\mathrm{MCS}}^{\mu}$ (sim)",
            "pion": r"$\sigma_{\mathrm{MCS}}^{\pi}$ (sim)",
            "proton": r"$\sigma_{\mathrm{MCS}}^{p}$ (sim)",
        }

        if mcs_correction_mode in {"both", "with_hc"}:
            # Plot MCS with HC correction.
            for series, name, uncertainty in [
                (mcs_e, "electron", uncertainty_mcs_e),
                (mcs_mu, "muon", uncertainty_mcs_mu),
                (mcs_pi, "pion", uncertainty_mcs_pi),
                (mcs_proton, "proton", uncertainty_mcs_proton),
            ]:
                px, py, uncertainties = valid(momenta, series, uncertainty)
                ax.errorbar(
                    list(px),
                    list(py),
                    yerr=list(uncertainties),
                    color=COLOURS[name],
                    marker="^",
                    linewidth=1.8,
                    markersize=6,
                    linestyle="-.",
                    label=LABELS_WITH_HC[name],
                )

        if mcs_correction_mode in {"both", "no_hc"}:
            # Plot MCS without HC correction.
            for series, name, uncertainty in [
                (mcs_nohc_e, "electron", uncertainty_mcs_nohc_e),
                (mcs_nohc_mu, "muon", uncertainty_mcs_nohc_mu),
                (mcs_nohc_pi, "pion", uncertainty_mcs_nohc_pi),
                (mcs_nohc_proton, "proton", uncertainty_mcs_nohc_proton),
            ]:
                px, py, uncertainties = valid(momenta, series, uncertainty)
                ax.errorbar(
                    list(px),
                    list(py),
                    yerr=list(uncertainties),
                    color=COLOURS[name],
                    marker="x",
                    linewidth=1.5,
                    markersize=6,
                    linestyle="--",
                    label=LABELS_NO_HC[name],
                )

    if plot_annotations:
        # --- run numbers as annotations above each measured point ---
        for r in binned_rows:
            ax.annotate(
                str(r["run"]),
                xy=(r["p_nom"], r["sx_meas"]),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                color="dimgray",
            )

    if plot_sigma_init:
        x_pos = momenta[-1]
        y_bottom = list(py)[-1]
        y_top = sy_meas[-1]
        ax.annotate(
            "",
            xy=(x_pos, y_bottom),
            xytext=(x_pos, y_top),
            arrowprops=dict(arrowstyle="<->", color="black", lw=1.5),
        )
        ax.text(
            x_pos,
            (y_bottom + y_top) / 2,
            r"$\propto\sigma_{init}$",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    plot_title: str
    plot_filename: str
    if not plot_estimations:
        plot_title = "Measured beam profiles vs MCS simulation"
        plot_filename = "measured_beam_profile.png"
    
    elif plot_MCS_sim:
        plot_title = "Measured beam profiles (corrected) vs. MCS simulation"
        plot_filename = "measured_vs_MCS_simulation.png"

    else:
        plot_title = "Measured beam profile at different momenta"
        plot_filename = "measured_beam_profile.png"

    ax.set_xlabel("Nominal beam momentum (MeV/c)", fontsize=24)
    ax.set_ylabel(r"$\sigma$ (cm)", fontsize=24)
    ax.set_title(
        plot_title,
        fontsize=28,
        pad=10,
    )
    if y_limits is not None:
        ax.set_ylim(y_limits)

    if plot_bias_line:
        ax.axvline(x=540, color="gray", linestyle=":", linewidth=1.5)
        # Keep x in data coordinates while y follows the current axes height.
        ax.text(
            540,
            0.98,
            "  trigger bias below",
            transform=ax.get_xaxis_transform(),
            color="gray",
            fontsize=20,
            va="top",
            ha="left",
            rotation=90,
        )

    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.grid(which="minor", linestyle=":", linewidth=0.3, alpha=0.4)
    ax.xaxis.label.set_fontsize(20)
    ax.yaxis.label.set_fontsize(20)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)

    ax.legend(loc="upper right", fontsize=20, framealpha=0.8)
    fig.tight_layout()

    if filename_suffix:
        stem = Path(plot_filename).stem
        suffix = Path(plot_filename).suffix
        plot_filename = f"{stem}{filename_suffix}{suffix}"

    output_base = Path(output_path) / plot_filename
    saved_paths = []
    for ext in (".png", ".pdf"):
        fig_output_path = output_base.with_suffix(ext)
        fig.savefig(fig_output_path, dpi=150)
        saved_paths.append(str(fig_output_path))
    print(f"Plot saved to: {', '.join(saved_paths)}")
    plt.close(fig)


def load_proton_sigma_y_from_summary(csv_path: str) -> dict[int, float]:
    """
    Load measured proton sigma_y values from gaussian summary CSV.

    The input file stores sigma values in mm. Returned values are converted to cm.
    """
    proton_sigma_y_cm_by_run: dict[int, float] = {}

    with open(csv_path, newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"run_number", "particle", "sigma_y"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(
                "CSV must contain columns: run_number, particle, sigma_y"
            )

        for row in reader:
            if row.get("particle", "").strip().lower() != "protons":
                continue

            run_raw = row.get("run_number")
            sigma_y_raw = row.get("sigma_y")
            if run_raw is None or sigma_y_raw is None:
                continue

            try:
                run_number = int(run_raw)
                sigma_y_cm = 0.1 * float(sigma_y_raw)
            except ValueError:
                continue

            proton_sigma_y_cm_by_run[run_number] = sigma_y_cm

    return proton_sigma_y_cm_by_run


def load_proton_measurements_from_summary(csv_path: str) -> dict[int, dict]:
    """
    Load measured proton sigma_y and source parquet paths from summary CSV.

    The input file stores sigma values in mm. Returned sigma values are in cm.
    """
    proton_measurements_by_run: dict[int, dict] = {}

    with open(csv_path, newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"run_number", "particle", "sigma_y", "path"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(
                "CSV must contain columns: run_number, particle, sigma_y, path"
            )

        for row in reader:
            if row.get("particle", "").strip().lower() != "protons":
                continue

            run_raw = row.get("run_number")
            sigma_y_raw = row.get("sigma_y")
            parquet_path = row.get("path")
            if run_raw is None or sigma_y_raw is None or parquet_path is None:
                continue

            try:
                run_number = int(run_raw)
                sigma_y_cm = 0.1 * float(sigma_y_raw)
            except ValueError:
                continue

            proton_measurements_by_run[run_number] = {
                "sigma_y_cm": sigma_y_cm,
                "parquet_path": parquet_path,
            }

    return proton_measurements_by_run


def get_parquet_row_count(parquet_path: str) -> Optional[int]:
    """
    Return number of rows in a parquet file using lightweight metadata when possible.
    """
    try:
        import pyarrow.parquet as pq  # type: ignore
        print(f"Reading parquet metadata for {parquet_path} to get row count...")
        return int(pq.ParquetFile(parquet_path).metadata.num_rows)
    except Exception:
        print(f"WARNING: could not read parquet metadata for {parquet_path}; falling back to pandas.")
        pass

    try:
        import pandas as pd  # type: ignore

        return int(len(pd.read_parquet(parquet_path)))
    except Exception:
        print(f"WARNING: could not read parquet file {parquet_path} with pandas; returning None for row count.")
        return None


def plot_proton_sigma_y_measured_vs_mcs(
    rows: list[dict],
    proton_summary_csv_path: str,
    output_path: str,
    min_proton_entries: int = 0,
    filename: str = "proton_sigma_y_measured_vs_mcs.png",
):
    """
    Plot measured proton sigma_y from gaussian summary against proton MCS simulation.

    Uses a band scan in the momentum range [700, 2000] MeV/c with one
    representative run per 50 MeV/c band.
    """
    MOMENTUM_MIN = 700.0
    MOMENTUM_MAX = 2000.0
    BAND_WIDTH = 50.0

    proton_measurements_by_run = load_proton_measurements_from_summary(
        proton_summary_csv_path
    )
    row_count_cache: dict[str, Optional[int]] = {}

    comparison_rows = []
    for row in rows:
        run_number = row.get("run")
        mcs_proton = row.get("mcs_proton")
        if run_number is None or mcs_proton is None:
            continue
        if run_number not in proton_measurements_by_run:
            continue

        measurement = proton_measurements_by_run[run_number]
        parquet_path = measurement["parquet_path"]

        n_entries: Optional[int] = None
        if min_proton_entries > 0:
            if parquet_path not in row_count_cache:
                row_count_cache[parquet_path] = get_parquet_row_count(parquet_path)
            n_entries = row_count_cache[parquet_path]

            if n_entries is None:
                print(
                    f"WARNING: could not read proton statistics from {parquet_path}; keeping run {run_number}."
                )
            elif n_entries < min_proton_entries:
                continue

        comparison_rows.append(
            {
                "run": run_number,
                "p_nom": row["p_nom"],
                "sigma_y_measured": measurement["sigma_y_cm"],
                "sigma_y_mcs_proton": mcs_proton,
                "sigma_y_mcs_proton_err": row.get("uncertainty_mcs_proton"),
                "n_entries": n_entries,
            }
        )

    if not comparison_rows:
        print(
            "No matched proton runs were found between gaussian summary CSV and MCS rows; skipping proton comparison plot."
        )
        return

    # Keep only target momentum window for band scan.
    comparison_rows = [
        item
        for item in comparison_rows
        if MOMENTUM_MIN <= item["p_nom"] <= MOMENTUM_MAX
    ]

    if not comparison_rows:
        print(
            "No matched proton runs in 700-2000 MeV/c for band scan after statistics filtering; skipping proton comparison plot."
        )
        return

    # Pick one representative point per 50 MeV/c band.
    # Deterministic choice: lowest momentum, then lowest run number.
    comparison_rows.sort(key=lambda item: (item["p_nom"], item["run"]))
    seen_bands: dict[int, dict] = {}
    for item in comparison_rows:
        band_idx = int((item["p_nom"] - MOMENTUM_MIN) // BAND_WIDTH)
        if band_idx not in seen_bands:
            seen_bands[band_idx] = item

    comparison_rows = sorted(seen_bands.values(), key=lambda item: item["p_nom"])

    momenta = [item["p_nom"] for item in comparison_rows]
    measured_sigma_y = [item["sigma_y_measured"] for item in comparison_rows]
    mcs_sigma_y = [item["sigma_y_mcs_proton"] for item in comparison_rows]
    mcs_sigma_y_err = [item["sigma_y_mcs_proton_err"] for item in comparison_rows]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        momenta,
        measured_sigma_y,
        color="black",
        marker="o",
        linewidth=1.8,
        markersize=6,
        label=r"Measured proton $\sigma_y$",
    )
    ax.errorbar(
        momenta,
        mcs_sigma_y,
        yerr=mcs_sigma_y_err,
        color="#3cb44b",
        marker="^",
        linewidth=1.8,
        markersize=6,
        linestyle="-.",
        label=r"MCS proton $\sigma_y$ (simulated)",
    )

    for item in comparison_rows:
        ax.annotate(
            str(item["run"]),
            xy=(item["p_nom"], item["sigma_y_measured"]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            color="dimgray",
        )

    ax.set_xlabel("Nominal beam momentum (MeV/c)", fontsize=12)
    ax.set_ylabel(r"$\sigma_y$ (cm)", fontsize=12)
    ax.set_title("Proton measured $\sigma_y$ vs proton MCS simulation", fontsize=12, pad=10)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.grid(which="minor", linestyle=":", linewidth=0.3, alpha=0.4)
    ax.legend(loc="best", fontsize=10, framealpha=0.9)

    fig.tight_layout()

    fig_output_path = Path(output_path) / filename
    fig.savefig(fig_output_path, dpi=150)
    print(
        f"Proton measured-vs-MCS comparison plot saved to: {fig_output_path} (band-scan points: {len(comparison_rows)}, min entries: {min_proton_entries})"
    )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Estimate initial beam profile by fitting sigma_y with electron MCS"
    )
    parser.add_argument("--config", required=True, help="Path to config.json")
    parser.add_argument(
        "--output",
        default="initial_beam_profile_fit.png",
        help="Output path for the plot (default: initial_beam_profile_fit.png)",
    )
    parser.add_argument(
        "--min-proton-entries",
        type=int,
        default=0,
        help="Minimum proton entries in parquet to keep a run in proton comparison plot (default: 0 = no filter)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Valid momentum ranges: <= 450 MeV/c and >= 750 MeV/c
    VALID_MOMENTUM_RANGES = [(0, 10000)]

    # Filter and collect data
    valid_runs = []

    for cf_run in config:
        # skip entries with no run number
        if not cf_run.get("run_number"):
            continue
        rn = int(cf_run["run_number"])

        # charged hadron runs only
        if "hadron" not in cf_run.get("beam_config", "").lower():
            continue

        # exclude tagged gamma runs
        beam_config_lower = cf_run.get("beam_config", "").lower()
        if "tagged" in beam_config_lower and "gamma" in beam_config_lower:
            continue

        if "T5_beam_sigma_x" not in cf_run or "T5_beam_sigma_y" not in cf_run:
            print(f"NOTE: run {rn} has no T5 sigma data, skipping.")
            continue
        if "lemb" in cf_run.get("trigger_config").lower():
            continue

        # beam_momentum is now a string in the new config
        if not cf_run.get("beam_momentum"):
            print(f"NOTE: run {rn} has no beam momentum, skipping.")
            continue
        p = float(cf_run["beam_momentum"])

        # Filter by valid momentum ranges
        in_valid_range = any(lo <= p <= hi for lo, hi in VALID_MOMENTUM_RANGES)
        if not in_valid_range:
            continue

        valid_runs.append((rn, p, cf_run))

    print(
        f"Found {len(valid_runs)} runs in valid momentum ranges (≤450 or ≥750 MeV/c)\n"
    )

    T5_config_path = "../../configs/T5_reconstruction_values.json"
    SiPM_resolution_cm, SiPM_resolution_cm_uncertainty = get_T5_resolution_cm(
        T5_config_path
    )
    print(
        f"SiPM resolution: {SiPM_resolution_cm:.4f} ± {SiPM_resolution_cm_uncertainty:.4f} cm"
    )

    rows = []
    # Lists for only scintillator corrected values
    sigma_y_scint_corrected = []  # Sigma_y corrected for scintillator only
    sigma_y_init_from_scint = []  # Initial sigma_y using scintillator dimensions corrected data
    sigma_y_init_from_scint_statistical_errors = []  # Errors on scintillator-corrected initial sigma
    sigma_y_init_from_scint_systematic_errors = []  # Systematic errors on scintillator-corrected initial sigma
    sigma_y_init_from_scint_total_errors = []  # Total errors on initial sigma_y (combined statistical uncertainty from 2D Gaussian fit with systematic uncertainty from MCS)

    # Lists for divergence-corrected values
    sigma_y_no_divergence = []  # Sigma_y after divergence correction
    sigma_y_init_from_noDiv = []  # Initial sigma_y from divergence-corrected (with div corr)
    sigma_y_init_from_noDiv_statistical_errors = []  # Errors on divergence-corrected initial sigma
    sigma_y_init_from_noDiv_systematic_errors = []  # Systematic errors on divergence-corrected initial sigma
    sigma_y_init_from_noDiv_total_errors = []  # Total errors on initial sigma_y (after divergence correction, combined)

    momenta_data = []
    run_numbers_data = []
    divergence_contrib_data = []  # Track divergence contribution for plotting

    invalid_runs = [] # Track runs where correction of sigma_x failed

    for rn, momentum, cf_run in valid_runs:
        # Skip if measurement values are None or invalid
        if (
            cf_run.get("T5_beam_sigma_y") is None
            or cf_run.get("T5_beam_sigma_y_error") is None
        ):
            print(f"NOTE: run {rn} has None values in T5 sigma data, skipping.")
            continue

        try:
            sy_meas_cm = float(cf_run["T5_beam_sigma_y"]) * 0.1
            sy_meas_err_cm = float(cf_run["T5_beam_sigma_y_error"]) * 0.1
            sx_meas_cm = float(cf_run["T5_beam_sigma_x"]) * 0.1
            sx_meas_err_cm = float(cf_run["T5_beam_sigma_x_error"]) * 0.1
        except (TypeError, ValueError) as e:
            print(f"NOTE: run {rn} has invalid T5 sigma data ({e}), skipping.")
            continue

        # Correct measured sigma_x by SiPM resolution
        sx_corrected_sq = sx_meas_cm**2 - SiPM_resolution_cm**2
        sx_corrected = math.sqrt(sx_corrected_sq) if sx_corrected_sq >= 0 else None
        if sx_corrected is not None and sx_corrected > 0:
            sx_corrected_err = math.sqrt(
                (sx_meas_cm / sx_corrected * sx_meas_err_cm) ** 2
                + (SiPM_resolution_cm / sx_corrected * SiPM_resolution_cm_uncertainty) ** 2
            )
        else:
            sx_corrected_err = None
            print(f"ERROR: run {rn} has sx_corrected imaginary, skipping...")
            invalid_runs.append(rn)

        # Correct measured sigma_y by scintillator block height
        scint_size_y_cm = 1.625
        sy_corrected = math.sqrt(sy_meas_cm**2 - scint_size_y_cm**2 / 12)
        sy_corrected_err = (
            sy_meas_err_cm * sy_meas_cm / sy_corrected if sy_corrected > 0 else None
        )



        # Subtract divergence contribution
        divergence_contrib_cm = calculate_divergence_contribution(
            BEAM_DIVERGENCE_MRAD, DISTANCE_WINDOW_TO_T5_M
        )
        sy_no_divergence_sq = sy_corrected**2 - divergence_contrib_cm**2
        sy_no_divergence = (
            math.sqrt(sy_no_divergence_sq) if sy_no_divergence_sq >= 0 else None
        )

        if sy_no_divergence is None:
            print(
                f"Run {rn} (p = {momentum:6.1f} MeV/c) → WARNING: divergence contribution "
                f"({divergence_contrib_cm:.4f} cm) exceeds corrected sigma ({sy_corrected:.4f} cm), skipping."
            )
            continue

        print(
            f"Run {rn} (p = {momentum:6.1f} MeV/c) → σy_meas = {sy_meas_cm:.4f} cm, "
            f"σy_corr = {sy_corrected:.4f} cm, σy_no_div = {sy_no_divergence:.4f} cm ...",
            flush=True,
        )

        row = {
            "run": rn,
            "p_nom": momentum,
            "sx_meas": sx_meas_cm,
            "sx_meas_err": sx_meas_err_cm,
            "sy_meas": sy_meas_cm,
            "sy_meas_err": sy_meas_err_cm,
            "sx_corrected": sx_corrected,
            "sx_corrected_err": sx_corrected_err,
            "sy_corrected": sy_corrected,
            "sy_corrected_err": sy_corrected_err,
            "mcs_e": None,
            "mcs_mu": None,
            "mcs_pi": None,
            "mcs_proton": None,
            "uncertainty_mcs_e": None,
            "uncertainty_mcs_mu": None,
            "uncertainty_mcs_pi": None,
            "uncertainty_mcs_proton": None,
            "mcs_nohc_e": None,
            "mcs_nohc_mu": None,
            "mcs_nohc_pi": None,
            "mcs_nohc_proton": None,
            "uncertainty_mcs_nohc_e": None,
            "uncertainty_mcs_nohc_mu": None,
            "uncertainty_mcs_nohc_pi": None,
            "uncertainty_mcs_nohc_proton": None,
        }

        # Calculate initial sigma_y using electron MCS estimation only
        try:
            sigma_mcs_e = simulate_mcs(rn, "electron", momentum, HC_veto=True)
            sigma_mcs_mu = simulate_mcs(rn, "muon", momentum, HC_veto=True)
            sigma_mcs_pi = simulate_mcs(rn, "pion", momentum, HC_veto=True)
            sigma_mcs_proton = simulate_mcs(rn, "proton", momentum, HC_veto=True)
            sigma_mcs_nohc_e = simulate_mcs(rn, "electron", momentum, HC_veto=False)
            sigma_mcs_nohc_mu = simulate_mcs(rn, "muon", momentum, HC_veto=False)
            sigma_mcs_nohc_pi = simulate_mcs(rn, "pion", momentum, HC_veto=False)
            sigma_mcs_nohc_proton = simulate_mcs(rn, "proton", momentum, HC_veto=False)
            uncertainty_mcs_e = calculate_highland_systematic_uncertainty(sigma_mcs_e)
            uncertainty_mcs_mu = calculate_highland_systematic_uncertainty(sigma_mcs_mu)
            uncertainty_mcs_pi = calculate_highland_systematic_uncertainty(sigma_mcs_pi)
            uncertainty_mcs_proton = calculate_highland_systematic_uncertainty(
                sigma_mcs_proton
            )
            uncertainty_mcs_nohc_e = calculate_highland_systematic_uncertainty(sigma_mcs_nohc_e)
            uncertainty_mcs_nohc_mu = calculate_highland_systematic_uncertainty(sigma_mcs_nohc_mu)
            uncertainty_mcs_nohc_pi = calculate_highland_systematic_uncertainty(sigma_mcs_nohc_pi)
            uncertainty_mcs_nohc_proton = calculate_highland_systematic_uncertainty(
                sigma_mcs_nohc_proton
            )

            row["mcs_e"] = sigma_mcs_e
            row["mcs_mu"] = sigma_mcs_mu
            row["mcs_pi"] = sigma_mcs_pi
            row["mcs_proton"] = sigma_mcs_proton
            row["uncertainty_mcs_e"] = uncertainty_mcs_e
            row["uncertainty_mcs_mu"] = uncertainty_mcs_mu
            row["uncertainty_mcs_pi"] = uncertainty_mcs_pi
            row["uncertainty_mcs_proton"] = uncertainty_mcs_proton
            row["mcs_nohc_e"] = sigma_mcs_nohc_e
            row["mcs_nohc_mu"] = sigma_mcs_nohc_mu
            row["mcs_nohc_pi"] = sigma_mcs_nohc_pi
            row["mcs_nohc_proton"] = sigma_mcs_nohc_proton
            row["uncertainty_mcs_nohc_e"] = uncertainty_mcs_nohc_e
            row["uncertainty_mcs_nohc_mu"] = uncertainty_mcs_nohc_mu
            row["uncertainty_mcs_nohc_pi"] = uncertainty_mcs_nohc_pi
            row["uncertainty_mcs_nohc_proton"] = uncertainty_mcs_nohc_proton

            # Calculate initial sigma from BOTH scintillator-corrected and divergence-corrected values
            # beam spot size corrected for scintillator height
            sigma_y_init_scint = initial_sigma(sy_corrected, sigma_mcs_e)
            # beam spot size corrected for divergence contribution
            sigma_y_init_nodiv = initial_sigma(sy_no_divergence, sigma_mcs_e)

            if sigma_y_init_nodiv is not None and sigma_y_init_nodiv >= 0:
                row["sigma_y_init"] = sigma_y_init_scint
                row["mcs_e"] = sigma_mcs_e

                # Error propagation: σ_init = √(σ_corr² - σ_MCS²)
                # dσ_init/dσ_corr = σ_corr / σ_init
                sigma_y_init_statistical_err_scint = None
                sigma_y_init_systematic_err_scint = None
                sigma_y_init_combined_err_scint = None

                sigma_y_init_statistical_err_nodiv = None
                sigma_y_init_systematic_err_nodiv = None
                sigma_y_init_combined_err_nodiv = None

                if sy_corrected_err is not None and sigma_y_init_scint > 0:
                    sigma_y_init_statistical_err_scint = (
                        sy_corrected_err * sy_corrected
                    ) / sigma_y_init_scint
                    sigma_y_init_systematic_err_scint = (
                        sy_corrected_err * sy_corrected
                    ) / sigma_y_init_scint
                    sigma_y_init_combined_err_scint = combine_uncertainties(
                        sigma_y_init_scint,
                        sy_corrected,
                        sigma_mcs_e,
                        sy_corrected_err,
                        uncertainty_mcs_e,
                    )
                    row["sigma_y_init_err"] = sigma_y_init_combined_err_scint

                if sy_corrected_err is not None and sigma_y_init_nodiv > 0:
                    sigma_y_init_statistical_err_nodiv = (
                        sy_corrected_err * sy_corrected
                    ) / sigma_y_init_nodiv
                    sigma_y_init_systematic_err_nodiv = (
                        sy_corrected_err * sy_corrected
                    ) / sigma_y_init_nodiv
                    sigma_y_init_combined_err_nodiv = combine_uncertainties(
                        sigma_y_init_nodiv,
                        sy_no_divergence,
                        sigma_mcs_e,
                        sy_corrected_err,
                        uncertainty_mcs_e,
                    )
                    row["sigma_y_init_err"] = sigma_y_init_combined_err_nodiv
                else:
                    row["sigma_y_init_err"] = None

                sigma_y_scint_corrected.append(sy_corrected)
                sigma_y_init_from_scint.append(sigma_y_init_scint)
                sigma_y_init_from_scint_statistical_errors.append(
                    sigma_y_init_statistical_err_scint
                )
                sigma_y_init_from_scint_systematic_errors.append(
                    sigma_y_init_systematic_err_scint
                )
                sigma_y_init_from_scint_total_errors.append(
                    sigma_y_init_combined_err_scint
                )

                sigma_y_no_divergence.append(sy_no_divergence)
                sigma_y_init_from_noDiv.append(sigma_y_init_nodiv)
                sigma_y_init_from_noDiv_statistical_errors.append(
                    sigma_y_init_statistical_err_nodiv
                )
                sigma_y_init_from_noDiv_systematic_errors.append(
                    sigma_y_init_systematic_err_nodiv
                )
                sigma_y_init_from_noDiv_total_errors.append(
                    sigma_y_init_combined_err_nodiv
                )

                momenta_data.append(momentum)
                run_numbers_data.append(rn)
                divergence_contrib_data.append(divergence_contrib_cm)

                print(
                    f"  MCS_e = {sigma_mcs_e:.4f} cm → σy_init(scint)={sigma_y_init_scint:.4f} cm, σy_init(no_div)={sigma_y_init_nodiv:.4f} cm"
                )
            else:
                print(
                    f"  MCS_e calculation resulted in negative sigma_y_init, skipping."
                )
                row["sigma_y_init"] = None

                row["sigma_y_init_err"] = None

        except Exception as exc:
            print(f"  WARNING: electron simulation failed for run {rn}: {exc}")
            row["sigma_y_init"] = None
            row["mcs_e"] = None
            row["sigma_y_init_err"] = None

        rows.append(row)

    print()
    if invalid_runs:
        print(f"Runs with invalid sx_corrected: {invalid_runs}")

    plot_MCS_data_comparison(
        rows,
        output_path=args.output,
        plot_estimations=True,
        plot_MCS_sim=True,
        plot_bias_line=True,
        plot_sigma_init=False,
        sipm_resolution_cm=SiPM_resolution_cm,
    )
    plot_MCS_data_comparison(
        rows,
        output_path=args.output,
        plot_estimations=True,
        plot_MCS_sim=True,
        plot_bias_line=True,
        plot_sigma_init=False,
        sipm_resolution_cm=SiPM_resolution_cm,
        y_limits=None,
        filename_suffix="_unzoomed",
    )
    plot_MCS_data_comparison(
        rows,
        output_path=args.output,
        plot_estimations=False  ,
        plot_MCS_sim=True,
        plot_bias_line=True,
        plot_sigma_init=False,
        sipm_resolution_cm=SiPM_resolution_cm,
    )
    plot_MCS_data_comparison(
        rows,
        output_path=args.output,
        plot_estimations=False,
        plot_MCS_sim=True,
        plot_bias_line=True,
        plot_sigma_init=False,
        sipm_resolution_cm=SiPM_resolution_cm,
        y_limits=None,
        filename_suffix="_unzoomed",
    )

    plot_MCS_data_comparison(
            rows,
            output_path=args.output,
            plot_estimations=False,
            plot_MCS_sim=True,
            plot_bias_line=True,
            plot_sigma_init=False,
            sipm_resolution_cm=SiPM_resolution_cm,
            mcs_correction_mode="with_hc",
            plot_y_corrected_estimation=True,
            filename_suffix="_onlyhc_ycorr_only",
        )

    plot_MCS_data_comparison(
        rows,
        output_path=args.output,
        plot_estimations=False,
        plot_MCS_sim=True,
        plot_bias_line=True,
        plot_sigma_init=False,
        sipm_resolution_cm=SiPM_resolution_cm,
        mcs_correction_mode="no_hc",
        plot_y_corrected_estimation=True,
        filename_suffix="_nohc_ycorr_only",
    )
    plot_MCS_data_comparison(
        rows,
        output_path=args.output,
        plot_estimations=False,
        plot_MCS_sim=True,
        plot_bias_line=True,
        plot_sigma_init=False,
        sipm_resolution_cm=SiPM_resolution_cm,
        y_limits=None,
        mcs_correction_mode="no_hc",
        plot_y_corrected_estimation=True,
        filename_suffix="_nohc_ycorr_only_unzoomed",
    )

    proton_summary_csv_path = (
        "../../analysis_tools/analysis_examples/run_gaussian_summary.csv"
    )
    plot_proton_sigma_y_measured_vs_mcs(
        rows=rows,
        proton_summary_csv_path=proton_summary_csv_path,
        output_path=args.output,
        min_proton_entries=args.min_proton_entries,
    )



    # Fit a constant to the initial sigma_y values
    if len(sigma_y_scint_corrected) > 0:
        print(
            f"Fitting constant to {len(sigma_y_scint_corrected)} initial sigma_y values..."
        )

        # === FIT 1: All energies ===
        print(f"\nFitting to ALL valid energy runs:\n")

        # fit for scintillator-corrected initial sigma_y values
        fitted_scint_only_all, uncertainty_scint_only_all = fit_constant_least_squares(
            momenta_data, sigma_y_init_from_scint, sigma_y_init_from_scint_total_errors
        )
        # calculate chi squared for scintillator-corrected initial sigma_y values
        chi_squared_scint_all = calculate_chi_squared(
            sigma_y_init_from_scint,
            [fitted_scint_only_all] * len(sigma_y_init_from_scint),
            sigma_y_init_from_scint_total_errors,
        )

        # fit for divergence-corrected initial sigma_y values
        fitted_no_divergence_all, uncertainty_no_divergence_all = (
            fit_constant_least_squares(
                momenta_data,
                sigma_y_init_from_noDiv,
                sigma_y_init_from_noDiv_total_errors,
            )
        )
        # calculate chi squared for divergence-corrected initial sigma_y values
        chi_squared_no_div_all = calculate_chi_squared(
            sigma_y_init_from_noDiv,
            [fitted_no_divergence_all] * len(sigma_y_init_from_noDiv),
            sigma_y_init_from_noDiv_total_errors,
        )

        # === FIT 2: High energy only (> 750 MeV/c) ===
        high_energy_mask = [p > 750 for p in momenta_data]

        run_numbers_high_e = [
            r for r, mask in zip(run_numbers_data, high_energy_mask) if mask
        ]
        momenta_high_e = [p for p, mask in zip(momenta_data, high_energy_mask) if mask]

        # beam widths corrected by scintillator height
        sigma_y_init_from_scint_high_e = [
            s for s, mask in zip(sigma_y_init_from_scint, high_energy_mask) if mask
        ]
        sigma_y_init_from_scint_total_errors_high_e = [
            e
            for e, mask in zip(sigma_y_init_from_scint_total_errors, high_energy_mask)
            if mask
        ]

        # beam widths corrected by beam divergence
        sigma_y_init_from_noDiv_high_e = [
            s for s, mask in zip(sigma_y_init_from_noDiv, high_energy_mask) if mask
        ]
        sigma_y_init_from_noDiv_total_errors_high_e = [
            e
            for e, mask in zip(sigma_y_init_from_noDiv_total_errors, high_energy_mask)
            if mask
        ]

        print(
            f"Fitting to HIGH-ENERGY runs only (> 750 MeV/c): {len(sigma_y_init_from_scint_high_e)} runs\n"
        )

        # fit and chi squared calculation for scintillator-corrected initial sigma_y values at high energies
        fitted_constant_scint_only_high_e, uncertainty_scint_only_high_e = (
            fit_constant_least_squares(
                momenta_high_e,
                sigma_y_init_from_scint_high_e,
                sigma_y_init_from_scint_total_errors_high_e,
            )
        )
        chi_squared_scint_only_high_e = calculate_chi_squared(
            sigma_y_init_from_scint_high_e,
            [fitted_constant_scint_only_high_e] * len(sigma_y_init_from_scint_high_e),
            sigma_y_init_from_scint_total_errors_high_e,
        )

        # fit and chi squared calculation for divergence-corrected initial sigma_y values at high energies
        fitted_no_divergence_high_e, uncertainty_no_divergence_high_e = (
            fit_constant_least_squares(
                momenta_high_e,
                sigma_y_init_from_noDiv_high_e,
                sigma_y_init_from_noDiv_total_errors_high_e,
            )
        )
        chi_squared_no_div_high_e = calculate_chi_squared(
            sigma_y_init_from_noDiv_high_e,
            [fitted_no_divergence_high_e] * len(sigma_y_init_from_noDiv_high_e),
            sigma_y_init_from_noDiv_total_errors_high_e,
        )

        print(f"\n{'=' * 70}")
        print(f"INITIAL BEAM PROFILE (σy at beam window exit)")
        print(f"{'=' * 70}")
        print(f"\n--- FIT RESULTS (ALL VALID ENERGIES) ---")
        print(
            f"Fitted constant (divergence corrected):  {fitted_no_divergence_all:.4f} ± {uncertainty_no_divergence_all:.4f} cm"
        )
        print(
            f"Fitted constant (scintillator only):     {fitted_scint_only_all:.4f} ± {uncertainty_scint_only_all:.4f} cm"
        )
        print(
            f"Difference:                              {fitted_scint_only_all - fitted_no_divergence_all:.4f} cm"
        )
        print(
            f"Number of data points:                   {len(sigma_y_init_from_scint)}"
        )

        print(f"\n--- FIT RESULTS (HIGH ENERGY > 750 MeV/c) ---")
        print(
            f"Fitted constant (divergence corrected):  {fitted_no_divergence_high_e:.4f} ± {uncertainty_no_divergence_high_e:.4f} cm"
        )
        print(
            f"Fitted constant (scintillator only):     {fitted_constant_scint_only_high_e:.4f} ± {uncertainty_scint_only_high_e:.4f} cm"
        )
        print(
            f"Difference:                              {fitted_constant_scint_only_high_e - fitted_no_divergence_high_e:.4f} cm"
        )
        print(
            f"Number of data points:                   {len(sigma_y_init_from_scint_high_e)}"
        )
        print(f"{'=' * 70}\n")

        # Print individual measurements
        print("Individual measurements:")
        print(
            f"{'Run':<6} {'Momentum':<12} {'σy_scint':<12} {'σy_no_div':<12} {'MCS_e':<12} {'σy_init(scint)':<15} {'σy_init(no_div)':<15}"
        )
        print(
            f"{'':6} {'[MeV/c]':<12} {'[cm]':<12} {'[cm]':<12} {'[cm]':<12} {'[cm]':<15} {'[cm]':<15}"
        )
        print("-" * 105)
        for i, (rn, p, s_scint, s_nodiv, s_init_s, s_init_n) in enumerate(
            zip(
                run_numbers_data,
                momenta_data,
                sigma_y_scint_corrected,
                sigma_y_no_divergence,
                sigma_y_init_from_scint,
                sigma_y_init_from_noDiv,
            )
        ):
            print(
                f"{rn:<6} {p:<12.1f} {s_scint:<12.4f} {s_nodiv:<12.4f} {rows[i]['mcs_e']:<12.4f} {s_init_s:<15.4f} {s_init_n:<15.4f}"
            )

        # Helper function to create comparison plots
        def create_comparison_plot(
            momenta,
            run_numbers,
            sigma_scint,
            sigma_nodiv,
            sigma_errors_scint,
            sigma_errors_nodiv,
            fitted_scint,
            unc_scint,
            fitted_nodiv,
            unc_nodiv,
            title_suffix,
            filename_suffix,
        ):
            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot initial sigma calculated from scintillator-corrected data
            ax.errorbar(
                momenta,
                sigma_scint,
                yerr=sigma_errors_scint,
                fmt="o",
                color="#4363d8",
                markersize=8,
                capsize=5,
                capthick=2,
                label="σy,init (scintillator corrected)",
                linewidth=1.5,
            )

            # Fit line for scintillator-corrected initial sigma
            momentum_range = [min(momenta) - 50, max(momenta) + 50]
            ax.axhline(
                y=fitted_scint,
                color="#4363d8",
                linestyle="--",
                linewidth=2,
                label=f"Fit (scint. only): {fitted_scint:.4f} ± {unc_scint:.4f} cm",
            )

            # Plot initial sigma calculated from divergence-corrected data
            ax.errorbar(
                momenta,
                sigma_nodiv,
                yerr=sigma_errors_nodiv,
                fmt="s",
                color="#e6194b",
                markersize=8,
                capsize=5,
                capthick=2,
                label="σy,init (divergence corrected)",
                linewidth=1.5,
            )

            # Fit line for divergence-corrected initial sigma
            ax.axhline(
                y=fitted_nodiv,
                color="#e6194b",
                linestyle=":",
                linewidth=2,
                label=f"Fit (div. corrected): {fitted_nodiv:.4f} ± {unc_nodiv:.4f} cm",
            )

            # Add shading for uncertainty bands
            ax.fill_between(
                momentum_range,
                fitted_scint - unc_scint,
                fitted_scint + unc_scint,
                alpha=0.1,
                color="#4363d8",
            )
            ax.fill_between(
                momentum_range,
                fitted_nodiv - unc_nodiv,
                fitted_nodiv + unc_nodiv,
                alpha=0.1,
                color="#e6194b",
            )

            # Add run numbers as annotations
            for rn, p, s_init in zip(run_numbers, momenta, sigma_scint):
                ax.annotate(
                    str(rn),
                    xy=(p, s_init),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    color="dimgray",
                )

            ax.set_xlabel("Nominal beam momentum (MeV/c)", fontsize=12)
            ax.set_ylabel(r"Initial $\sigma_y$ (cm)", fontsize=12)
            ax.set_title(
                f"Divergence Correction Comparison {title_suffix}",
                fontsize=12,
                pad=10,
            )
            ax.set_xlim(momentum_range)
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.6)
            ax.grid(which="minor", linestyle=":", linewidth=0.3, alpha=0.4)
            ax.legend(loc="best", fontsize=10, framealpha=0.9)

            fig.tight_layout()
            return fig, filename_suffix

        # uncertainty_high_e = calculate_fit_uncertainty(sigma_y_init_from_noDiv, fitted_constant_high_e)
        # Helper function to create final plots
        def create_final_plot(
            momenta,
            run_numbers,
            sigma_final,
            sigma_errors,
            fitted_value,
            uncertainty,
            chi_squared,
            NDF,
            title_suffix,
            filename_suffix,
        ):
            fig, ax = plt.subplots(figsize=(10, 6))
            # ax.set_ylim(0, 5)
            # Plot individual points with error bars
            if sigma_errors and len(sigma_errors) == len(sigma_final):
                ax.errorbar(
                    momenta,
                    sigma_final,
                    yerr=sigma_errors,
                    fmt="o",
                    fillstyle="none",
                    color="#e6194b",
                    markersize=6,
                    alpha=0.7,
                    capsize=2,
                    capthick=2,
                    label="Initial $\sigma_y$",
                    linewidth=0.8,
                )
            else:
                ax.plot(
                    momenta,
                    sigma_final,
                    "o",
                    color="#e6194b",
                    markersize=8,
                    label="Initial $\sigma_y$",
                    linewidth=1.5,
                )

            # Plot fitted constant
            momentum_range = [min(momenta) - 50, max(momenta) + 50]
            ax.axhline(
                y=fitted_value,
                color="black",
                linestyle="-",
                linewidth=2,
                label=f"Fit: {fitted_value:.4f} ± {uncertainty:.4f} cm",
            )

            # Add shading for uncertainty band
            ax.fill_between(
                momentum_range,
                fitted_value - uncertainty,
                fitted_value + uncertainty,
                alpha=0.2,
                color="gray",
                label="Fit uncertainty",
            )

            # Add run numbers as annotations
            # for rn, p, s in zip(run_numbers, momenta, sigma_final):
            #     ax.annotate(
            #         str(rn),
            #         xy=(p, s),
            #         xytext=(0, 8),
            #         textcoords="offset points",
            #         ha="center",
            #         fontsize=8,
            #         color="dimgray",
            #     )
            #
            ax.set_xlabel("Nominal beam momentum (MeV/c)", fontsize=12)
            ax.set_ylabel(r"Initial $\sigma_y$ (cm)", fontsize=12)
            ax.set_title(
                f"Initial Beam Profile Estimation {title_suffix}",
                fontsize=12,
                pad=10,
            )
            ax.text(
                0.76,
                0.83,
                f"χ² = {chi_squared:.2f}, NDF = {NDF}",
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
            )
            ax.text(
                0.76,
                0.78,
                f"χ²/NDF = {chi_squared / NDF:.2f}",
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
            )
            ax.set_xlim(momentum_range)
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.6)
            ax.grid(which="minor", linestyle=":", linewidth=0.3, alpha=0.4)
            ax.legend(loc="best", fontsize=10, framealpha=0.9)

            fig.tight_layout()
            return fig, filename_suffix

        plot_name = args.output + "Initial_profile.png"

        # === PLOT 1: Comparison between the final estimates for data corrected only by scintillator size, and for data corrected by both scintillator size and initial beam divergence ===
        fig, suffix = create_comparison_plot(
            momenta_high_e,
            run_numbers_high_e,
            sigma_y_init_from_scint_high_e,
            sigma_y_init_from_noDiv_high_e,
            sigma_y_init_from_scint_total_errors_high_e,
            sigma_y_init_from_noDiv_total_errors_high_e,
            fitted_constant_scint_only_high_e,
            uncertainty_scint_only_high_e,
            fitted_no_divergence_high_e,
            uncertainty_no_divergence_high_e,
            "(high energy > 750 MeV/c)",
            "_comparison_high_energy",
        )
        plot_path = plot_name.replace(".png", suffix + ".png")
        fig.savefig(plot_path, dpi=150)
        print(f"Plot saved to: {plot_path}")
        plt.close(fig)

        # === PLOT 2: Initial profile fit for scintillator correction only (high energy only) ===
        fig, suffix = create_final_plot(
            momenta_high_e,
            run_numbers_high_e,
            sigma_y_init_from_scint_high_e,
            sigma_y_init_from_scint_total_errors_high_e,
            fitted_constant_scint_only_high_e,
            uncertainty_scint_only_high_e,
            chi_squared_scint_only_high_e,
            len(sigma_y_init_from_scint_high_e) - 1,
            "(high energy > 750 MeV/c)",
            "_high_energy_scint_correction_only_fit",
        )
        plot_path = plot_name.replace(".png", suffix + ".png")
        fig.savefig(plot_path, dpi=150)
        print(f"Plot saved to: {plot_path}")
        plt.close(fig)

        # === PLOT 3: Initial profile fit for divergence correction AND scintillator correction (high energy only) ===

        fig, suffix = create_final_plot(
            momenta_high_e,
            run_numbers_high_e,
            sigma_y_init_from_noDiv_high_e,
            sigma_y_init_from_noDiv_total_errors_high_e,
            fitted_no_divergence_high_e,
            uncertainty_no_divergence_high_e,
            chi_squared_no_div_high_e,
            len(sigma_y_init_from_noDiv_high_e) - 1,
            "(high energy > 750 MeV/c)",
            "_no_divergence_fit",
        )
        plot_path = plot_name.replace(".png", suffix + ".png")
        fig.savefig(plot_path, dpi=150)
        print(f"Plot saved to: {plot_path}")
        plt.close(fig)
    else:
        print("ERROR: No valid data points found for fitting!")

if __name__ == "__main__":
    main()