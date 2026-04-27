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
import json
import math
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import beam_setup
import layers


MOMENTUM_BANDS = [(lo, lo + 50) for lo in range(200, 600, 50)] + [
    (lo, lo + 200) for lo in range(600, 1500, 200)
]
PARTICLE_NAMES = ["electron", "muon", "pion"]

PROBLEMATIC_RUNS = [
    1870,
    1873,
    1875,
    2176,
    2177,
    2216,
    2217,
    2219,
    2245,
    2254,
    2273,
    2282,
    2284,
    2320,
]


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        raise ValueError(f"ERROR: config file '{path}' not found")


def simulate_mcs(run_number: int, particle_name: str, momentum_MeV_c: float) -> float:
    detectors_list = layers.return_built_beam(run_number=run_number)
    beamline = beam_setup.beamline_setup(name="mcs_sim")
    part = beam_setup.particle(name=particle_name, momentum_MeV_c=momentum_MeV_c)
    beamline.set_particle(part)
    beamline.create_setup(detectors_list)
    _stdout = sys.stdout
    sys.stdout = io.StringIO()
    # variance = beamline.get_final_variance()
    variance = beamline.get_final_variance_HC_veto()
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


def fmt_f(val: Optional[float], w: int = 10, d: int = 4) -> str:
    return ("N/A" if val is None else f"{val:.{d}f}").rjust(w)


def fmt_i(val: int, w: int = 7) -> str:
    return str(val).rjust(w)


def fmt_band(lo: int, hi: int, w: int = 11) -> str:
    return f"{lo}-{hi}".rjust(w)


def print_table(rows: list[dict]):
    # Column layout
    # Band | Run | p_nom | sx_meas | sy_meas
    # | mcs_e | mcs_mu | mcs_pi
    # | s0x_e | s0x_mu | s0x_pi

    cols = [
        ("Band[MeV/c]", 11),
        ("Run", 7),
        ("p[MeV/c]", 9),
        ("σx_meas", 10),
        ("σy_meas", 10),
        ("σ_MCS_e", 10),
        ("σ_MCS_μ", 10),
        ("σ_MCS_π", 10),
        ("σ0x_e", 10),
        ("σ0x_μ", 10),
        ("σ0x_π", 10),
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
            fmt_f(r["s0x_e"], 10),
            fmt_f(r["s0x_mu"], 10),
            fmt_f(r["s0x_pi"], 10),
        ]
        print(row_line(cells))

    print(sep)


def plot_results(rows: list[dict], output_path: str = "band_survey_plot.png"):
    """
    Plot σx_meas, σy_meas and σ_MCS for each particle species
    as a function of nominal beam momentum.
    """
    momenta = [r["p_nom"] for r in rows]
    sx_meas = [r["sx_meas"] for r in rows]
    sy_meas = [r["sy_meas"] for r in rows]
    sx_meas_err = [r["sx_meas_err"] for r in rows]
    sy_meas_err = [r["sy_meas_err"] for r in rows]

    mcs_e = [r.get("mcs_e") for r in rows]
    mcs_mu = [r.get("mcs_mu") for r in rows]
    mcs_pi = [r.get("mcs_pi") for r in rows]

    # Filter out None values per series for clean line drawing
    def valid(xs, ys):
        pairs = [(x, y) for x, y in zip(xs, ys) if y is not None]
        return zip(*pairs) if pairs else ([], [])

    fig, ax = plt.subplots(figsize=(9, 5.5))

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
    true_sigma_y = list(calculate_true_sigma_y(sy_meas))
    ax.plot(
        momenta,
        true_sigma_y,
        color="gray",
        marker="^",
        linewidth=1.8,
        markersize=6,
        linestyle="--",
        label=r"$\sigma_{y}^{\mathrm{true}}(estimated)$",
    )

    # --- MCS per particle (coloured, dashed) ---
    COLOURS = {"electron": "#e6194b", "muon": "#4363d8", "pion": "#f58231"}
    LABELS = {
        "electron": r"$\sigma_{\mathrm{MCS}}^{e}$",
        "muon": r"$\sigma_{\mathrm{MCS}}^{\mu}$",
        "pion": r"$\sigma_{\mathrm{MCS}}^{\pi}$",
    }
    for series, name in [(mcs_e, "electron"), (mcs_mu, "muon"), (mcs_pi, "pion")]:
        px, py = valid(momenta, series)
        ax.plot(
            list(px),
            list(py),
            color=COLOURS[name],
            marker="^",
            linewidth=1.8,
            markersize=6,
            linestyle="-.",
            label=LABELS[name],
        )

    # --- run numbers as annotations above each measured point ---
    for r in rows:
        ax.annotate(
            str(r["run"]),
            xy=(r["p_nom"], r["sx_meas"]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            color="dimgray",
        )

    ax.axvline(x=500, color="gray", linestyle=":", linewidth=1.5)
    ax.text(
        500,
        ax.get_ylim()[1],
        "  trigger bias below",
        color="gray",
        fontsize=9,
        va="top",
        ha="left",
        rotation=90,
    )

    ax.set_xlabel("Nominal beam momentum (MeV/c)", fontsize=12)
    ax.set_ylabel(r"$\sigma$ (cm)", fontsize=12)
    ax.set_title(
        "Measured beam profile vs. MCS simulation across momentum bands",
        fontsize=12,
        pad=10,
    )

    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.grid(which="minor", linestyle=":", linewidth=0.3, alpha=0.4)

    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    fig.tight_layout()

    fig.savefig(output_path, dpi=150)
    print(f"Plot saved to: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="One representative run per momentum band — MCS survey"
    )
    parser.add_argument("--config", required=True, help="Path to config.json")
    parser.add_argument(
        "--output",
        default="band_survey_plot.png",
        help="Output path for the plot (default: band_survey_plot.png)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Bucket all runs, keep only the lowest per band.
    # Runs missing T5_beam_sigma_x/y are skipped — they can't be processed.
    band_representative: dict[tuple, Optional[tuple]] = {
        b: None for b in MOMENTUM_BANDS
    }

    for key, cf_run in config.items():
        rn = int(key)
        if "T5_beam_sigma_x" not in cf_run or "T5_beam_sigma_y" not in cf_run:
            print(f"NOTE: run {rn} has no T5 sigma data, skipping.")
            continue
        if (
            "T5_beam_sigma_x_error" not in cf_run
            or "T5_beam_sigma_y_error" not in cf_run
        ):
            print(f"NOTE: run {rn} has no T5 sigma error data, skipping.")
            continue
        if rn in PROBLEMATIC_RUNS:
            print(f"Run {rn} is a problematic run, skipping...")
            continue
        p = float(cf_run["Beam momentum (MeV/c)"])
        for band in MOMENTUM_BANDS:
            lo, hi = band
            if lo <= p < hi:
                current = band_representative[band]
                if current is None or rn < current[0]:
                    band_representative[band] = (rn, p)
                break

    rows = []
    for band in MOMENTUM_BANDS:
        lo, hi = band
        entry = band_representative[band]
        if entry is None:
            print(f"NOTE: no runs found in band {lo}–{hi} MeV/c, skipping.")
            continue

        rn, momentum = entry
        cf_run = config[str(rn)]
        sx_meas_cm = float(cf_run["T5_beam_sigma_x"]) * 0.1
        sy_meas_cm = float(cf_run["T5_beam_sigma_y"]) * 0.1

        sx_meas_err_cm = float(cf_run["T5_beam_sigma_x_error"])
        sy_meas_err_cm = float(cf_run["T5_beam_sigma_y_error"])

        print(
            f"Band {lo}-{hi} MeV/c → run {rn}  (p = {momentum} MeV/c) ...", flush=True
        )

        row = {
            "band_lo": lo,
            "band_hi": hi,
            "run": rn,
            "p_nom": momentum,
            "sx_meas": sx_meas_cm,
            "sy_meas": sy_meas_cm,
            "sx_meas_err": sx_meas_err_cm,
            "sy_meas_err": sy_meas_err_cm,
        }

        for pname, key_mcs, key_s0x in [
            ("electron", "mcs_e", "s0x_e"),
            ("muon", "mcs_mu", "s0x_mu"),
            ("pion", "mcs_pi", "s0x_pi"),
        ]:
            try:
                sigma_mcs = simulate_mcs(rn, pname, momentum)
                row[key_mcs] = sigma_mcs
                row[key_s0x] = initial_sigma(sx_meas_cm, sigma_mcs)
            except Exception as exc:
                print(f"  WARNING: {pname} simulation failed for run {rn}: {exc}")
                row[key_mcs] = None
                row[key_s0x] = None

        rows.append(row)

    print()
    print_table(rows)
    plot_results(rows, output_path=args.output)


if __name__ == "__main__":
    main()
