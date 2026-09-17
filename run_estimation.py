from typing_extensions import runtime
from yaml import parse
import math
import beam_setup
import layers
import json
from pathlib import Path
import argparse


def load_run(
    run_number: int = 1373,
    pth: str = "/home/frantisek/Analysis/beam_profile_estimation/config.json",
):
    path = Path(pth)
    try:
        with open(path) as f:
            cf = json.load(f)
    except FileNotFoundError:
        raise ValueError(f"ERROR: file '{path}' not found, terminating...")
    momentum = cf[str(run_number)]["Beam momentum (MeV/c)"]
    measured_sigma_x_cm = cf[str(run_number)]["T5_beam_sigma_x"] * 0.1
    measured_sigma_y_cm = cf[str(run_number)]["T5_beam_sigma_y"] * 0.1

    return momentum, measured_sigma_x_cm, measured_sigma_y_cm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run_number", type=int, required=True)

    args = parser.parse_args()
    run_number = args.run_number
    momentum, measured_sigma_x, measured_sigma_y = load_run(run_number=run_number)

    detectors_list = layers.return_built_beam(run_number=run_number)

    Beam = beam_setup.beamline_setup(name="charged_hadron_beamline")
    particle = beam_setup.particle(name="pion", momentum_MeV_c=momentum)
    Beam.set_particle(particle)
    Beam.create_setup(detectors_list)
    total_sigma_mcs_squared, _ = Beam.get_final_variance(HC_veto=True)
    total_sigma_mcs = math.sqrt(total_sigma_mcs_squared)

    print("=" * 100)

    print(
        f"measured_sigma_x: {measured_sigma_x};   measured_sigma_y: {measured_sigma_y}"
    )
    print(f"sigma_MCS: {total_sigma_mcs}")
    # print(f"sigma^2 due to MCS: {total_sigma_mcs_squared}")
    try:
        print(
            f"estimated initial beam size: {math.sqrt(measured_sigma_x**2 - total_sigma_mcs_squared)} cm"
        )
    except ValueError:
        raise ValueError(
            "ERROR: square root of a negative number -- propagated beam width is wider than the measured profile"
        )
