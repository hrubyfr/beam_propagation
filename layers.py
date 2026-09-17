from beam_setup import beamline_layer
import json
from pathlib import Path
import yaml


def calculate_density_ACT(refraction_index: float):
    k = 0.21
    return (refraction_index - 1) / k


def load_ACT_config(
    run_number: int,
    configpath: str = "/home/frantisek/Analysis/beam_profile_estimation/config.json",
):
    path = Path(configpath)
    config = None
    try:
        with open(path) as f:
            config = json.load(f)
    except FileNotFoundError:
        raise ValueError(f"ERROR, did not find a file in '{path}'")
    return config[str(run_number)]["aerogels"]


def make_aerogel_dictionary(run_number: int):
    conf = load_ACT_config(run_number=run_number)
    aerogel_layers = []
    for aerogel in conf:
        index = float(aerogel["refractive_index"])
        name = f"ACT_{index}"
        thickness = aerogel["thickness_cm"]
        aerogel_layers.append((name, thickness, index))

    return aerogel_layers


def open_yaml(
    path: str = "/home/frantisek/Analysis/beam_profile_estimation/wcte_beam_detectors.yaml",
):
    file_path = Path(path)

    try:
        with open(file_path) as f:
            detectors = yaml.safe_load(f)
            return detectors
    except FileNotFoundError:
        raise ValueError(f"ERROR: yaml file '{file_path}' not found")


def build_dict(run_number: int):
    complete_ACTs_dict = []
    aerogel_layers = make_aerogel_dictionary(run_number=run_number)

    yaml_config = open_yaml()
    for name, thickness, index in aerogel_layers:
        for layer_name, layer_thickness in yaml_config["detectors"][name][
            "layers_m"
        ].items():
            layer_thickness = layer_thickness * 100  # convert to cm from m
            if layer_name == "aerogel":
                complete_ACTs_dict.append(
                    (name, "SiO2", layer_thickness, calculate_density_ACT(index))
                )
            else:
                material = layer_name.split("_")[0].capitalize()
                complete_ACTs_dict.append((name, material, layer_thickness, None))
    return complete_ACTs_dict


def Calculate_total_ACT_thickness(ACTs: list):
    total_thickness_cm = 0
    for _, _, thickness_cm, _ in ACTs:
        total_thickness_cm += thickness_cm
    return total_thickness_cm


def return_built_beam(run_number: int = 1373):

    ACT_dict = build_dict(run_number)

    Vinyl_thickness_cm = 0.003e2
    Mylar_thickness_cm = 0.00005e2
    Scint_thickness_cm = 0.64

    air_gap_T4_ACTs = 11.045
    total_ACT_thickness = Calculate_total_ACT_thickness(ACT_dict)
    air_gap_T4_T1 = 131.66
    air_gap_ACTs_T1 = air_gap_T4_T1 - air_gap_T4_ACTs - total_ACT_thickness
    air_gap_HC1_T1 = 10.2
    air_gap_ACTs_HC1 = (
        air_gap_ACTs_T1
        - 10.2
        - Scint_thickness_cm
        - 2 * Mylar_thickness_cm
        - 2 * Vinyl_thickness_cm
    )
    print(f"There is a {air_gap_ACTs_T1} cm air gap between ACTs and T1")
    air_gap_T1_T5 = 203.28

    recipe = [
        # mylar beam window
        ("Mylar_beam_window", "Mylar", 0.025, None),
        # air gap to T0
        ("Air_gap", "Air", 3.7, None),
        # T0
        ("T0", "Vinyl", Vinyl_thickness_cm, None),
        ("T0", "Air", 2.075, None),
        ("T0", "Mylar", Mylar_thickness_cm, None),
        ("T0", "Scintillator", 0.64, None),
        ("T0", "Mylar", Mylar_thickness_cm, None),
        ("T0", "Air", 2.075, None),
        ("T0", "Vinyl", Vinyl_thickness_cm, None),
        # air gap T0 to T4
        ("Air_gap", "Air", 293.36, None),
        # T4
        ("T4", "Vinyl", Vinyl_thickness_cm, None),
        ("T4", "Air", 3.275, None),
        ("T4", "Mylar", Mylar_thickness_cm, None),
        ("T4", "Scintillator", 0.64, None),
        ("T4", "Mylar", Mylar_thickness_cm, None),
        ("T4", "Air", 3.275, None),
        ("T4", "Vinyl", Vinyl_thickness_cm, None),
        # air gap to ACTs
        ("Air_gap", "Air", air_gap_T4_ACTs, None),
        # ACTs
        *ACT_dict,
        # air gap to HC1
        ("Air_gap", "Air", air_gap_ACTs_HC1, None),
        # HC1
        (
            "HC1",
            "Air",
            Scint_thickness_cm + 2 * Vinyl_thickness_cm + 2 * Mylar_thickness_cm,
            None,
        ),
        # air gap to T1
        ("Air_gap", "Air", air_gap_HC1_T1, None),
        # T1
        ("T1", "Vinyl", Vinyl_thickness_cm, None),
        ("T1", "Air", 2.205, None),
        ("T1", "Mylar", Mylar_thickness_cm, None),
        ("T1", "Scintillator", 0.64, None),
        ("T1", "Mylar", Mylar_thickness_cm, None),
        ("T1", "Air", 2.205, None),
        ("T1", "Vinyl", Vinyl_thickness_cm, None),
        # air gap to T5/TOF
        ("Air_gap", "Air", air_gap_T1_T5, None),
        # T5 detector -- first half, only until the scintillators
        ("T5", "Vinyl", Vinyl_thickness_cm, None),
        ("T5", "Air", 1.525, None),
        ("T5", "Mylar", Mylar_thickness_cm, None),
    ]
    return recipe


if __name__ == "__main__":
    return_built_beam()
