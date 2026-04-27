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
                    ("SiO2", layer_thickness, calculate_density_ACT(index))
                )
            else:
                material = layer_name.split("_")[0].capitalize()
                complete_ACTs_dict.append((material, layer_thickness, None))
    return complete_ACTs_dict


def Calculate_total_ACT_thickness(ACTs: list):
    total_thickness_cm = 0
    for _, thickness_cm, _ in ACTs:
        total_thickness_cm += thickness_cm
    return total_thickness_cm


def return_built_beam(run_number: int = 1373):

    ACT_dict = build_dict(run_number)

    Vinyl_thickness_cm = 0.003e2
    Mylar_thickness_cm = 0.00005e2

    air_gap_T4_ACTs = 11.045
    total_ACT_thickness = Calculate_total_ACT_thickness(ACT_dict)
    air_gap_T4_T1 = 131.66
    air_gap_ACTs_T1 = air_gap_T4_T1 - air_gap_T4_ACTs - total_ACT_thickness
    air_gap_T1_T5 = 203.28

    recipe = [
        # mylar beam window
        ("Mylar", 0.025, None),
        # air gap to T0
        ("Air", 3.7, None),
        # T0
        ("Vinyl", Vinyl_thickness_cm, None),
        ("Air", 2.075, None),
        ("Mylar", Mylar_thickness_cm, None),
        ("Scintillator", 0.64, None),
        ("Mylar", Mylar_thickness_cm, None),
        ("Air", 2.075, None),
        ("Vinyl", Vinyl_thickness_cm, None),
        # air gap T0 to T4
        ("Air", 293.36, None),
        # T4
        ("Vinyl", Vinyl_thickness_cm, None),
        ("Air", 3.275, None),
        ("Mylar", Mylar_thickness_cm, None),
        ("Scintillator", 0.64, None),
        ("Mylar", Mylar_thickness_cm, None),
        ("Air", 3.275, None),
        ("Vinyl", Vinyl_thickness_cm, None),
        # air gap to ACTs
        ("Air", air_gap_T4_ACTs, None),
        # ACTs
        *ACT_dict,
        # air gap to T1
        ("Air", air_gap_ACTs_T1, None),
        # T1
        ("Vinyl", Vinyl_thickness_cm, None),
        ("Air", 2.205, None),
        ("Mylar", Mylar_thickness_cm, None),
        ("Scintillator", 0.64, None),
        ("Mylar", Mylar_thickness_cm, None),
        ("Air", 2.205, None),
        ("Vinyl", Vinyl_thickness_cm, None),
        # air gap to T5/TOF
        ("Air", air_gap_T1_T5, None),
        # T5 detector -- first half, only until the scintillators
        ("Vinyl", Vinyl_thickness_cm, None),
        ("Air", 1.525, None),
        ("Mylar", Mylar_thickness_cm, None),
    ]
    return recipe
