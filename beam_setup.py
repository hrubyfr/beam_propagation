from dataclasses import dataclass, field
from typing import Optional
import math

PARTICLES = {
    "electron": 0.511,
    "muon": 105.7,
    "pion": 139.6,
    "proton": 938.272,
}

RADIATION_LENGTHS_gcm2 = {
    "Air": 36.62,
    "Vinyl": 25.51,
    "Mylar": 39.95,
    "Scintillator": 43.79,
    "SiO2": 27.05,
}

MATERIAL_DENSITIES = {
    "Air": 1.205e-3,
    "Vinyl": 1.30,
    "Mylar": 1.39,
    "Scintillator": 1.032,
    "SiO2": 2.200,
}


def Highland_theta(part: "particle", material: "beamline_layer"):
    Highland_correction = 1 + 0.038 * math.log(material.thickness_X0)
    return (
        13.6
        / (part.beta * part.momentum_MeV_c)
        * part.charge
        * math.sqrt(material.thickness_X0)
        * Highland_correction
    )


def sigma_MCS_squared(part: "particle", material: "beamline_layer"):
    return (Highland_theta(part, material) * material.thickness_cm) ** 2 / 3


@dataclass
class particle:
    name: str = "electron"
    momentum_MeV_c: float = 250.0
    charge: int = 1
    variance_width_cm: float = 0.0
    variance_angle_rad: float = 0.0

    mass_MeV: float = field(init=False)
    energy_MeV: float = field(init=False)
    beta: float = field(init=False)

    def __post_init__(self):
        try:
            self.mass_MeV = PARTICLES[self.name]
        except KeyError:
            ValueError(f"ERROR: particle '{self.name}' not found")
        self.energy_MeV = math.sqrt(self.momentum_MeV_c**2 + self.mass_MeV**2)
        self.beta = self.momentum_MeV_c / self.energy_MeV


@dataclass
class beamline_layer:
    name: Optional[str] = None
    material: str = "Air"
    thickness_cm: float = 1.0
    position_cm: Optional[float] = None

    density_gcm3: Optional[float] = None
    radiation_length_gcm2: float = field(init=False)
    thickness_X0: float = field(init=False)

    def __post_init__(self):
        try:
            if self.density_gcm3 is None:
                self.density_gcm3 = MATERIAL_DENSITIES[self.material]
            self.radiation_length_gcm2 = RADIATION_LENGTHS_gcm2[self.material]
        except KeyError:
            raise ValueError(
                f"ERROR: material '{self.material}' does not exist in the database, terminating"
            )
        if self.density_gcm3 <= 0:
            raise ValueError(
                f"ERROR: density for material '{self.material}' is zero or negative, terminating"
            )
        self.thickness_X0 = self.thickness_cm / (
            self.radiation_length_gcm2 / self.density_gcm3
        )

    def propagate(self, particle: particle):
        MCS_angle_kick = Highland_theta(particle, self)
        sigma_space_scattering = 1 / math.sqrt(3) * self.thickness_cm * MCS_angle_kick
        sigma_init_free_propagation = (
            math.sqrt(particle.variance_angle_rad) * self.thickness_cm
        )
        particle.variance_width_cm = (
            particle.variance_width_cm
            + sigma_space_scattering**2
            + sigma_init_free_propagation**2
        )
        particle.variance_angle_rad = particle.variance_angle_rad + MCS_angle_kick**2

    def propagate_with_HC_veto(self, particle: particle):
        MCS_angle_kick = Highland_theta(particle, self)
        sigma_space_scattering = 1 / math.sqrt(3) * self.thickness_cm * MCS_angle_kick
        sigma_init_free_propagation = (
            math.sqrt(particle.variance_angle_rad) * self.thickness_cm
        )
        particle.variance_width_cm = (
            particle.variance_width_cm
            + sigma_space_scattering**2
            + sigma_init_free_propagation**2
        )
        particle.variance_angle_rad = particle.variance_angle_rad + MCS_angle_kick**2
        if (
            math.sqrt(particle.variance_width_cm) > (6 / math.sqrt(12))
            and self.name == "HC1"
        ):
            print("Beam width larger than HC1 hole size, slimming...")
            particle.variance_width_cm = 36 / 12
            particle.variance_angle_rad = (3 / 444.75) ** 2


class beamline_setup:
    def __init__(self, name):
        self.name = name
        self.layers = []
        self.particle: Optional[particle] = None

    def set_particle(self, particle: particle):
        self.particle = particle

    def add_layer(
        self,
        layer: beamline_layer,
    ):
        self.layers.append(layer)

    def create_setup(self, all_layers: list):
        position_from_start = 0
        for name, material, thickness, density in all_layers:
            position_from_start += thickness / 2
            self.layers.append(
                beamline_layer(
                    name=name,
                    material=material,
                    thickness_cm=thickness,
                    density_gcm3=density,
                    position_cm=position_from_start,
                )
            )
            position_from_start += thickness / 2

    def get_final_variance(self, HC_veto: bool):
        if HC_veto:
            print("Getting final variance...")
            for i, layer in enumerate(self.layers):
                layer.propagate_with_HC_veto(particle=self.particle)
                print(
                    f"Layer: {layer.material} | thickness_cm: {layer.thickness_cm} | density_gcm3: {layer.density_gcm3}"
                )
                if i == (len(self.layers) - 8):
                    print(
                        f"Estimated profile at T1 is {math.sqrt(self.particle.variance_width_cm)}"
                    )
        else:
            for i, layer in enumerate(self.layers):
                layer.propagate(particle=self.particle)
                print(
                    f"Layer: {layer.material} | thickness_cm: {layer.thickness_cm} | density_gcm3: {layer.density_gcm3}"
                )
                if i == (len(self.layers) - 8):
                    print(
                        f"Estimated profile at T1 is {math.sqrt(self.particle.variance_width_cm)}"
                    )

        return (
            self.particle.variance_width_cm,
            self.particle.variance_angle_rad,
        )


if __name__ == "__main__":
    my_setup = beamline_setup(name="beam_profile_estimation")
    my_particle = particle(name="electron", momentum_MeV_c=250)
    my_setup.set_particle(my_particle)
