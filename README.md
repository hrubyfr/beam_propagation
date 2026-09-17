This is a simple simulation of the WCET beamline to estimate the approximate increase in beam width caused by multiple Coulomb scattering. 

The `beam_setup.py` file contains the particle beam class and the beamline layer class definitions, along with the logic of how the particle beam propagates through the layers. 

The `layers.py` file contains the methods and the material list to construct the layers of materials in the beamline.

The `multi_run_estimation.py` is the main file. It loads configuration files to extract the measured beam spot sizes from the analysis, it draws plots and does the steering of the simulation.
