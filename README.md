# Tunable Directional Emission and Collective Dissipation with Quantum Metasurfaces
Codes to reproduce the figures shown in the article: Tunable directional emission and collective dissipation with quantum metasurfaces.

<p align="center">
  <img src="https://github.com/Davtax/Tunable-emission-metasurfaces/blob/main/img/schematic_system.png" width="350" title="system_schematic">
</p>


## Dependences

This package needs the following packages:

```bash
pip install numpy
pip install scipy
pip install matplotlib
pip install tqdm
pip install joblib
```

## Usage
In [`general_functions.py`](https://github.com/Davtax/Tunable-emission-metasurfaces/blob/main/general_functions.py) contains the main functions. These functions are called from the different jupyter notebooks that reproduce the figures of the article (main text and supplemental material). Inside the folder [img](https://github.com/Davtax/Tunable-emission-metasurfaces/tree/main/img) are located the pictures used in the notebooks to schematically describe the system. Data for the energy at the Van Hove singularity for the shifted bilayer case, for different lattice distances is located in [`divergence_energy_bilayer_shift.npy`](https://github.com/Davtax/Tunable-emission-metasurfaces/blob/main/divergence_energy_bilayer_shift.npy). This data is used in several notebooks, to load it
```python
import numpy as np
from scipy.interpolate import CubicSpline

data = np.load('divergence_energy_bilayer_shift.npy', allow_pickle=True).item()
omega_X_int = CubicSpline(data['z_vec'], data['omega_X'])
```
