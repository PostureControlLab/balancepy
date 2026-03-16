# BalancePy

Human balance analysis using Python.

This package implements a series of published modeling approaches, where systems identification techniques are used to identify the neuro-mechanical and sensori-motor mechanisms underlying human balance.

Furthermore, the package integrates with the virtual reality balance analysis software [anaropia](https://github.com/PostureControlLab/anaropia-rfm). Data recorded with anaropia can be directly analyzed using BalancePy.

## Installation

```bash
pip install balancepy
```

For development with documentation and testing tools:

```bash
pip install balancepy[dev]
```

For working with Jupyter notebooks:

```bash
pip install balancepy[notebooks]
```

## Quick Start

```python
import balancepy

# Access submodules
from balancepy import anaropia, biomechanics, timeseries, frequency
from balancepy import data_class, make_stimulus
from balancepy.model_sim import base_model, parameter

# Load data from anaropia format
data = anaropia.getdata_anaropia("path/to/data.csv")

# Perform frequency domain analysis
freq_data = frequency.analyze(data)

# Access biomechanics calculations
com = biomechanics.get_com(mass_kg=70, height_m=1.75)
```

## Features

- **Multi-model framework**: Implement and compare different balance control models
- **Systems identification**: Parameter recovery and model fitting
- **Data handling**: Standardized time series data structures
- **Biomechanics**: Common center-of-mass and biomechanics calculations
- **Frequency domain analysis**: FFT and spectral analysis tools
- **Virtual reality integration**: Direct support for anaropia data format

## Documentation

Full documentation available at: [https://posturecontrollab.github.io/balancepy/](https://posturecontrollab.github.io/balancepy/)

## References

The package implements modeling approaches from peer-reviewed publications including:
- Peterka, R. J. (2018)
- Assländer, L. (2023)

## License

MIT License - See [LICENSE](LICENSE) file for details

## Authors

- Lorenz Assländer (lorenz.asslaender@uni-konstanz.de)
- Matthias Albrecht (matthias.albrecht@uni-konstanz.de)