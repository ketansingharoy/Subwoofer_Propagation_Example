# Subwoofer propagation example

Code and example data accompanying *Tracking Near-Surface Vertical Structure in the Urban Boundary Layer with Continuous Infrasound*.

## Run

```sh
mamba env create -f environment.yml
mamba activate sa
jupyter lab
```

Run the three notebooks for:
1. Vibroseis processing and acoustic picking.
2. Celerity and transmission-loss observations.
3. Atmospheric-gradient estimation.

Functions are in `utils.py`; accompanying data are in `data/`.
This package includes selected example recordings, not the full raw archive.

## Citation

Singha Roy, K. (2026). Subwoofer Propagation Example (v1.0.0) [Software]. Zenodo.
https://doi.org/10.5281/zenodo.23891050

License: CC BY 4.0.