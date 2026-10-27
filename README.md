# Subwoofer propagation: independent example

This folder contains all code and data needed for the three notebooks.
It runs without `/Users/ketan/SMU_ERDC_2024/` or access to the full raw archive.
All reusable functions are in **`utils.py`**. Figures are displayed and embedded
in the notebooks only; no PNG or SVG files are written.

## Install and run

From this folder, create the environment and register its notebook kernel:

```sh
mamba env create -f environment.yml
mamba activate sa
python -m ipykernel install --user --name sa --display-name "Python (sa)"
jupyter lab
```

Conda users can replace `mamba` with `conda`. Select the **Python (sa)**
kernel. Open and run the notebooks in order; each notebook also runs independently
of the other notebooks. Notebook 3 calculates its own gradient tables.

## Files

| File or folder | Contents |
|---|---|
| `example.pkl` | Dictionary with `stream` (three BDF ObsPy traces), `event_time` (UTC-naive timestamp), and `events` (all 1,325 chirps). Load only trusted pickle files. |
| `1_vibroseis_response.ipynb` | One-hour preprocessing, whitened PWS and linear stacks, impulse response, travel-time/celerity picks, amplitude/TL picks, and timing plots. |
| `2_observations.ipynb` | 2×2 TL/celerity comparison: WNW and South columns, with AWS surface sound-speed curves. |
| `3_gradient_estimation.ipynb` | Recompute minute and hourly gradients and plot the Figure 4 layout with RMS backgrounds and ERA5 comparisons. |
| `utils.py` | All processing, reading and inversion functions, with data paths relative to this folder. |
| `environment.yml` | Python dependencies and environment name. |
| `data/acoustic_metrics/` | Original raw-band metric CSVs for the ten requested receivers. |
| `data/aws/` | Original AWS observations and location. |
| `data/stations/` | Station coordinates, quality selection and time exclusions. |
| `data/era5/` | One CSV of hourly 0–50 m ERA5 sound-speed and wind gradients (`a_t`, `a_u`, `a_v`; s⁻¹). |
| `data/gradients/` | Recomputed WNW/South minute and hourly gradients and ERA5 comparison table. |
| `validation/` | Processing provenance and verification results. |

## Settings and interpretation

- Example centre: **2024-04-08 19:03:47 UTC**, or **2:03:47 PM CDT**.
  Recordings span 18:33:47–19:33:47 UTC, subject to the native sample grid.
  Stations: S056a (source), S007a (reference), S094a (receiver); channel BDF.
- The raw export contains exactly one hour. Chirps lacking complete extraction
  buffers at its edges are excluded. It therefore demonstrates the full method,
  rather than claiming identical picks to the database's extra-buffered window.
- Raw preprocessing: 4–20 Hz. Picking and inversion: **10–12 Hz**.
  Travel time comes from whitened PWS cross-correlation; amplitude comes from
  unwhitened linear impulse response. **TL = 20 log10(A / A_reference)**.
- Notebook 2 uses exactly the requested receivers. The source Figure 3 workflow
  also merges S068b into S068a; that extra deployment is not included here.
  The source manuscript uses AWS **adiabatic** sound speed. Notebook 2 shows it
  together with the requested **ground effective** speed, including along-path wind.
- Gradient fitting follows the existing 1-minute workflow: 60 s observation
  windows, mean SNR ≥ 3, at least three receivers, 300–1050 m range, and an
  AWS adiabatic surface speed. Hourly fits use 300 s observation windows.
  ERA5 is a reference, not an input to the inversion.
- No pressure calibration is added: raw samples and metadata are retained from
  the original reader. Relative TL uses receiver/reference amplitudes.

## Precomputed ERA5 gradients

`data/era5/gradients_0_50m.csv` contains all 48 supplied hourly times in UTC.
Each component is a linear least-squares slope over the same profile samples
as the original 0–50 m comparison (height cutoff 50.1 m for numerical tolerance).
Sound speed is computed as `sqrt(1.4 * 287.058 * T_K)` before fitting `a_t`.
For azimuth clockwise from north, `a_eff = a_t + a_u*sin(az) + a_v*cos(az)`.
Linearity of least squares preserves the original directional gradients.
The celerity comparison continues to use AWS surface sound speed. Full ERA5
profiles and higher vertical derivatives cannot be recovered from these slopes.
