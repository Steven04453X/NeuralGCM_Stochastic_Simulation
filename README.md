# NeuralGCM Stochastic Continuous Run

Scripts for running continuous (multi-month) free simulations using [NeuralGCM](https://github.com/google-research/neuralgcm), a neural general circulation model for weather and climate, initialized from ERA5 reanalysis data.

> **Reference:** 
- Kochkov, D., and Coauthors, 2024: Neural general circulation models for weather and climate. *Nature*. https://doi.org/10.1038/s41586-024-07744-y
- Yuval, J., et al. (2026). Neural general circulation models optimized to predict satellite-based precipitation observations. *Science Advances*, 12(eadv6891), 1–11. https://doi.org/10.1126/sciadv.adv6891
- Chen, Z., Leung, L. R., Zhou, W., Lu, J., Lubis, S. W., Liu, Y., et al. (2026). Hierarchical Testing of a Hybrid Machine Learning‐Physics Global Atmosphere Model. *AGU Advances*, 7(2), 1–23. https://doi.org/10.1029/2025AV002075


---

## Features

- Continuous free-running simulation initialized from ERA5 at a user-specified date
- Support for multi-month/multi-year simulation spans
- Optional **uniform SST warming** experiments (AMIP-P*x*K)
- Hourly or daily output in NetCDF format
- Precipitation computed as hourly increments (mm day⁻¹) from NeuralGCM's cumulative output
- Designed to run on GPU nodes (tested on NERSC Perlmutter)

---

## Repository Structure

```
.
├── 01.Control_07.StochasticRun_test_ContinueRun.sh    # SLURM/bash control script
├── 02.Stochastic_ContinuousRun_IndicateInitCon.py     # Main Python simulation script
├── ERA5/                                              # ERA5 input data (not tracked by git)
│   ├── e5.oper.an.sfc/                                # Surface variables (SST, sea ice, SKT)
│   └── e5.oper.an.pl/                                 # Pressure-level variables (u, v, T, Z, q, etc.)
├── Models/                                            # NeuralGCM model weights (not tracked by git)
│   └── models_v1_precip_stochastic_precip_2_8_deg.pkl
└── logs/                                              # Log files (not tracked by git)
```

---

## Requirements

### Python environment

This code runs with the `py_ai2` conda environment. Key dependencies:

- [`neuralgcm`](https://github.com/google-research/neuralgcm)
- [`dinosaur`](https://github.com/google-research/dinosaur)
- `jax` / `jaxlib` (GPU-enabled)
- `xarray`, `numpy`, `netCDF4`

### Model weights

Download NeuralGCM model weights (`.pkl`) from the [NeuralGCM releases](https://github.com/google-research/neuralgcm) and place them in the `Models/` directory.

> **Direct download (hosted on Zenodo):** [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
> *(Replace the DOI placeholder once you upload to Zenodo)*

---

## Data Availability

Large files are **not tracked by this repository**. Download them from the links below and place them in the corresponding directories.

| Data | Directory | Source | Download |
|------|-----------|--------|---------|
| NeuralGCM model weights (`.pkl`) | `Models/` | Google / NeuralGCM | [NeuralGCM GitHub releases](https://github.com/google-research/neuralgcm) |
| ERA5 reanalysis (pressure-level & surface) | `ERA5/` | ECMWF / CISL RDA | [Copernicus CDS](https://cds.climate.copernicus.eu) / [CISL RDA ds633.6](https://rda.ucar.edu/datasets/d633006/) |
| Sample ERA5 subset for testing | `ERA5/` | This study | [![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX) *(placeholder)* |
| Sample output (2025-12 AMIP control) | `Output_AMIP_InitCon*/` | This study | [![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX) *(placeholder)* |

> **How to upload to Zenodo:**
> 1. Go to [https://zenodo.org](https://zenodo.org) and log in (free, supports login via GitHub/ORCID).
> 2. Click **New Upload**, drag your files (e.g., `ERA5_subset.tar.gz`, `Output_AMIP_InitCon20251201.tar.gz`).
> 3. Fill in the metadata (title, authors, license). Under **Related works**, link to this GitHub repo.
> 4. Click **Publish** — Zenodo will assign a permanent DOI.
> 5. Replace the `XXXXXXX` placeholders above with the actual DOI numbers.

---

### ERA5 data

The following ERA5 variables are required (available via [Copernicus CDS](https://cds.climate.copernicus.eu)):

| Type           | Variables                                      |
|----------------|------------------------------------------------|
| Pressure-level | `u`, `v`, `z`, `t`, `q`, `ciwc`, `clwc`       |
| Surface        | `ci` (sea ice), `sstk` (SST), `skt` (skin T)  |

Expected file naming convention (CISL RDA format):
```
e5.oper.an.pl.128_131_u.ll025uv.YYYYMMDDHH_YYYYMMDDHH.nc   # example
e5.oper.an.sfc.128_031_ci.ll025sc.YYYYMM0100_YYYYMMddHH.daymean.nc
```

---

## Usage

### 0. Set up conda env, py_ai2
conda env create -f py_ai2.yml --name py_ai2

### 1. Configure the control script

Edit `01.Control_07.StochasticRun_test_ContinueRun.sh` to set:

| Variable            | Description                                      |
|---------------------|--------------------------------------------------|
| `iyr`, `i_Mon`, `iDD` | Initial condition date (ERA5 start date)       |
| `SimulationPeriod`  | Start and end dates of the simulation            |
| `i_UnifWarming`     | (Optional) Uniform SST warming in K (AMIP-P*x*K)|

### 2. Run interactively or submit to SLURM

**Interactive:**
```bash

conda activate py_ai2
bash 01.Control_02.StochasticRun_test_ContinueRun.sh 2>&1 | tee logs/run.out
```

**SLURM (Perlmutter GPU node):**
```bash
sbatch 01.Control_02.StochasticRun_test_ContinueRun.sh
```

The SLURM script requests:
- 1 node, 4 A100 80 GB GPUs
- 128 CPUs per task
- Up to 14 hours walltime

### 3. Uniform warming experiment (AMIP-P*x*K)

Uncomment the warming block in the control script and set `i_UnifWarming` (e.g., `1` for +1 K SST). The output directory will be named automatically, e.g., `Output_amip-p1K_InitCon20251201/`.

---

## Output

NetCDF files are saved to `./Output_AMIP_InitCon<YYYYMMDD>/` (or `./Output_amip-pXK_InitCon<YYYYMMDD>/` for warming runs).

| Variable                          | Units         | Description                        |
|-----------------------------------|---------------|------------------------------------|
| `temperature`                     | K             | Air temperature                    |
| `geopotential`                    | m² s⁻²        | Geopotential                       |
| `u_component_of_wind`             | m s⁻¹         | Zonal wind                         |
| `v_component_of_wind`             | m s⁻¹         | Meridional wind                    |
| `specific_humidity`               | kg kg⁻¹       | Specific humidity                  |
| `specific_cloud_liquid_water_content` | kg kg⁻¹   | Cloud liquid water                 |
| `specific_cloud_ice_water_content`    | kg kg⁻¹   | Cloud ice water                    |
| `precipitation`                   | mm day⁻¹      | Hourly precipitation (incremental) |
| `precipitation_cumulative_mean`   | kg m⁻² hr⁻¹  | Cumulative precipitation (raw)     |
| `evaporation`                     | mm day⁻¹      | Evaporation (incremental)          |

Time axis is encoded as `YYYYMMDD.fraction` (fraction = hour/24).

---

## Authors

- **Ziming Chen** — PNNL (ziming.chen@pnnl.gov)
- **Ya Wang** — IAP, CAS (wangya@mail.iap.ac.cn)
- **Letian Gu** — IAP, CAS

---

## License

Please cite the NeuralGCM paper if you use this code in your research:

> Kochkov, D., and Coauthors, 2024: Neural general circulation models for weather and climate. *Nature*. https://doi.org/10.1038/s41586-024-07744-y
