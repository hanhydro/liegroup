# Online Resource 2: code and numerical results

This material accompanies the article "Explicit Error Bounds for Client-Level Private
Federated Averaging on Compact Lie Groups" and its Online Resource 1. It holds the experiment
code, the numerical results behind every reported number, table, and figure, and the camera and
edge records of the camera experiments.

## Contents

| Folder | Contents |
|---|---|
| `code/` | Python programs for the experiments, the figures, and the reporting checks |
| `results/` | Saved numerical results (`.npz`, `.json`, `.txt`) from the runs reported in the article |
| `figures/` | Fig. 1 (`overview`) and Fig. 2 (`robust_camera`) as PDF and PNG |
| `data/` | Download and checksum files for the 1DSfM Ellis Island inputs |
| `environment/` | Python package requirements and the versions used for the reported runs |

The 1DSfM inputs are third-party data and are not redistributed. `results/real_ra_edge_records.json`
records the target cameras, the candidate edges, the public-frame residuals, and the accept or
reject decision for each edge.


## Setup

Python 3.11 was used for the reported runs.

    pip install -r environment/requirements.txt

The camera experiments also need the 1DSfM inputs. The following command downloads them from
the benchmark website and verifies their SHA-256 sums.

    bash data/fetch_1dsfm.sh

## Reproducing the results

Run every command from `code/`. Each program writes to `results/` or `figures/` and overwrites
the saved file of the same name. Programs marked "data" need the 1DSfM inputs.

| Article section | Command | Output |
|---|---|---|
| 5.1 Declared-domain exit (Table 1, Fig. 1d) | `python rq1_locality_ablation.py` | `rq1_locality_ablation.npz` |
| 5.1 Post-noise radius sweep (Online Resource 1, S6.1) | `python rq1_b_sweep.py` | `rq1_b_sweep.npz` |
| 5.2 Heterogeneous rotation example, 5.7 Table 4 | `python certified_heterogeneous_so3.py` | `certified_heterogeneous_so3.json` |
| 5.2 Pseudo-Huber example (Online Resource 1, S6.3) | `python certified_pseudo_huber_so3.py` | `certified_pseudo_huber_so3.json` |
| Homogeneous implementation check (Online Resource 1, S6) | `python certified_so3.py` | `certified_so3.json` |
| 5.3 Camera task, iterative update (data) | `python real_rotation_averaging.py` | `real_ra_results.npz`, `real_ra_edge_records.json` |
| 5.3 Camera task, one-shot estimator (data) | `python rq3_oneshot_baseline.py` | `rq3_oneshot_baseline.npz` |
| 5.3 Camera task, public bounds (Table 2) | `python camera_public_bound.py` | `camera_public_bound.json` |
| 5.4 Robust camera task (data) | `python robust_camera_task.py` | `robust_camera_task.npz`, `.json` |
| 5.4 Single gradient steps (data) | `python robust_onestep_sweep.py` | `robust_onestep_sweep.npz`, `.json` |
| 5.4 Fig. 2 | `python regen_robust_fig.py` | `figures/robust_camera.pdf`, `.png` |
| 5.5 Fixed-size participation (data) | `python fixed_size_participation.py` | `fixed_size_participation.npz` |
| 5.5 Grouped privacy units (data) | `python grouped_client_stress.py` | `grouped_client_stress.npz` |
| 5.6 Noise allocation | `python rq4_schedule_seeds.py` | `rq4_schedule_seeds.npz` |
| 5.6 Short designs (Online Resource 1, S6.6) | `python rq4_short_horizon.py` | `rq4_short_horizon.npz`, `.json` |
| Fig. 1, with the bounds of panel (c) | `python regen_overview.py` | `figures/overview.pdf`, `.png` |
| Stopping residuals of the targets (data) | `python target_residuals.py` | `target_residuals.json` |
| Scaling runs (Online Resource 1, S6; not reported) | `python run_checks.py` | `results.json`, `results.txt` |

`camera_public_bound.py`, `regen_robust_fig.py`, and `regen_overview.py` read saved results and
run no simulation. `fedopt_stability.py`, `capped_schedule.py`, and `fig_style.py` are modules
used by the other programs.

## Reporting checks

    python section10_reporting_checks.py

This program recomputes the reported summaries from the files in `results/`, without
simulation or data download, and writes `results/section10_reporting_checks.json`.

## Notes

Every simulation uses a fixed seed. Results can differ at machine precision across numerical
libraries. A deployed private mechanism must draw fresh noise from a cryptographically secure
random generator instead of a seeded one.

## License

The code is released under the MIT License (see `LICENSE`). The 1DSfM inputs are not part of
this material and remain under the terms of their source.
