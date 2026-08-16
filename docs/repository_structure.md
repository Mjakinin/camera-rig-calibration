# Repository structure

The repository root contains only durable project areas. Generated Python
caches, local input data, temporary runs, and large new result payloads are
ignored by Git.

| Path | Purpose | Keep in the submission? |
|---|---|---|
| `config/` | Versioned configuration templates and managed camera-intrinsics profiles | Yes |
| `data_local/` | Local landing zone for images, video, ROS bags, and optional route JSON files | Only its instructions; user data stays local |
| `docs/` | Architecture, configuration, extension, Method-SDK, and storage documentation | Yes |
| `examples/` | Small runnable UI/CLI configuration examples | Yes |
| `results/` | Locally generated published experiments and scientific outputs | README only; generated results stay local |
| `src/camera_rig_calibration/` | Installable Python application | Yes |
| `src/calib_lab/` | Reviewed ROS/Gazebo bus-world package, marker layout, routes, and numbered asset generators | Yes |
| `tests/` | Behavior, scientific-contract, UI/CLI, and architecture tests | Yes |
| `tools/` | The two repository-wide checks: source-size limit and tracked-file hygiene | Yes |
| `workspace/` | Regenerable preparation caches, method caches, and resumable temporary runs | No; local and ignored |

There is deliberately no root `run/`: `rigcal` and
`python -m camera_rig_calibration` are the two supported entrypoints. There is
no root `datasets/`: an experiment's immutable input is stored together with
its published result, while regenerable preparation data lives in
`workspace/preparation_cache/`. One-off maintenance scripts and frozen
comparison archives are not product code and are not part of the cleaned
layout.

## Python package map

Python modules are named by responsibility, not by an artificial execution
number. Imports do not execute in filename order; `00_`/`01_` names would
therefore be misleading. The actual pipeline order is declared by the queue
and runtime coordinators and printed in the terminal.

```text
camera_rig_calibration/
  cli.py, wizard.py, queueing.py, runtime.py, preflight.py,
  publication.py, observations.py, experiments.py   stable facades
  components/ + registry.py                         component registration
  method_sdk/                                       new-method contract + 6DOF result
  methods/ap01, ap02, ap03/                         scientific algorithms
  config/                                           validated configuration
  dataset/ + input/                                 discovery and preparation
  observation_services/                             detection, filtering, selection
  queue_services/                                   queue state and execution phases
  runtime_services/                                 environment, commands, stages, cache
  preflight_services/                               validation and common-anchor phases
  evaluation/                                       method-independent reporting
  publication_services/                             atomic dataset/result publication
  visualization/                                    pose and RViz output
  ui/                                               terminal wizard screens and editors
  policies/                                         installed product-policy composition
  experiment_services/                              experiment identity and manifests
```

The small facade files preserve stable UI/CLI and import boundaries. Actual
implementations live in their focused subpackages. Every productive Python
file is checked at 999 lines or fewer; new modules target 850 lines or fewer.

`policies/` contains cross-cutting composition only. It is intentionally
separate from scientific algorithms: selection authority, report front doors,
result display, and product defaults can wrap stable facades without being
copied into AP01/AP02/AP03. Its installation order is explicit in
`bootstrap.py`.
