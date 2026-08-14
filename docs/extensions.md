# Extension guide

The active internal registries expose four component contracts:

- `InputAdapter`
- `CalibrationMethod`
- `Evaluator`
- `ExperimentProvider`

Each component has a stable technical ID, display name, strict configuration
model, compatibility check, and the relevant command/collection hooks. The
method picker enumerates `calibration_methods` in registration order.
AP01–AP03 keep their dedicated editors. A new method's Pydantic fields,
including nested models, enums, literals and booleans, become terminal-UI rows
automatically. A validated YAML row remains available as an advanced fallback;
an unusually complex method may additionally register a focused editor.
Adding a component does not require a new public command or main-menu branch.

A calibration method must keep algorithm-specific work in its own package or a
clearly named research implementation. Its registry component declares its
algorithm version, canonical input requirements, preflight check, commands,
artifact directory, status collector and result adapter. Native artifacts are
preserved, while every usable calibration is adapted to
`CanonicalMethodResult`: validated static-camera 6DOF poses in one explicit
reference frame. Publication, comparison, the results browser and the native
pose-only RViz view consume that method-independent contract.

Register the component once in `register_builtin_components()` (or in the
extension's trusted startup hook), restart `rigcal`, and select it from the
queue. Required Pydantic fields are prompted individually when the row is
created; defaults fill all other fields. The same values are stored below
`methods.extensions.<method-id>` in schema-v5 config files, so interactive UI
and prompt-free CLI execution use one configuration contract.

The copyable implementation checklist, interface fields, result convention and
example are in [Method SDK](method_sdk.md). The executable reference class is
`camera_rig_calibration.method_sdk.CanonicalPoseImportMethod`; it is not
registered by default because importing externally supplied poses is not a new
calibration algorithm.

## Where extension code belongs

The public facade modules are compatibility boundaries, not implementation
containers. Add new behavior to the focused area that owns it:

- method configuration and AP01/AP02/AP03-specific handlers belong below
  `ui/`; SDK field rendering belongs in `ui/auto_form.py`, and the validated
  YAML fallback remains available for additional methods;
- queue phases belong below `queue_services/`, runtime stages below
  `runtime_services/`, and preflight checks below `preflight_services/`;
- scientific report formats belong in a focused `evaluation/reporting_*.py`
  module and publication mechanics in `publication_services/`;
- new observation ranking or selection behavior belongs in the corresponding
  `observation_services/` module;
- AP01/AP02 scientific changes belong in their focused method modules, never
  in the compatibility facade.

Re-export a deliberately public symbol from the existing facade when backward
compatibility requires it. If a product policy or test may replace the symbol,
resolve it through the area's typed binding object at call time. Do not import
the facade back into implementation modules except inside such a lazy binding
factory. The source-layout check rejects any productive module above 999 lines;
new modules should target at most 850.

## Adding a simulation parameter or route

The simulation world is deliberately bus-only, but moving-camera routes are a
safe data extension. Put route JSON below
`data_local/simulation_routes/` (subdirectories are allowed) and restart the
wizard. The file is validated and appears by its stable relative-path ID in UI
route choices. A route contains at least two ordered, uniquely numbered frames:

```json
{
  "contract": "rigcal_simulation_route_v1",
  "frames": [
    {"frame": 0, "segment": "approach", "x": 0.0, "y": 0.0, "z": 1.2,
     "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
    {"frame": 1, "segment": "approach", "x": 0.2, "y": 0.0, "z": 1.2,
     "roll": 0.0, "pitch": 0.0, "yaw": 0.1}
  ]
}
```

Coordinates are metres and angles are radians. All pose values must be finite.
The optional target frame count deterministically resamples positions and
angles. Route content, SHA-256, resolved pose sequence and captured-frame hashes
are retained in run metadata, so changing a route invalidates capture and
downstream stages without changing a method fingerprint.

A new bus parameter type changes the runtime contract and is implemented
through this checklist:

1. Add the validated field and baseline default to `SimulationSettings`.
2. Apply it in bus-world simulation composition/capture without rewriting unrelated
   static-camera XML.
3. Include it in experiment identity and storage classification.
4. Add its wizard row, scope explanation, documentation, and focused tests.

The GUI does not duplicate the scientific implementation; it only authors the
validated configuration.

An SDF path is not an extension point. Capture accepts only the reviewed bus
world and its maintained lighting variants; the UI never asks for a world
file. Future world changes such as adding or moving markers must become typed,
validated bus-world parameters and be applied to a generated copy in
`input/simulation_variants.py`. They must not load an arbitrary user SDF or
modify the reviewed source file in place. This keeps topics, camera names,
resources, ground truth and Gazebo plugins reproducible.

Schema v5 intentionally has no frame-selection policy contract. All active
methods receive observations accepted by their effective
`observation_quality_v2` configuration and any deterministic method-specific
selection stage.
Future research selection must be introduced as a new versioned stage and
config contract rather than reusing the removed v1-v3 policy switches.

Experiment providers return labeled, independent `RigConfig` snapshots for one
dataset queue. Each variant receives a distinct `run_label`, resolved
configuration, and method directory. This supports matcher comparisons,
reference-marker sweeps, and later optimization research inside `Start a new
calibration`; it does not add another public menu.
