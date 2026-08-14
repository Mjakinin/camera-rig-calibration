# Extension guide

The active internal registries expose four component contracts:

- `InputAdapter`
- `CalibrationMethod`
- `Evaluator`
- `ExperimentProvider`

Each component has a stable technical ID, display name, strict configuration
model, compatibility check, and the relevant command/collection hooks. The
method picker enumerates `calibration_methods` in registration order.
AP01–AP03 keep their dedicated editors; another method receives one generic
YAML field validated by its `config_model`. Adding a component does not
require a new public command or main-menu branch.

A calibration method must keep algorithm-specific work in its existing runner
or a clearly named research implementation. Its registry component declares
requirements and maps canonical paths to command arguments. Status collection
must produce a normalized dictionary while preserving all native artifacts.
Register the component once in `register_builtin_components()` (or in the
extension's startup hook), restart `rigcal`, and select it from the queue.
Methods with required config fields are prompted for their initial YAML
mapping; defaults are used when the model can be constructed without input.

## Where extension code belongs

The public facade modules are compatibility boundaries, not implementation
containers. Add new behavior to the focused area that owns it:

- method configuration and AP01/AP02/AP03-specific handlers belong below
  `ui/`, with the established YAML fallback retained for additional methods;
- queue phases belong below `queue_services/`, runtime stages below
  `runtime_services/`, and preflight checks below `preflight_services/`;
- scientific report formats belong in a focused `evaluation/reporting_*.py`
  module and publication mechanics in `publication_*.py`;
- new observation ranking or selection behavior belongs in the corresponding
  `observation_*.py` module;
- AP01/AP02 scientific changes belong in their focused method modules, never
  in the compatibility facade.

Re-export a deliberately public symbol from the existing facade when backward
compatibility requires it. If a product policy or test may replace the symbol,
resolve it through the area's typed binding object at call time. Do not import
the facade back into implementation modules except inside such a lazy binding
factory. The source-layout check rejects any productive module above 999 lines;
new modules should target at most 850.

## Adding a simulation parameter

The simulation surface is deliberately bus-only. A new bus parameter type
changes the runtime contract and is implemented through this checklist:

1. Add the validated field and baseline default to `SimulationSettings`.
2. Apply it in bus-world simulation composition/capture without rewriting unrelated
   static-camera XML.
3. Include it in experiment identity and storage classification.
4. Add its wizard row, scope explanation, documentation, and focused tests.

The GUI does not duplicate the scientific implementation; it only authors the
validated configuration. A fully dynamic parameter-form schema is deliberately
out of scope for the stable university release.

Additional Gazebo-world manifests are not an extension point in this release.
The built-in SDF, camera/sensor/topic contract and Route-1/Route-2 files form
one reviewed reproducibility boundary. Supporting another rig later requires a
new versioned product contract, not a manual path or an unvalidated manifest.

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
