# Extension guide

The active internal registries expose four component contracts:

- `InputAdapter`
- `CalibrationMethod`
- `Evaluator`
- `ExperimentProvider`

Each component has a stable technical ID, display name, strict configuration
model, compatibility check, and the relevant command/collection hooks. The
wizard derives the queue editor from registered schemas. Adding a component
does not require a new public command or a new main-menu branch.

A calibration method must keep algorithm-specific work in its existing runner
or a clearly named research implementation. Its registry component declares
requirements and maps canonical paths to command arguments. Status collection
must produce a normalized dictionary while preserving all native artifacts.

Schema v4 intentionally has no frame-selection policy contract. All active
methods receive every observation accepted by `observation_quality_v1`.
Future research selection must be introduced as a new versioned stage and
config contract rather than reusing the removed v1-v3 policy switches.

Experiment providers return labeled, independent `RigConfig` snapshots for one
dataset queue. Each variant receives a distinct `run_label`, resolved
configuration, and method directory. This supports matcher comparisons,
reference-marker sweeps, and later optimization research inside `Start a new
calibration`; it does not add another public menu.
