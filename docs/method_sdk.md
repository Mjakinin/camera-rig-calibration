# Method SDK

The Method SDK is the single extension path for a calibration algorithm. A
method receives the same prepared images, moving frames, intrinsics and marker
observations regardless of whether the original acquisition was an image
folder, video, ROS bag or simulation. It publishes its native artifacts and one
method-independent static-camera 6DOF result.

## Minimal method component

Use `camera_rig_calibration.method_sdk.example_method` as the copyable working
reference. A registered component provides:

```python
@dataclass(frozen=True)
class MyMethod:
    id = "my_method"
    display_name = "My calibration method"
    algorithm_version = "my_method_v1"
    artifact_directory = "my_method"
    primary_pose_path = None
    result_contract_required = True
    input_requirements = MethodInputRequirements()
    config_model = MyMethodOptions

    def requirements(self, context: RunContext) -> RequirementResult: ...
    def commands(self, context: RunContext) -> Sequence[CommandSpec]: ...
    def collect(self, context: RunContext) -> dict: ...
    def canonical_result(
        self, context: RunContext, status: dict
    ) -> CanonicalMethodResult: ...
```

`requirements()` is the method preflight and must explain a missing executable
or incompatible input before work starts. `commands()` returns argument arrays,
never shell strings. `collect()` normalizes completion and quality status from
native artifacts. `canonical_result()` is the adapter between the solver's
native coordinate/result format and rigcal's shared result.

Register the component through `calibration_methods.register(MyMethod())` in a
trusted startup hook. For an in-tree maintained method, add the component to the
idempotent tuple in `components/registration.py`. It then appears in the method
picker, queue, runtime, publication, results browser and comparison without an
AP-specific branch.

## Parameters in UI and CLI

Define every algorithm parameter in one strict Pydantic model:

```python
class MyMethodOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    iterations: int = Field(default=50, ge=1, description="Solver budget.")
    loss: Literal["linear", "huber"] = Field(
        default="huber", description="Robust loss."
    )
```

The wizard automatically renders validated rows for scalar fields, nested
models, booleans, enums and literals. Required fields are requested when a
method row is added. A complex interactive workflow may expose a `config_editor`
object with `edit(console, config) -> BaseModel`; the normal auto-generated rows
and YAML fallback remain available.

The CLI consumes exactly the values authored by the UI:

```yaml
methods:
  enabled: [my_method]
  extensions:
    my_method:
      iterations: 50
      loss: huber
```

Run the saved queue or config without method-specific flags:

```bash
rigcal --config workspace/<dataset>/queue/queue.yaml --dry-run
rigcal --config workspace/<dataset>/queue/queue.yaml --yes
```

## Canonical 6DOF result

Each usable calibration result has contract
`rigcal_canonical_method_result_v1`, one `reference_frame`, and one pose per
available static camera. Its transform is:

```text
T_reference_camera
p_reference = T_reference_camera @ p_camera
translation: metres
RPY: radians, R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
quaternion order: qx, qy, qz, qw
```

Create poses with `CanonicalCameraPose.from_transform(...)`; this validates a
finite rigid SE(3) matrix and derives consistent translation, quaternion and
RPY values. `CanonicalMethodResult` rejects duplicate cameras, mixed reference
frames and an `available` result without poses.

The runtime writes both:

```text
<artifact_directory>/canonical_method_result.json
<artifact_directory>/camera_poses_6dof.csv
```

Publication copies them into the public method directory, creates the
established `camera_extrinsics.csv` bridge for an SDK method, and exposes pose
count/status in `RESULT.json`, comparison files and `View results`. The native
pose-only RViz scene works for canonical variants that use the same reference
frame; an AP03 point cloud is optional context, not a requirement.

`status: incomplete` or `diagnostic` may preserve useful solver diagnostics but
is not a usable calibration. Native outputs remain in diagnostics and are never
discarded by the adapter.

## Extension checklist

1. Put scientific code and tests in a focused method package.
2. Define the strict Pydantic options and stable algorithm version.
3. Declare canonical input requirements and implement preflight.
4. Return explicit `CommandSpec` stages or no subprocess for an in-process adapter.
5. Collect native status without renaming or discarding native artifacts.
6. Convert poses to `CanonicalMethodResult` and test transform direction.
7. Register once and test UI creation, CLI config, queue execution, publication and results.
8. Keep new modules below 850 lines; every productive module must remain below 1000.
