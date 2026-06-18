# Common Utilities

Shared utilities used by more than one experiment world.

Keep this folder small. Experiment-specific code should stay inside its experiment folder:

```text

bus_corridor_relay/

```

---

## Files

```text
common/scripts/transform_utils.py
```

Actual transform and pose math implementation used by the calibration scripts.

```text
common/transform_utils.py
```

Compatibility wrapper for older imports:

```python
from common.transform_utils import ...
```

The wrapper forwards to:

```python
from common.scripts.transform_utils import *
```

This avoids breaking existing Checkerboard and ArUco scripts after the repository was reorganized.

---

## Rule of Thumb

Put code here only if it is truly shared by multiple experiment worlds. Otherwise, keep it in the corresponding experiment folder.
