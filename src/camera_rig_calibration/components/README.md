# Runtime components

This package connects maintained implementations to rigcal's registries. It
does not contain calibration mathematics.

- `inputs.py` validates the selected acquisition source.
- `evaluation.py` schedules the common post-method evaluation.
- `experiments.py` creates controlled configuration variants.
- `registration.py` registers the built-in components once.
- `common.py` contains the small input/status contract shared by AP01–AP03.

The method-specific pipeline plans live beside their implementations:

- `methods/ap01/pipeline.py`
- `methods/ap02/pipeline.py`
- `methods/ap03/pipeline.py`

Those files only define stage order, command-line arguments and dependencies.
The numerical work is performed by the other modules in each method folder.
