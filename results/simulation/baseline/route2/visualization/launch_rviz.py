import json
from pathlib import Path
from camera_rig_calibration.visualization import launch_isolated_rviz

if __name__ == '__main__':
    experiment = Path(__file__).resolve().parent.parent
    repository = next((p for p in experiment.parents if (p / 'pyproject.toml').is_file()), Path.cwd())
    print(json.dumps(launch_isolated_rviz(experiment, repository), indent=2))
