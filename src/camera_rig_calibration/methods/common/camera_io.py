"""Read normalized ROS/OpenCV camera-info JSON."""
import json
from pathlib import Path
import numpy as np


def load_camera_info_json(path):
    path = Path(path)
    if not path.exists():
        raise RuntimeError(f"Missing camera_info JSON: {path}")

    data = json.loads(path.read_text())

    # ROS/Gazebo exported shape used in this project.
    if "k" in data:
        K = np.asarray(data["k"], dtype=np.float64).reshape(3, 3)
    elif "K" in data:
        K = np.asarray(data["K"], dtype=np.float64).reshape(3, 3)
    elif "camera_matrix" in data:
        cm = data["camera_matrix"]
        if isinstance(cm, dict) and "data" in cm:
            K = np.asarray(cm["data"], dtype=np.float64).reshape(3, 3)
        else:
            K = np.asarray(cm, dtype=np.float64).reshape(3, 3)
    else:
        fx = data["fx"]
        fy = data.get("fy", fx)
        cx = data["cx"]
        cy = data["cy"]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    if "d" in data:
        D = np.asarray(data["d"], dtype=np.float64).reshape(-1)
    elif "D" in data:
        D = np.asarray(data["D"], dtype=np.float64).reshape(-1)
    elif "distortion_coefficients" in data:
        dc = data["distortion_coefficients"]
        if isinstance(dc, dict) and "data" in dc:
            D = np.asarray(dc["data"], dtype=np.float64).reshape(-1)
        else:
            D = np.asarray(dc, dtype=np.float64).reshape(-1)
    elif "distortion" in data:
        D = np.asarray(data["distortion"], dtype=np.float64).reshape(-1)
    else:
        D = np.zeros(5, dtype=np.float64)

    width = data.get("width", data.get("image_width", None))
    height = data.get("height", data.get("image_height", None))

    return {
        "K": K,
        "D": D,
        "width": width,
        "height": height,
        "raw": data,
    }
