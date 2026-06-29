#!/usr/bin/env python3
import subprocess
import sys

if len(sys.argv) != 5:
    print("usage: python3 16_live_set_sun_direction.py dx dy dz brightness")
    print("brightness: baseline low extreme_low high")
    sys.exit(1)

dx, dy, dz = map(float, sys.argv[1:4])
brightness = sys.argv[4]

B = {
    "baseline":    ((0.80, 0.80, 0.78, 1.0), (0.20, 0.20, 0.20, 1.0)),
    "low":         ((0.42, 0.42, 0.40, 1.0), (0.08, 0.08, 0.08, 1.0)),
    "extreme_low": ((0.28, 0.28, 0.28, 1.0), (0.04, 0.04, 0.04, 1.0)),
    "high":        ((1.00, 0.98, 0.90, 1.0), (1.00, 1.00, 1.00, 1.0)),
}
diffuse, specular = B[brightness]

req = (
    f'name: "sun", '
    f'type: DIRECTIONAL, '
    f'cast_shadows: true, '
    f'direction: {{x: {dx}, y: {dy}, z: {dz}}}, '
    f'diffuse: {{r: {diffuse[0]}, g: {diffuse[1]}, b: {diffuse[2]}, a: {diffuse[3]}}}, '
    f'specular: {{r: {specular[0]}, g: {specular[1]}, b: {specular[2]}, a: {specular[3]}}}'
)

cmd = [
    "ign", "service",
    "-s", "/world/bus_real_data_camera_layout/light_config",
    "--reqtype", "ignition.msgs.Light",
    "--reptype", "ignition.msgs.Boolean",
    "--timeout", "1000",
    "--req", req,
]
print("[INFO]", req)
subprocess.run(cmd, check=False)
