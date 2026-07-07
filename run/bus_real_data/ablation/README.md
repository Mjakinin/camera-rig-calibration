# Ablations

Controlled AP01/AP02/AP03 sensitivity experiments.

```text
ablation/
├── _shared/
├── moving_cam/
│   ├── fov/
│   ├── motion_blur/
│   └── res/
└── world/
    └── lighting/
```

Canonical ordered variants:

- FOV: 40, 69, 100, 140 degrees.
- Motion blur: kernel 0, 9, 21, 41.
- Resolution: 160x90, 320x180, 1280x720, 2560x1440.
- Lighting: dark, low, normal, bright.

Each evaluated variant writes:

```text
results/bus_real_data/ablation/<group>/<variant>/FINAL_RESULTS/
```

There are 16 canonical packages.
