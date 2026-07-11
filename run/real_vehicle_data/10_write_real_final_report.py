#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import xml.etree.ElementTree as ET
from itertools import combinations
from pathlib import Path

import numpy as np


CAMERAS = [
    'cam_edge_0',
    'cam_edge_1',
    'cam_edge_3',
    'cam_edge_5',
]
METHODS = ['AP01', 'AP02', 'AP03']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Create one real-data report for AP01, AP02 and AP03, including '
            'GT-free diagnostics, optional real measurements, and a clearly '
            'labelled comparison against the simulation camera layout.'
        )
    )
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--results-root', required=True)
    parser.add_argument(
        '--simulation-world-sdf',
        default='src/calib_lab/bus_real_data/worlds/bus_real_data_moving_camera.sdf',
    )
    return parser.parse_args()


def read_json(path: Path, default=None):
    if not path.is_file():
        return {} if default is None else default
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return {} if default is None else default


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def pose_positions(path: Path) -> dict[str, np.ndarray]:
    result = {}
    for row in read_csv(path):
        camera = row.get('entity_id', '')
        if camera not in CAMERAS:
            continue
        try:
            result[camera] = np.asarray(
                [float(row['x_m']), float(row['y_m']), float(row['z_m'])],
                dtype=np.float64,
            )
        except Exception:
            continue
    return result


def simulation_positions_from_sdf(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        return {}

    root = ET.parse(path).getroot()
    result = {}

    for model in root.iter('model'):
        name = model.attrib.get('name', '')
        if name not in CAMERAS:
            continue

        pose = model.find('pose')
        if pose is None or not pose.text:
            continue

        values = [float(value) for value in pose.text.split()]
        if len(values) < 3:
            continue

        result[name] = np.asarray(values[:3], dtype=np.float64)

    return result


def pair_key(first: str, second: str) -> str:
    return f'{first}--{second}'


def distance_map(positions: dict[str, np.ndarray]) -> dict[str, float]:
    result = {}
    for first, second in combinations(CAMERAS, 2):
        if first in positions and second in positions:
            result[pair_key(first, second)] = float(
                np.linalg.norm(positions[first] - positions[second])
            )
    return result


def format_value(value, precision: int = 6) -> str:
    if value is None:
        return 'NA'
    try:
        number = float(value)
    except Exception:
        return 'NA'
    if not math.isfinite(number):
        return 'NA'
    return f'{number:.{precision}f}'


def method_paths(results_root: Path) -> dict:
    return {
        'AP01': {
            'root': results_root / '02_ap01_real',
            'pose': (
                results_root / '02_ap01_real' / '03_static_extrinsics'
                / 'AP01_STATIC_CAMERA_POSES_CAM3_REFERENCE.csv'
            ),
            'diagnostics': (
                results_root / '02_ap01_real' / '03_static_extrinsics'
                / 'AP01_DIAGNOSTICS.json'
            ),
        },
        'AP02': {
            'root': results_root / '03_ap02_real',
            'pose': (
                results_root / '03_ap02_real' / '07_graph_ba' / 'with_moving'
                / 'optimized_static_camera_poses_ref_marker.csv'
            ),
            'diagnostics': (
                results_root / '03_ap02_real' / '08_final_results'
                / 'AP02_DIAGNOSTICS.json'
            ),
        },
        'AP03': {
            'root': results_root / '04_ap03_real',
            'pose': (
                results_root / '04_ap03_real' / '07_final_results'
                / 'AP03_MARKER_SIZE_SCALE_ONLY_STATIC_CAMERA_POSES.csv'
            ),
            'diagnostics': (
                results_root / '04_ap03_real' / '07_final_results'
                / 'AP03_DIAGNOSTICS.json'
            ),
        },
    }


def diagnostic_lines(method: str, diagnostics: dict) -> list[str]:
    if not diagnostics:
        return ['  diagnostics: unavailable']

    lines = []

    if method == 'AP01':
        scale = diagnostics.get('metric_scale', {})
        lines.extend([
            f"  registered moving frames: {diagnostics.get('registered_moving_frames', 'NA')}",
            f"  input moving frames: {diagnostics.get('input_moving_frames', 'NA')}",
            f"  scale m/COLMAP unit: {format_value(scale.get('scale_m_per_colmap_unit'), 9)}",
            f"  scale relative std: {format_value(scale.get('used_relative_std'), 6)}",
            f"  static camera methods: {diagnostics.get('static_camera_methods', {})}",
        ])
    elif method == 'AP02':
        reprojection = diagnostics.get('reprojection', {})
        lines.extend([
            f"  selected moving frames: {diagnostics.get('selected_moving_frames', 'NA')}",
            f"  optimized moving frames: {diagnostics.get('optimized_moving_frames', 'NA')}",
            f"  reprojection mean px: {format_value(reprojection.get('mean_reprojection_px'), 4)}",
            f"  reprojection median px: {format_value(reprojection.get('median_reprojection_px'), 4)}",
            f"  reprojection max px: {format_value(reprojection.get('maximum_reprojection_px'), 4)}",
            f"  positive-depth fraction: {format_value(diagnostics.get('positive_depth_fraction'), 6)}",
            f"  estimated rig diameter m: {format_value(diagnostics.get('estimated_rig_diameter_m'), 6)}",
        ])
    elif method == 'AP03':
        reconstruction = diagnostics.get('reconstruction', {})
        scale = diagnostics.get('marker_scale', {})
        lines.extend([
            f"  registered images: {reconstruction.get('registered_images', scale.get('registered_images', 'NA'))}",
            f"  registered static cameras: {reconstruction.get('registered_static_cameras', scale.get('registered_static_cameras', 'NA'))}",
            f"  registered moving frames: {reconstruction.get('registered_moving_frames', scale.get('registered_moving_frames', 'NA'))}",
            f"  sparse points: {reconstruction.get('num_3d_points', scale.get('num_sparse_points3d', 'NA'))}",
            f"  scale m/COLMAP unit: {format_value(scale.get('scale_m_per_colmap_unit'), 9)}",
            f"  scale relative std: {format_value(scale.get('used_rel_std_scale'), 6)}",
            f"  triangulated marker corners: {scale.get('triangulated_marker_corners', 'NA')}",
        ])

    lines.append(
        f"  available static cameras: {diagnostics.get('available_static_cameras', [])}"
    )
    lines.append(
        f"  missing static cameras: {diagnostics.get('missing_static_cameras', [])}"
    )
    lines.append(
        f"  runtime seconds: {format_value(diagnostics.get('runtime_seconds'), 2)}"
    )
    return lines


def main() -> None:
    args = parse_args()
    dataset = Path(args.dataset).resolve()
    results_root = Path(args.results_root).resolve()
    simulation_world = Path(args.simulation_world_sdf).resolve()

    final_root = results_root / '99_FINAL_RESULTS'
    final_root.mkdir(parents=True, exist_ok=True)

    paths = method_paths(results_root)
    statuses = {}
    positions = {}
    distances = {}
    diagnostics = {}

    for method in METHODS:
        root = paths[method]['root']
        statuses[method] = read_json(
            root / 'METHOD_STATUS.json',
            {'method': method, 'status': 'NOT_RUN', 'success': False},
        )
        positions[method] = pose_positions(paths[method]['pose'])
        distances[method] = distance_map(positions[method])
        diagnostics[method] = read_json(paths[method]['diagnostics'], {})

    simulation_positions = simulation_positions_from_sdf(simulation_world)
    simulation_distances = distance_map(simulation_positions)

    simulation_reference_path = final_root / 'simulation_gt_reference_distances.json'
    simulation_reference_path.write_text(
        json.dumps(
            {
                'source_world_sdf': str(simulation_world),
                'warning': (
                    'These are camera-center distances from the Gazebo simulation layout. '
                    'They are a contextual reference only and are not real-world ground truth.'
                ),
                'distances_m': simulation_distances,
            },
            indent=2,
        ) + '\n',
        encoding='utf-8',
    )

    measured_reference_path = final_root / 'measured_reference_distances.json'
    if not measured_reference_path.is_file():
        measured_reference_path.write_text(
            json.dumps(
                {
                    pair_key(first, second): None
                    for first, second in combinations(CAMERAS, 2)
                },
                indent=2,
            ) + '\n',
            encoding='utf-8',
        )
    measured_references = read_json(measured_reference_path, {})

    rows = []
    for first, second in combinations(CAMERAS, 2):
        key = pair_key(first, second)

        measured = measured_references.get(key)
        try:
            measured = float(measured) if measured is not None else None
        except Exception:
            measured = None

        simulation_gt = simulation_distances.get(key)

        row = {
            'camera_a': first,
            'camera_b': second,
            'ap01_distance_m': distances['AP01'].get(key),
            'ap02_distance_m': distances['AP02'].get(key),
            'ap03_distance_m': distances['AP03'].get(key),
            'simulation_gt_reference_m': simulation_gt,
            'measured_real_reference_m': measured,
        }

        for method in METHODS:
            value = distances[method].get(key)
            prefix = method.lower()
            row[f'{prefix}_vs_simulation_gt_cm'] = (
                abs(value - simulation_gt) * 100.0
                if value is not None and simulation_gt is not None
                else None
            )
            row[f'{prefix}_vs_measured_real_cm'] = (
                abs(value - measured) * 100.0
                if value is not None and measured is not None
                else None
            )

        values = {method: distances[method].get(key) for method in METHODS}
        row['ap01_vs_ap02_cm'] = (
            abs(values['AP01'] - values['AP02']) * 100.0
            if values['AP01'] is not None and values['AP02'] is not None
            else None
        )
        row['ap01_vs_ap03_cm'] = (
            abs(values['AP01'] - values['AP03']) * 100.0
            if values['AP01'] is not None and values['AP03'] is not None
            else None
        )
        row['ap02_vs_ap03_cm'] = (
            abs(values['AP02'] - values['AP03']) * 100.0
            if values['AP02'] is not None and values['AP03'] is not None
            else None
        )
        rows.append(row)

    pairwise_fields = [
        'camera_a',
        'camera_b',
        'ap01_distance_m',
        'ap02_distance_m',
        'ap03_distance_m',
        'simulation_gt_reference_m',
        'ap01_vs_simulation_gt_cm',
        'ap02_vs_simulation_gt_cm',
        'ap03_vs_simulation_gt_cm',
        'measured_real_reference_m',
        'ap01_vs_measured_real_cm',
        'ap02_vs_measured_real_cm',
        'ap03_vs_measured_real_cm',
        'ap01_vs_ap02_cm',
        'ap01_vs_ap03_cm',
        'ap02_vs_ap03_cm',
    ]
    pairwise_csv = final_root / 'REAL_DATA_PAIRWISE_DISTANCES.csv'
    write_csv(pairwise_csv, rows, pairwise_fields)

    status_rows = []
    for method in METHODS:
        status = statuses[method]
        status_rows.append({
            'method': method,
            'status': status.get('status', 'UNKNOWN'),
            'success': status.get('success', False),
            'available_static_cameras': ';'.join(sorted(positions[method])),
            'camera_count': len(positions[method]),
            'runtime_seconds': status.get(
                'runtime_seconds', diagnostics[method].get('runtime_seconds')
            ),
            'error': status.get('error', ''),
            'pose_file': str(paths[method]['pose']),
            'diagnostics_file': str(paths[method]['diagnostics']),
        })

    status_csv = final_root / 'REAL_DATA_METHOD_STATUS.csv'
    write_csv(
        status_csv,
        status_rows,
        [
            'method',
            'status',
            'success',
            'available_static_cameras',
            'camera_count',
            'runtime_seconds',
            'error',
            'pose_file',
            'diagnostics_file',
        ],
    )

    moving_count = len(list((dataset / 'raw_images' / 'moving').glob('frame_*.png')))
    static_multi_root = dataset / 'raw_images' / 'static_multi'
    static_multi_counts = {
        camera: len(list((static_multi_root / camera).glob('*.png')))
        for camera in CAMERAS
    }

    lines = [
        'REAL-DATA CAMERA-RIG CALIBRATION: AP01 / AP02 / AP03',
        '=' * 160,
        '',
        f'Dataset: {dataset}',
        'Primary evaluation mode: GT-free.',
        (
            'Additional context: comparison against Gazebo simulation camera-center '
            'distances. These differences are NOT real-world accuracy errors.'
        ),
        f'Simulation reference source: {simulation_world}',
        'Marker dictionary: DICT_4X4_50',
        'Marker side length: 0.170 m',
        f'Moving frames: {moving_count}',
        f'Selected static multi-frames: {static_multi_counts}',
        '',
        'METHOD STATUS',
        '-' * 160,
        f"{'Method':8s}{'Status':28s}{'Cameras':>10s}{'Runtime [s]':>18s}  Error",
    ]

    for method in METHODS:
        status = statuses[method]
        error = str(status.get('error', ''))
        if len(error) > 62:
            error = error[:62] + '...'
        lines.append(
            f"{method:8s}"
            f"{str(status.get('status', 'NOT_RUN')):28s}"
            f"{len(positions[method]):>4d}/4      "
            f"{format_value(status.get('runtime_seconds'), 2):>15s}  "
            f"{error}"
        )

    lines.extend([
        '',
        'PAIRWISE CAMERA DISTANCES WITH SIMULATION-LAYOUT COMPARISON',
        '-' * 160,
        (
            f"{'Camera pair':29s}"
            f"{'AP01 [m]':>12s}"
            f"{'AP02 [m]':>12s}"
            f"{'AP03 [m]':>12s}"
            f"{'Sim GT [m]':>13s}"
            f"{'|AP01-Sim| [cm]':>18s}"
            f"{'|AP02-Sim| [cm]':>18s}"
            f"{'|AP03-Sim| [cm]':>18s}"
        ),
    ])

    for row in rows:
        pair = f"{row['camera_a']} - {row['camera_b']}"
        lines.append(
            f"{pair:29s}"
            f"{format_value(row['ap01_distance_m']):>12s}"
            f"{format_value(row['ap02_distance_m']):>12s}"
            f"{format_value(row['ap03_distance_m']):>12s}"
            f"{format_value(row['simulation_gt_reference_m']):>13s}"
            f"{format_value(row['ap01_vs_simulation_gt_cm'], 3):>18s}"
            f"{format_value(row['ap02_vs_simulation_gt_cm'], 3):>18s}"
            f"{format_value(row['ap03_vs_simulation_gt_cm'], 3):>18s}"
        )

    measured_available = any(
        row['measured_real_reference_m'] is not None
        for row in rows
    )

    lines.extend([
        '',
        'INDEPENDENT REAL-WORLD REFERENCE DISTANCES',
        '-' * 160,
    ])

    if measured_available:
        lines.append(
            f"{'Camera pair':29s}"
            f"{'Measured [m]':>15s}"
            f"{'AP01 err [cm]':>17s}"
            f"{'AP02 err [cm]':>17s}"
            f"{'AP03 err [cm]':>17s}"
        )
        for row in rows:
            pair = f"{row['camera_a']} - {row['camera_b']}"
            lines.append(
                f"{pair:29s}"
                f"{format_value(row['measured_real_reference_m']):>15s}"
                f"{format_value(row['ap01_vs_measured_real_cm'], 3):>17s}"
                f"{format_value(row['ap02_vs_measured_real_cm'], 3):>17s}"
                f"{format_value(row['ap03_vs_measured_real_cm'], 3):>17s}"
            )
    else:
        lines.append(
            'No independent real-world pair distances entered; real accuracy remains unknown.'
        )

    lines.extend([
        '',
        'CROSS-METHOD DISTANCE DISAGREEMENT',
        '-' * 160,
        (
            f"{'Camera pair':29s}"
            f"{'|AP01-AP02| [cm]':>21s}"
            f"{'|AP01-AP03| [cm]':>21s}"
            f"{'|AP02-AP03| [cm]':>21s}"
        ),
    ])

    for row in rows:
        pair = f"{row['camera_a']} - {row['camera_b']}"
        lines.append(
            f"{pair:29s}"
            f"{format_value(row['ap01_vs_ap02_cm'], 3):>21s}"
            f"{format_value(row['ap01_vs_ap03_cm'], 3):>21s}"
            f"{format_value(row['ap02_vs_ap03_cm'], 3):>21s}"
        )

    lines.extend(['', 'DIAGNOSTICS', '-' * 160])
    for method in METHODS:
        lines.append(method)
        lines.extend(diagnostic_lines(method, diagnostics[method]))
        lines.append('')

    lines.extend([
        'INTERPRETATION',
        '-' * 160,
        '- Pairwise distances evaluate translation and metric scale, not full camera orientation.',
        '- Gazebo simulation distances are shown only as a layout plausibility reference.',
        '- Differences to simulation GT are not real-world calibration errors.',
        '- Real-world accuracy requires independent physical measurements or trusted mounting transforms.',
        '- Cross-method agreement is a consistency diagnostic, not ground-truth accuracy.',
        '- A low reprojection error alone does not prove correct global rig geometry.',
        '- Partial or failed camera registration is retained as a method result, not silently discarded.',
        '',
        f'Pairwise CSV: {pairwise_csv}',
        f'Method-status CSV: {status_csv}',
        f'Simulation-reference JSON: {simulation_reference_path}',
        f'Measured-real-reference template: {measured_reference_path}',
        '',
    ])

    report_path = final_root / 'REAL_DATA_ALL_METHODS.txt'
    report_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(report_path.read_text(encoding='utf-8'))


if __name__ == '__main__':
    main()
