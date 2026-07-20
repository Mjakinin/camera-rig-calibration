#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DEFAULT_INPUT = REPO / "results/bus_real_data/02_ref_marker_graph_ba/02_aruco_observations/ap02_all_aruco_observations.csv"
DEFAULT_OUT = REPO / "results/bus_real_data