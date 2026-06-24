#!/usr/bin/env python3
from pathlib import Path
import re

BASE = Path("src/calib_lab/bus_real_data/worlds/bus_real_data_moving_camera.sdf")
OUT_DIR = Path("src/calib_lab/bus_real_data/worlds/ablation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

START = "<!-- ABLATION_LIGHTS_START -->"
END = "<!-- ABLATION_LIGHTS_END -->"


def remove_old_ablation_block(txt: str) -> str:
    return re.sub(
        rf"\s*{re.escape(START)}.*?{re.escape(END)}\s*",
        "\n",
        txt,
        flags=re.DOTALL,
    )


def insert_before_world_end(txt: str, block: str) -> str:
    if "</world>" not in txt:
        raise RuntimeError("No </world> tag found.")
    return txt.replace("</world>", f"\n{START}\n{block}\n{END}\n</world>", 1)


def dim_existing_lights(txt: str) -> str:
    # Dim vorhandene diffuse/specular Werte in bestehenden Light-Blöcken.
    def dim_block(match):
        block = match.group(0)

        if "<diffuse>" in block:
            block = re.sub(
                r"<diffuse>.*?</diffuse>",
                "<diffuse>0.08 0.08 0.08 1</diffuse>",
                block,
                flags=re.DOTALL,
            )
        else:
            block = block.replace(">", ">\n      <diffuse>0.08 0.08 0.08 1</diffuse>", 1)

        if "<specular>" in block:
            block = re.sub(
                r"<specular>.*?</specular>",
                "<specular>0.02 0.02 0.02 1</specular>",
                block,
                flags=re.DOTALL,
            )
        else:
            block = block.replace(">", ">\n      <specular>0.02 0.02 0.02 1</specular>", 1)

        if "<intensity>" in block:
            block = re.sub(
                r"<intensity>.*?</intensity>",
                "<intensity>0.15</intensity>",
                block,
                flags=re.DOTALL,
            )

        return block

    return re.sub(
        r"<light\b.*?</light>",
        dim_block,
        txt,
        flags=re.DOTALL,
    )


SIDE_LIGHT_BLOCK = """
<light type="spot" name="ablation_side_light_left">
  <pose>-4 0 2.2 0 0 0</pose>
  <cast_shadows>true</cast_shadows>
  <diffuse>1.0 0.92 0.78 1</diffuse>
  <specular>0.25 0.25 0.25 1</specular>
  <attenuation>
    <range>14</range>
    <constant>0.6</constant>
    <linear>0.04</linear>
    <quadratic>0.005</quadratic>
  </attenuation>
  <direction>1 0 -0.15</direction>
  <spot>
    <inner_angle>0.45</inner_angle>
    <outer_angle>1.10</outer_angle>
    <falloff>0.8</falloff>
  </spot>
</light>

<light type="point" name="ablation_inside_fill_light">
  <pose>0 0 2.0 0 0 0</pose>
  <cast_shadows>false</cast_shadows>
  <diffuse>0.55 0.55 0.60 1</diffuse>
  <specular>0.05 0.05 0.05 1</specular>
  <attenuation>
    <range>7</range>
    <constant>0.9</constant>
    <linear>0.08</linear>
    <quadratic>0.01</quadratic>
  </attenuation>
</light>
"""


STRONG_LIGHT_BLOCK = """
<light type="spot" name="ablation_strong_glare_light">
  <pose>-5 -1 2.4 0 0 0</pose>
  <cast_shadows>true</cast_shadows>
  <diffuse>1.0 0.95 0.80 1</diffuse>
  <specular>0.8 0.8 0.8 1</specular>
  <attenuation>
    <range>20</range>
    <constant>0.35</constant>
    <linear>0.015</linear>
    <quadratic>0.001</quadratic>
  </attenuation>
  <direction>1 0.2 -0.1</direction>
  <spot>
    <inner_angle>0.30</inner_angle>
    <outer_angle>0.85</outer_angle>
    <falloff>0.6</falloff>
  </spot>
</light>

<light type="point" name="ablation_strong_fill_light">
  <pose>0 0 2.4 0 0 0</pose>
  <cast_shadows>false</cast_shadows>
  <diffuse>0.9 0.9 0.85 1</diffuse>
  <specular>0.25 0.25 0.25 1</specular>
  <attenuation>
    <range>10</range>
    <constant>0.55</constant>
    <linear>0.04</linear>
    <quadratic>0.006</quadratic>
  </attenuation>
</light>
"""


def write_variant(name: str, txt: str, block: str = ""):
    txt = remove_old_ablation_block(txt)
    if block:
        txt = insert_before_world_end(txt, block)
    out = OUT_DIR / f"bus_real_data_moving_camera_{name}.sdf"
    out.write_text(txt)
    print(f"Wrote: {out}")


def main():
    base = BASE.read_text()

    # 1. Baseline copy
    write_variant("baseline_light", base)

    # 2. Low light: vorhandene Lights abdunkeln
    low = dim_existing_lights(base)
    write_variant("low_light", low)

    # 3. Side illumination
    write_variant("side_light", base, SIDE_LIGHT_BLOCK)

    # 4. Strong/glare illumination
    write_variant("strong_light", base, STRONG_LIGHT_BLOCK)


if __name__ == "__main__":
    main()
