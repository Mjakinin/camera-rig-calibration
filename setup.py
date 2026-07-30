"""Compatibility shim for ROS 2 Humble images with pre-PEP-660 pip/setuptools.

Modern installers use ``pyproject.toml``. Ubuntu 22.04's stock pip otherwise
rejects ``pip install -e .`` before it can read the modern project metadata.
"""

from setuptools import find_packages, setup


setup(
    name="camera-rig-calibration",
    version="0.2.0",
    description="Reproducible command-line camera-rig calibration experiments",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.10,<3.14",
    install_requires=[
        "pydantic>=2.7,<3",
        "PyYAML>=6,<7",
        "rich>=13,<15",
        "typer>=0.12,<1",
    ],
    extras_require={
        "scientific": ["numpy>=1.26,<2.3", "scipy>=1.14,<2"],
        "standalone": ["opencv-contrib-python-headless>=4.5,<5"],
        "dev": ["build>=1.2,<2", "pytest>=7,<9", "pytest-cov>=5,<7"],
    },
    entry_points={"console_scripts": ["rigcal=camera_rig_calibration.cli:main"]},
)
