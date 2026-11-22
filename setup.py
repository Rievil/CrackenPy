# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 17:03:20 2024

@author: dvorr
"""


from setuptools import Extension, setup, find_packages
from pathlib import Path

# README is shown as the project description on the PyPI page
long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")


setup(
    name="crackenpy",
    version="0.1.87",
    description="Library for detection of cracks on test specimens of building materials",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Rievil/CrackenPy",
    project_urls={
        "Paper": "https://doi.org/10.1016/j.measurement.2026.123831",
        "Model": "https://huggingface.co/rievil/crackenpy",
        "Dataset": "https://huggingface.co/datasets/rievil/crackenpy_dataset",
    },
    author="Richard Dvorak, Vlastimil Bilek, Barbara Kucharczikova, Rostislav Krc",
    author_email="richard.dvorak@vutbr.cz",
    packages=find_packages(where="src"),
    install_requires=[
        "torch>=2.0.0",
        "pandas>=0.23.3",
        "numpy>=1.26.0",
        "matplotlib>=2.2.0",
        "gdown>=5.0.0",
        "sknw>=0.15",
        "tqdm>=4.66.0",
        "scikit-image>=0.21.0",
        "scikit-learn>=0.19.1",
        "wand>=0.6.13",
        "segmentation_models_pytorch>=0.3.3",
        "opencv-python>=4.8.0.76",
        "huggingface_hub",
        "torchvision",
        "scipy",
    ],
    package_dir={"": "src"},
    package_data={
        "": ["*.pt"],
        "crackpy_models": [
            "resnext101_32x8d_N387_C5_30102023",
            "resnext101_32x8d_N387_C5_310124",
        ],
    },
)
