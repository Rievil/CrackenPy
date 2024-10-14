# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 17:03:20 2024

@author: dvorr
"""


from setuptools import Extension, setup, find_packages


setup(
    name="crackpy",
    version="0.1.53",
    packages=find_packages(where="src"),
    install_requires=[
        "torch>=2.0.0",
        "pandas>=0.23.3",
        "numpy>=2.0.0",
        "matplotlib>=2.2.0",
        "gdown>=5.0.0",
        "sknw>=0.15",
        "tqdm>=4.66.0",
        "scikit-image>=0.21.0",
        "scikit-learn>=0.19.1",
        "wand>=0.6.13",
        "segmentation_models_pytorch>=0.3.3",
        "opencv-python>=4.8.0.76",
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
