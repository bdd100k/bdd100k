"""Package setup."""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bdd100k",  # Replace with your own username
    version="1.0.0",
    author="Fisher Yu",
    author_email="i@yf.io",
    description="BDD100K Dataset Toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://www.bdd100k.com/",
    project_urls={
        "Documentation": "https://doc.bdd100k.com/",
        "Source": "https://github.com/bdd100k/bdd100k",
        "Tracker": "https://github.com/bdd100k/bdd100k/issues",
    },
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "gmplot",
        "joblib",
        "matplotlib",
        "motmetrics",
        "numpy",
        "pandas",
        "pillow",
        "pycocotools",
        "scalabel @ git+https://github.com/scalabel/scalabel.git",
        "scikit-image",
        "toml",
        "tqdm",
        "tabulate",
        "tqdm",
    ],
    package_data={
        "bdd100k": [
            "configs/box_track.toml",
            "configs/det.toml",
            "configs/drivable.toml",
            "configs/ins_seg.toml",
            "configs/lane_mark.toml",
            "configs/pan_seg.toml",
            "configs/pose.toml",
            "configs/seg_track.toml",
            "configs/sem_seg.toml",
            "py.typed",
        ]
    },
    include_package_data=True,
)
