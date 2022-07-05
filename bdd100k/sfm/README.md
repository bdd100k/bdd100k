# COLMAP reconstruction tool for any Sequences (BDD100K mainly)

This repo focuses on generating metric-scale sparse and dense reconstruction using [COLMAP](https://colmap.github.io/tutorial.html) for one or several sequences, especially for BDD100K sequences.  

## Tutorial

### Structure from Motion (SfM)

<img src="https://user-images.githubusercontent.com/49795181/177323090-d3ff5028-481b-4080-8532-fcf42cfa54aa.png" width="800">

### Multi-View Stereo (MVS)

<img src="https://user-images.githubusercontent.com/49795181/177323054-6311d21e-3f05-45eb-a63a-3c9cfab199ce.png" width="800">


## Installation

### Hardware

* OS: Ubuntu >18.04
* NVIDIA GPU with **CUDA>=10.1** (tested with 1 RTX1080Ti or 2080Ti)
    * GPU is only required for MVS. You can specify ```--no-gpu``` for SfM if no GPU on your machine

### Software
- Install [COLMAP](https://github.com/kevinhangoat/colmap)
    * This is an unofficial implementation of COLMAP with GPS information as a reference in Bundle adjustment. The official colmap currently does not support such feature, but the discussion about adding it can be found [here](https://github.com/colmap/colmap/pull/1409) contributed by [ferreram](https://github.com/ferreram/colmap/tree/sfm_gps_ba_aligner_4_clem). 
- Clone this repo and install libraries 
```
git clone --recursive git@github.com:bdd100k/bdd100k.git
git checkout sfm-module
conda create -n sfm python=3.7
cd bdd100k
pip install -r bdd100k/sfm/requirements.txt
```

## Demo

### KITTI

Run (example)
```
python -m bdd100k.sfm.run_colmap \
    -i #input_path \
    -o #output_path \
    --info-path #info_path \
    # --colmap-path # location_colmap \
    --matcher-method spatial -j sparse \
```

### BDD100K
Run (example)
```
python -m bdd100k.sfm.run_colmap \
    -i #input_path \
    -o #output_path \
    --info-path #info_path_1 \
    --info-path #info_path_2 \
    # --colmap-path # location_colmap \
    --matcher-method spatial -j sparse \
```

### Visualization 
[<img src="https://user-images.githubusercontent.com/49795181/177338103-be278f72-c5a6-44d2-9e1b-6782adfbb411.png" width="500">](https://www.youtube.com/watch?v=Na3kHOR5_eM)

## TODO

- [ ] Demo details
- [ ] Visualization Demo
- [ ] Post process
- [ ] Example dataset