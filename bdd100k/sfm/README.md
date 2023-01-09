# Sequence Reconstruction and Depth Post-processing

We provide the code for our metric-scale reconstructions of BDD100K using [COLMAP](https://colmap.github.io/tutorial.html). The pipeline is tested with BDD100K, KITTI and nuScenes sequences. We also provide our approach to filter out noise in the raw depth maps from COLMAP.  

## 1. Introduction

[COLMAP](https://colmap.github.io/tutorial.html) is a state-of-art tool for Structure-from-Motion (SfM) and Multi-View-Stereo (MVS). We utilize COLMAP's interfaces to build our pipeline, which is summarized in the two figures below.

<img src="https://user-images.githubusercontent.com/49795181/177323090-d3ff5028-481b-4080-8532-fcf42cfa54aa.png" width="800">


<img src="https://user-images.githubusercontent.com/49795181/177323054-6311d21e-3f05-45eb-a63a-3c9cfab199ce.png" width="800">

The highlights of our COLMAP implementation are: 
- GPS information is utilized during both spatial feature matcher and bundle adjustment. 
- Inconsistent pose estimates are automatically removed from sparse reconstruction.
- The depth maps are filtered before fusing them into a semi-dense point cloud.

Since the raw depth maps are usually nosiy, we post-process them by applying several filters, including:  
- Range filtering
- Use Panoptic segmentation to remove dynamic objects
- Use a RANSAC-based plane fitting algorithm for ground pixels to remove outliers 

## 2. Installation

### 2.1 Hardware

* OS: Ubuntu >18.04
* NVIDIA GPU with **CUDA>=10.1** (tested with 1 RTX1080Ti or 2080Ti)
    * GPU is only required for MVS. You can specify ```--no-gpu``` for SfM if there is no GPU on your machine

### 2.2 Software
- Install [COLMAP](https://github.com/kevinhangoat/colmap)
    * This implementation of COLMAP can use GPS information as a reference in Bundle adjustment. The official colmap currently does not support this feature, but the discussion about adding it can be found [here](https://github.com/colmap/colmap/pull/1409) contributed by [ferreram](https://github.com/ferreram/colmap/tree/sfm_gps_ba_aligner_4_clem). 
- Clone this repo and install the required libraries
```
git clone --recursive git@github.com:bdd100k/bdd100k.git
cd bdd100k
python setup.py develop
pip install -r requirements.txt
```
- (Optional) To use the depth post-procesing tool, also install the additional libraries:
```
pip install -r bdd100k/sfm/requirements.txt
```
## 3. Example Demo

### 3.1 COLMAP reconstruction

Assuming you want to reconstruct a BDD100K sequence and the data is stored in the following structure:

    sequence/path/
    +── images
    │   +── image1.jpg
    │   +── image2.jpg
    │   +── ...
    │   +── imageN.jpg
    +── info.json  // a .json file containing GPS information 


The command to run the whole pipeline is:

```
python -m bdd100k.sfm.run_colmap \
    --job all \
    --image-path sequence/path/images \
    --output-path sequence/path/ \
    --info-path sequence/path/info.json \
```

If you only want to run a part of the colmap pipeline run the following commands one by one:

Feature extraction and matching:
```bash
python -m bdd100k.sfm.run_colmap --job feature --image-path sequence/path/images --output-path sequence/path/ --info-path sequence/path/info.json
```

Structure-from-Motion (Make sure you specify the path to the modified version of colmap):
```bash
python -m bdd100k.sfm.run_colmap --job mapper --image-path sequence/path/images --output-path sequence/path/ --info-path sequence/path/info.json --colmap-path path/to/colmap/app
```

Orientation Aligner:
```bash
python -m bdd100k.sfm.run_colmap --job orien_aligner --image-path sequence/path/images --output-path sequence/path/ --info-path sequence/path/info.json
```

Multi-View Stereo:
```bash
python -m bdd100k.sfm.run_colmap --job dense --image-path sequence/path/images --output-path sequence/path/
```

Fusion (A path of image masks will be automatically created for fusion using the panoptic mask, unless a mask-path is specified):
```bash
python -m bdd100k.sfm.run_colmap --job stereo_fusion --image-path sequence/path/images --output-path sequence/path/
```
Check run_colmap.py for more features and functions.


### 3.2 Depth Map Post-processing

After the COLMAP reconstruction, the sequence path becomes:

    sequence/path/
    +── images
    │   +── image1.jpg
    │   +── image2.jpg
    │   +── ...
    │   +── imageN.jpg
    +── sparse
    +── dense
    │   +── 0_dense
    │   +── +── ...
    +── pan_mask   // Panoptic Segmenatation
    │   +── pan_seg.json
    +── info.json 

To obtain the panoptic segmentation, you can either download the provided results or use a pre-trained model in our Model Zoo [bdd100k-models](https://github.com/SysCV/bdd100k-models/tree/main/pan_seg).

Run the following commad for depth map post processing.
```
python3 -m bdd100k.sfm.run_postprocess \
    --target_path sequence/path/ \
    --dense_path sequence/path/dense/0_dense
```
The results will be saved to the folder `0_dense/depth_maps_processed`.

### 4. Visualization Example
[<img src="https://user-images.githubusercontent.com/49795181/177338103-be278f72-c5a6-44d2-9e1b-6782adfbb411.png" width="500">](https://www.youtube.com/watch?v=Na3kHOR5_eM)
