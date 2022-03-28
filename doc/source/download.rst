.. default-role:: code

Data Download
---------------

The BDD100K data and annotations can be obtained at
https://bdd-data.berkeley.edu/. You can simply log in and download the data in
your browser after agreeing to :ref:`BDD100K license<license>`. On the downloading portal, you will see a
list of downloading buttons with the name corresponding to the subsections on this page. 
The files behind the buttons are described below.

Videos
~~~~~~

100K video clips

+------+----------------------------------+
| Size | 1.8TB                            |
+------+----------------------------------+
| md5  | 253d9a2f9d89d2b09d8d93f397aecdd7 |
+------+----------------------------------+


Video Torrent
~~~~~~~~~~~~~

Torrent for the 100K video clips


Video Parts
~~~~~~~~~~~~
The 100K videos broken into 100 parts for easy downloading.

Info
~~~~

The GPS/IMU information recorded along with the videos

+------+----------------------------------+
| Size | 3.9GB                            |
+------+----------------------------------+
| md5  | 043811ff34b2fca6d50f37d263a65c93 |
+------+----------------------------------+


100K Images
~~~~~~~~~~~~

The images in this package are the frames at the 10th second in the videos.
The split of train, validation, and test sets are the same with the whole video set.
They are used for object detection, drivable area, lane marking.

+------+----------------------------------+
| Size | 5.3GB                            |
+------+----------------------------------+
| md5  | 5a0359c86a0b8713adab1eee9a3041cb |
+------+----------------------------------+

:: 

    - bdd100k
        - images
            - 100k
                - train
                - val
                - test

10K Images
~~~~~~~~~~~~

There are 10K images in this package for for semantic segmentation, instance
segmentation and panoptic segmentation. Due to some legacy reasons, not all the
images here have corresponding videos. So it is not a subset of the 100K images,
even though there is a significant overlap.

+------+----------------------------------+
| Size | 1.1GB                            |
+------+----------------------------------+
| md5  | 08f26aecceda982568063d3d5873378e |
+------+----------------------------------+

:: 

    - bdd100k
        - images
            - 10k
                - train
                - val
                - test


Labels
~~~~~~~

Annotations of road object detection in JSON format released
in 2018. The video attributes, including `weather`, `scene`, and `timeofday`,
are also stored in the downloaded json files. We revised the detection annotations in 2020
and released them as Detection 2020 Labels in the list. You are recommended to
use the new labels. This detection annotation set is kept for comparison with
legacy results.

+------+----------------------------------+
| Size | 107MB                            |
+------+----------------------------------+
| md5  | e21be3e7d6a07ee439faf61e769667e4 |
+------+----------------------------------+

Drivable Area
~~~~~~~~~~~~~~

Masks, colormaps, RLEs, and original json files for drivable area.
The mask format is explained at: :ref:`Semantic Segmentation Format <seg mask>`.

+------+----------------------------------+
| Size | 514MB                            |
+------+----------------------------------+
| md5  | 0abc320461200b1d7916f82fdcd64a96 |
+------+----------------------------------+

:: 

    - bdd100k
        - labels
            - drivable
                - masks
                    - train
                    - val
                - colormaps
                    - train
                    - val
                - polygons
                    - drivable_train.json
                    - drivable_val.json
                - rles
                    - drivable_train.json
                    - drivable_val.json


Lane Marking
~~~~~~~~~~~~~~

Masks, colormaps and original json files for lane marking.
The mask format is explained at: :ref:`Lane Marking Format <lane mask>`.

+------+----------------------------------+
| Size | 434MB                            |
+------+----------------------------------+
| md5  | dfe74f9ed6800765a0047414d620a186 |
+------+----------------------------------+

:: 

    - bdd100k
        - labels
            - lane 
                - masks
                    - train
                    - val
                - colormaps
                    - train
                    - val
                - polygons
                    - lane_train.json
                    - lane_val.json


Semantic Segmentation
~~~~~~~~~~~~~~~~~~~~~~

Masks, colormaps, RLEs, and original json files for semantic segmentation.
The mask format is explained at: :ref:`Semantic Segmentation Format <seg mask>`.

+------+----------------------------------+
| Size | 419MB                            |
+------+----------------------------------+
| md5  | 9a2968dde3345eeb689cffb1e26f9c78 |
+------+----------------------------------+

:: 

    - bdd100k
        - labels
            - sem_seg 
                - masks
                    - train
                    - val
                - colormaps
                    - train
                    - val
                - polygons
                    - sem_seg_train.json
                    - sem_seg_val.json
                - rles
                    - sem_seg_train.json
                    - sem_seg_val.json


Instance Segmentation
~~~~~~~~~~~~~~~~~~~~~~

Masks, colormaps, RLEs, and original json files for instance segmentation.
The bitmask format is explained at: :ref:`Instance Segmentation Format <bitmask>`.

+------+----------------------------------+
| Size | 111MB                            |
+------+----------------------------------+
| md5  | 651b41f229d7327d8c4af97772de4390 |
+------+----------------------------------+


:: 

    - bdd100k
        - labels
            - ins_seg
                - bitmasks
                    - train
                    - val
                - colormaps
                    - train
                    - val
                - polygons
                    - ins_seg_train.json
                    - ins_seg_val.json
                - rles
                    - ins_seg_train.json
                    - ins_seg_val.json


Panoptic Segmentation
~~~~~~~~~~~~~~~~~~~~~~

Bitmasks, colormaps and original json files for panoptic segmentation.
The bitmask format is explained at: :ref:`Panoptic Segmentation Format <bitmask>`.

+------+----------------------------------+
| Size | 363MB                            |
+------+----------------------------------+
| md5  | fc37642ae024ffb223182ef01238d007 |
+------+----------------------------------+


:: 

    - bdd100k
        - labels
            - pan_seg
                - bitmasks
                    - train
                    - val
                - colormaps
                    - train
                    - val
                - polygons
                    - pan_seg_train.json
                    - pan_seg_val.json


MOT 2020 Labels
~~~~~~~~~~~~~~~~

Multi-object bounding box tracking training and validation labels released in 2020.
This is a subset of the 100K videos, but the videos are resampled to 5Hz from 30Hz. The labels are in `Scalabel Format
<https://doc.scalabel.ai/format.html>`_. The same object in each video has the same 
label id but objects across videos are always distinct even if they have the same id.

+------+----------------------------------+
| Size | 115MB                            |
+------+----------------------------------+
| md5  | 6be40e0ca56a83ddeba2ed6bff50f9e6 |
+------+----------------------------------+

:: 

    - bdd100k
        - labels
            - box_track_20
                - train
                - val


MOT 2020 Images
~~~~~~~~~~~~~~~~

Multi-object bounding box tracking videos in frames released in 2020.
The videos are a subset of the 100K videos, but they are resampled to 5Hz from 30Hz.


:: 

    - bdd100k
        - images
            - track
                - train
                - val
                - test


Detection 2020 Labels
~~~~~~~~~~~~~~~~~~~~~~

Multi-object detection validation and testing labels released in 2020. This is
for the same set of images in the previous key frame annotation. However, this
annotation went through the additional quality check. The original detection set
is deprecated.

+------+----------------------------------+
| Size | 53MB                             |
+------+----------------------------------+
| md5  | b86a3e1b7edbcad421b7dad2b3987c94 |
+------+----------------------------------+

:: 

    - bdd100k
        - labels
            - det_20
                - det_train.json
                - det_val.json

MOTS 2020 Labels
~~~~~~~~~~~~~~~~~

Multi-object tracking and segmentation training and validation labels released in 2020
The bitmask format is explained at: :ref:`Instance Segmentation Format <bitmask>`.


+------+----------------------------------+
| Size | 452MB                            |
+------+----------------------------------+
| md5  | 8822a8b72c2c6719f4573bc4d7077020 |
+------+----------------------------------+

:: 

    - bdd100k
        - labels
            - seg_track_20
                - bitmasks
                    - train
                    - val
                - colormaps
                    - train
                    - val
                - polygons
                    - train
                    - val
                - rles
                    - train
                    - val

MOTS 2020 Images
~~~~~~~~~~~~~~~~~

Multi-object tracking and segmentation videos in frames released in 2020. This is a subset of `MOT 2020 Images`_.

+------+----------------------------------+
| Size | 5.4GB                            |
+------+----------------------------------+
| md5  | 7c52a52f3c9cc880c91b264870a1d4bb |
+------+----------------------------------+

:: 

    - bdd100k
        - images
            - seg_track_20
                - train
                - val
                - test

Pose Estimation Labels
~~~~~~~~~~~~~~~~~~~~~~

Pose estimation training and validation labels.

+------+----------------------------------+
| Size | 17MB                             |
+------+----------------------------------+
| md5  | 2e8738d3fd0ac432e64d9a72df2f7aa4 |
+------+----------------------------------+

:: 

    - bdd100k
        - labels
            - pose_21
                - pose_train.json
                - pose_val.json
