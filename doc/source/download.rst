Data Download
---------------

The BDD100K data and annotations can be obtained at
https://bdd-data.berkeley.edu/. You can simply log in and download the data in
your browser after agreeing to :ref:`BDD100K license<license>`. You will see a
list of downloading buttons:

.. figure:: ../media/images/download_buttons.png
   :alt: Downloading buttons

The files behinds the buttons are described below.

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
The 100K videos broken into 100 parts

Info
~~~~

The GPS/IMU information recorded along with the videos

+------+----------------------------------+
| Size | 3.9GB                            |
+------+----------------------------------+
| md5  | 043811ff34b2fca6d50f37d263a65c93 |
+------+----------------------------------+

Images
~~~~~~~

It has two subfolders. 1) 100K labeled key frame images extracted from the
videos at 10th second 2) 10K key frames for full-frame semantic segmentation.

+------+----------------------------------+
| Size | 6.5GB                            |
+------+----------------------------------+
| md5  | b538e3731a132e28dae37f18c442c51e |
+------+----------------------------------+

:: 

    - bdd100k
        - images
            - 100k
                - train
                - val
                - test
            - 10k
                - train
                - val
                - test

Labels
~~~~~~~

Annotations of road objects, lanes, and drivable areas in JSON format released
in 2018. Details at Github repo. We revised the detection annotations in 2020
and released them as Detection 2020 Labels in the list. You are recommended to
use the new labels. This detection annotation set is kept for comparison with
legacy results.

+------+----------------------------------+
| Size | 107MB                            |
+------+----------------------------------+
| md5  | e21be3e7d6a07ee439faf61e769667e4 |
+------+----------------------------------+

Drivable Maps
~~~~~~~~~~~~~~

Segmentation maps of Drivable areas.

+------+----------------------------------+
| Size | 661MB                            |
+------+----------------------------------+
| md5  | 1bd70019468a81572c70d374751eb9e2 |
+------+----------------------------------+

Semantic Segmentation
~~~~~~~~~~~~~~~~~~~~~~

Bitmasks, colormaps and original json files for semantic segmentation.

+------+----------------------------------+
| Size | 370MB                            |
+------+----------------------------------+
| md5  | cb9b2ececcdb56f1ca3ae54d4ed883bd |
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


Instance Segmentation
~~~~~~~~~~~~~~~~~~~~~~

Bitmasks, colormaps and original json files for instance segmentation.

+------+----------------------------------+
| Size | 96MB                             |
+------+----------------------------------+
| md5  | e29d82458a6c487b444c40ab82426d03 |
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


MOT 2020 Labels
~~~~~~~~~~~~~~~~

Multi-object bounding box tracking training and validation labels released in
2020

+------+----------------------------------+
| Size | 104MB                            |
+------+----------------------------------+
| md5  | 931813bcec4e0483f57b443c4cbd6c5c |
+------+----------------------------------+

:: 

    - bdd100k
        - labels
            - box_track_20
                - train
                - val


MOT 2020 Images
~~~~~~~~~~~~~~~~

Multi-object bounding box tracking videos in frames released in 2020

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


+------+----------------------------------+
| Size | 370MB                            |
+------+----------------------------------+
| md5  | 13e15f1f3728a9a3f27a3f4e5e2a5885 |
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

MOTS 2020 Images
~~~~~~~~~~~~~~~~~

Multi-object tracking and segmentation videos in frames released in 2020

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
