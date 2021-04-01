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

Labels
~~~~~~~

Annotations of road objects, lanes, and drivable areas in JSON format released
in 2018. Details at Github repo. We revised the detection annotations in 2020
and released them as Detection 2020 Labels in the list. You are recommended to
use the new labels. This detection anntoation set is kept for comparison with
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

Segmentation
~~~~~~~~~~~~~

Full-frame semantic segmentation maps. The corresponding images are in the same
folder.

+------+----------------------------------+
| Size | 1.2GB                            |
+------+----------------------------------+
| md5  | 0baeaf8bed8f1a7feb1e8755bcce7169 |
+------+----------------------------------+

MOT 2020 Labels
~~~~~~~~~~~~~~~~

Multi-object bounding box tracking training and validation labels released in
2020

+------+----------------------------------+
| Size | 103MB                            |
+------+----------------------------------+
| md5  | 00398674e62d05206b0847be3f638d74 |
+------+----------------------------------+

MOT 2020 Images
~~~~~~~~~~~~~~~~

Multi-object bounding box tracking videos in frames released in 2020

Detection 2020 Labels
~~~~~~~~~~~~~~~~~~~~~~

Multi-object detection validation and testing labels released in 2020. This is
for the same set of images in the previous key frame annotation. However, this
annotation went through the additional quality check. The original detection set
is deprecated.

+------+----------------------------------+
| Size | 51MB                             |
+------+----------------------------------+
| md5  | 6e761be56b6c2f6cf6a0c3a38eb5da0e |
+------+----------------------------------+

MOTS 2020 Labels
~~~~~~~~~~~~~~~~~

Multi-object tracking aand segmentation training and validation labels released in 2020


+------+----------------------------------+
| Size | 465.5MB                          |
+------+----------------------------------+
| md5  | c6968ad8b86c31f49dd66efde97f53f0 |
+------+----------------------------------+

MOTS 2020 Images
~~~~~~~~~~~~~~~~~

Multi-object tracking and segmentation videos in frames released in 2020