Categories
===========

Object Detection
~~~~~~~~~

For object detection, 10 classes are evalued, they are:
::

    1: pedestrian
    2: rider
    3: car
    4: truck
    5: bus
    6: train
    7: motorcycle
    8: bicycle
    9: traffic light
    10: traffic sign

Note that, the field `category_id` range from **1** instead of 0.

Instance Segmentation, Box Tracking, Segmentation Tracking
~~~~~~~~~

For instance segmentation, multi object tracking (box tracking) and multi object tracking and segmentation (segmentation tracking),
only the first **8** classes are used and evaluated.

Semantic Segmentation
~~~~~~~~~

Meanwhile, for the semantic segmentation task, 19 classes are evaluated, they are:
::

    0: road 
    1: sidewalk
    2: building
    3: wall
    4: fence
    5: pole
    6: traffic light
    7: traffic sign
    8: vegetation
    9: terrain
    10: sky
    11: person
    12: rider
    13: car
    14: truck
    15: bus
    16: train
    17: motorcycle
    18: bicycle

`category_id` ranges from **0** for the semantic segmentation task.
**255** is used for "unknown" category, and will not be evaluated.
