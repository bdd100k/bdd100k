Evaluation
===========


Detection
~~~~~~~~~



Multiple Object Tracking
~~~~~~~~~~~~~~~~~~~~~~~~


Segmentation Tracking
~~~~~~~~~~~~~~~~~~~~~~~~

We use the following 3 metrics to evaluate the performance of segmetation tracking:

+--------+------------------------------------------------------+
| Metric | Description                                          |
+========+======================================================+
| AP     | instance segmentation AP                             |
+--------+------------------------------------------------------+
| MOTSA  | multi-object tracking and segmentation accuracy      |
+--------+------------------------------------------------------+
| MOTSP  | multi-object tracking and segmentation precision     |
+--------+------------------------------------------------------+
| sMOTSA | soft multi-object tracking and segmentation accuracy |
+--------+------------------------------------------------------+
| IDSW   | identity switch                                      |
+--------+------------------------------------------------------+
| IDF1   | identification F1                                    |
+--------+------------------------------------------------------+

Submission format
^^^^^^^^^^^^^^^^^

The entire result struct array is stored as a single JSON file (save via gason in Matlab or json.dump in Python).
::

    [  
        {  
            "image_id": int,
            "instance_id": int,
            "category_id": int,
            "segmentation": polygon or RLE,
        }
    ]

Note that, the "instance_id" of the same object through an video should be the same. Moreover, it should be unique inside an video, but not need to be unique between objects belonging to different videos.
Candidates for `category` are `['person', 'rider', 'car', 'bus', 'truck', 'bike', 'motor', 'train']`.
