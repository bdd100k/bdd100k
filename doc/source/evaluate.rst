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

The entire result struct array is stored as a single JSON file (save via gason in Matlab or json.dump in Python), which consists a list of frame objects with the fields below.
::

    - name: string
    - video_name: string
    - index: int (frame index in this video)
    - labels []:
        - id: int32
        - category: string
        - poly2d []:
                - vertices: [][]float (list of 2-tuples [x, y])
                - types: string (each character corresponds to the type of the vertex with the same index in vertices. ‘L’ for vertex and ‘C’ for control point of a bezier curve.
                - closed: boolean (closed for polygon and otherwise for path)

Note that, the "id" of the same object through an video should be the same.
Candidates for `category` are `['person', 'rider', 'car', 'bus', 'truck', 'bike', 'motor', 'train']`.
