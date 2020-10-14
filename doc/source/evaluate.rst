Evaluation
===========


Detection
~~~~~~~~~



Multiple Object Tracking
~~~~~~~~~~~~~~~~~~~~~~~~

Submission format
^^^^^^^^^^^^^^^^^^^^^^

To evaluate your algorithms on BDD100K multiple object tracking benchmark,
please first format your results in `Scalabel Format <www.scalabel.ai/doc/format.html>`_ 
with two save/submit options:

- A zip file or a folder that contains JSON files of each video.

- A zip file or a file that contains a JSON file of the entire evaluation set.

Evaluation method
^^^^^^^^^^^^^^^^^^^^^^

You can evaluate your algorithms with public annotations by running
::

    python -m bdd100k.eval.run -t mot -g ${gt_file} -r ${res_file} 

To obain results on val/test phase, submit your result files at `BDD100K 2D Multiple Object Tracking Challenge <TODO>`_.



Evaluation Metrics
^^^^^^^^^^^^^^^^^^^^^^


We employ mean Multiple Object Tracking Accuracy (mMOTA, mean of MOTA of the 8 categories)
as our primary evaluation metric for ranking. 
We also employ mean ID F1 score (mIDF1) to highlight the performance 
of tracking consistency that is crucial for object tracking.
All metrics are detailed below.
Note that the overall performance are measured for all objects without considering the category if not mentioned.

- mMOTA (%): mean Multiple Object Tracking Accuracy across all 8 categories.

- mIDF1 (%): mean ID F1 score across all 8 categories.

- mMOTP (%): mean Multiple Object Tracking Precision across all 8 categories.

- MOTA (%): Multiple Object Tracking Accuracy [1]. It measures the errors from false positives, false negatives and identity switches.

- IDF1 (%): ID F1 score [2]. The ratio of correctly identified detections over the average number of ground-truths and detections.

- MOTP (%): Multiple Object Tracking Precision [1]. It measures the misalignments between ground-truths and detections.

- FP: Number of False Positives [1].
 
- FN: Number of False Negatives [1].

- IDSw: Number of Identity Switches [1]. An identity switch is counted when a ground-truth object is matched with a identity that is different from the last known assigned identity.

- MT: Number of Mostly Tracked identities. At least 80 percent of their lifespan are tracked.

- PT: Number of Partially Tracked identities. At least 20 percent and less than 80 percent of their lifespan are tracked.

- ML: Number of Mostly Lost identities. Less of 20 percent of their lifespan are tracked.

- FM: Number of FragMentations. Total number of switches from tracked to not tracked detections.


[1] `Bernardin, Keni, and Rainer Stiefelhagen. "Evaluating multiple object tracking performance: the CLEAR MOT metrics." EURASIP Journal on Image and Video Processing 2008 (2008): 1-10. <https://link.springer.com/article/10.1155/2008/246309>`_

[2] `Ristani, Ergys, et al. "Performance measures and a data set for multi-target, multi-camera tracking." European Conference on Computer Vision. Springer, Cham, 2016. <https://arxiv.org/abs/1609.01775>`_



Super-category
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In addition to the evaluation of all 8 classes, 
we also evaluate results for 3 super-categories specified below.
The super-category evaluation results are provided only for the purpose of reference.
::
    "HUMAN":   ["pedestrian", "rider"],
    "VEHICLE": ["car", "bus", "truck", "train"],
    "BIKE":    ["motorcycle", "bicycle"]


Ignore regions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
After the bounding box matching proccess in evaluation, we ignore all detected false-positive boxes that have >50% overlap with the crowd region (ground-truth boxes with the "Crowd" attribute).

We also ignore object regions that are annotated as 3 distracting classes ("other person", "trailer", and "other vehicle") by the same strategy of crowd regions for simplicity. 


Pre-training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
It is a fair game to pre-train your network with ImageNet or COCO, 
but if other datasets are used, please note in the submission description. 
We will rank the methods without using external datasets except ImageNet and COCO.

.. Jiangmiao: online or offline constrains??
.. Jiangmiao: ranking metric by mMOTA? KITTI said no ranking metric. 


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