Evaluation
===========


Detection
~~~~~~~~~

Submission format
^^^^^^^^^^^^^^^^^^^^^^

To evaluate your algorithms on the BDD100K detection benchmark, you may prepare
your prediction results using the `Scalabel Format <https://doc.scalabel.ai/format.html>`_.
Specifically, these fields are required:
::

    - name: str
    - labels []:
        - id: str, not used here, but required for loading
        - category: str
        - score: float
        - box2d:
            - x1: float
            - y1: float
            - x2: float
            - y2: float

When you submit your results, save your results in a JSON file and then compress it into a zip file.

Run Evaluation on Your Own
^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can evaluate your algorithm with public annotations by running 
::
    
    python3 -m bdd100k.eval.run -t det -g ${gt_file} -r ${res_file} 

- ``gt_file``: ground truth file in JSON, either in Scalabel format or COCO format. If using COCO format, add a flag `--ann-format coco`
- ``res_file``: prediction results file in JSON, format as described above.

Other options.
- If you want to evaluate the detection performance on the BDD100K MOT set, 
you can add a flag `--mode track`. 
- You can also specify the output directory to save the evaluation results by updating `--out-dir ${out_dir}`.

Evaluation Metrics
^^^^^^^^^^^^^^^^^^^^^^

Similar to COCO evaluation, we report 12 scores as 
"AP", "AP_50", "AP_75", "AP_small", "AP_medium", "AP_large", "AR_max_1", "AR_max_10",
"AR_max_100", "AR_small", "AR_medium", "AR_large" across all the classes. 



Instance Segmentation
~~~~~~~~~~~~~~~~~~~~~~~~

We use the same metrics set as DET above. The only difference lies in the computation of distance matrixes.
Concretely, in DET, it is computed using box IoU. While for InsSeg, the mask IoU is used.

Submission format
^^^^^^^^^^^^^^^^^^^^^^

To evaluate your algorithms on the BDD100K detection benchmark, you may prepare predictions in bitmask format,
which is illustrated in :ref:`Instance Segmentaiton Bitmask <bitmask>`.
Moreover, a score file is needed, with the following format:
::

    - name: str, name of the input image,
    - labels []:
        - id: str, not used here, but required for loading
        - index: int, in range [1, 65535], the index in B and A channel
        - score: float, confidence score of the prediction

- `index`: the value correspondence to the "ann_id" stored in B and A channels.

To be evaluated on the Codalab server, the submission file needs to be a zipped nested folder with the following structure:
::

    - score.json
    - bitmasks
        - xxx.png
        - yyy.png
        ...

Run Evaluation on Your Own
^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can evaluate your algorithm with public annotations by running 
::
    
    python3 -m bdd100k.eval.run -t ins_seg -g ${gt_path} -r ${res_path} --score-file ${res_score_file} 

- `gt_path`: the path to ground-truch bitmask images folder.
- `res_path`: the path to the results bitmask images folder.
- `res_score_file`: the json file with the confidence scores.



Panoptic Segmentation
~~~~~~~~~~~~~~~~~~~~~~~~

We use the same metrics as COCO panoptic segmentation.
PQ, RQ and SQ are computed for things, stuffs and all.

Submission format
^^^^^^^^^^^^^^^^^^^^^^

To evaluate your algorithms on the BDD100K detection benchmark, you may prepare predictions in bitmask format,
which is illustrated in :ref:`Panoptic Segmentaiton Bitmask <bitmask>`.
To be evaluated on the Codalab server, the submission file needs to be a zipped folder.

[1] `Kirillov, A., He, K., Girshick, R., Rother, C., & Dollár, P. (2019). Panoptic segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 9404-9413). <https://arxiv.org/abs/1801.00868>`_

Run Evaluation on Your Own
^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can evaluate your algorithm with public annotations by running 
::
    
    python3 -m bdd100k.eval.run -t pan_seg -g ${gt_path} -r ${res_path}

- `gt_path`: the path to ground-truch bitmask images folder.
- `res_path`: the path to the results bitmask images folder.



Semantic Segmentation
~~~~~~~~~~~~~~~~~~~~~~~~

We assess the performance using the standaard Jaccard Index, commonly known as mean-IoU.
Moreover, IoU for each class are also displayed for reference.

Submission format
^^^^^^^^^^^^^^^^^^^^^^

To evaluate your algorithms on the BDD100K detection benchmark, you may prepare predictions in 1-channel png files.

Run Evaluation on Your Own
^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can evaluate your algorithm with public annotations by running 
::
    
    python3 -m bdd100k.eval.run -t sem_seg -g ${gt_path} -r ${res_path}

- `gt_path`: the path to ground-truch bitmask images folder.
- `res_path`: the path to the results bitmask images folder.


Drivable Area
~~~~~~~~~~~~~~~~~~~~~~~~

The drivable area task applies the same rule with semantic segmentation.
One notable difference is that they have different class definitions and numbers.
Another is that the prediction of background pixels matters for drivable area.
Unlike semantic segmentation, which ignores *unknown* pixels, drivable area instead takes consideration of
*background* pixels when computing IoUs. Though the *background* class is not counted into the final mIoU.

Run Evaluation on Your Own
^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can evaluate your algorithm with public annotations by running 
::
    
    python3 -m bdd100k.eval.run -t drivable -g ${gt_path} -r ${res_path}

- `gt_path`: the path to ground-truch bitmask images folder.
- `res_path`: the path to the results bitmask images folder.


Lane Marking
~~~~~~~~~~~~~~~~~~~~~~~~

The lane marking takes the F-score [1] as the measurement.
We evaluate the F-score for each cateogry of the three sub-tasks with threshold as 1, 2 and 10 pixels.
Before the evaluation, morphological thinning is adopted to get predictions of 1-pixel width.
For each sub-task, the mean F-score will be showed.
The main item for the leaderboard is the averaged mean F-score of these three sub-tasks.

[1] `A Benchmark Dataset and Evaluation Methodology for Video Object Segmentation. F. Perazzi, J. Pont-Tuset, B. McWilliams, L. Van Gool, M. Gross, and A. Sorkine-Hornung. Computer Vision and Pattern Recognition (CVPR) 2016 <https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Perazzi_A_Benchmark_Dataset_CVPR_2016_paper.pdf>`_

Submission format
^^^^^^^^^^^^^^^^^^^^^^

To evaluate your algorithms on the BDD100K detection benchmark, you may prepare predictions in 1-channel png files.
The submission format should be aligned with label format defined in :ref:`Lane Marking Format <lane mask>`.


Run Evaluation on Your Own
^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can evaluate your algorithm with public annotations by running 
::
    
    python3 -m bdd100k.eval.run -t lane_mark -g ${gt_path} -r ${res_path}


Multiple Object Tracking
~~~~~~~~~~~~~~~~~~~~~~~~

Submission format
^^^^^^^^^^^^^^^^^^^^^^

To evaluate your algorithms on BDD100K multiple object tracking benchmark, the submission must be in one of these formats:

- A zip file of a folder that contains JSON files of each video.

- A zip file of a file that contains a JSON file of the entire evaluation set.

The JSON file for each video should contain a list of per-frame result dictionaries with the following structure:
::

    - videoName: str, name of the current sequence,
    - name: str, name of the current frame,
    - framIndex: int, index of the current frame within the sequence,
    - labels []:
        - id: str, unique instance id of the prediction in the current sequence,
        - category: str, name of the predicted category,
        - box2d []:
            - x1: float,
            - y1: float,
            - x2: float,
            - y2: float

You can find an example result file in `bbd100k.eval.testcases <https://github.com/bdd100k/bdd100k/blob/master/bdd100k/eval/testcases/track_predictions.json>`_

Run Evaluation on Your Own
^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can evaluate your algorithms with public annotations by running
::

    python -m bdd100k.eval.run -t box_track -g ${gt_file} -r ${res_file} 


Evaluation Metrics
^^^^^^^^^^^^^^^^^^^^^^

We employ mean Multiple Object Tracking Accuracy (mMOTA, mean of MOTA of the 8 categories)
as our primary evaluation metric for ranking. 
We also employ mean ID F1 score (mIDF1) to highlight the performance 
of tracking consistency that is crucial for object tracking.
All metrics are detailed below.
Note that the overall performance is measured for all objects without considering the category if not mentioned.

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
After the bounding box matching process in evaluation, we ignore all detected false-positive boxes that have >50% overlap with the crowd region (ground-truth boxes with the "Crowd" attribute).

We also ignore object regions that are annotated as 3 distracting classes ("other person", "trailer", and "other vehicle") by the same strategy of crowd regions for simplicity. 


Pre-training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
It is a fair game to pre-train your network with **ImageNet**, 
but if other datasets are used, please note in the submission description. 
We will rank the methods without using external datasets except **ImageNet**.

.. Jiangmiao: online or offline constrains??
.. Jiangmiao: ranking metric by mMOTA? KITTI said no ranking metric. 


Multi Object Tracking and Segmentation (Segmentation Tracking)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We use the same metrics set as MOT above. The only difference lies in the computation of distance matrixes.
Concretely, in MOT, it is computed using box IoU. While for MOTS, the mask IoU is used.

Submission format
^^^^^^^^^^^^^^^^^

The submission should be a zipped nested folder for bitmask images.
Moreover, images belonging to the same video should be placed in the same folder, named by ${videoName}.

You can find an example bitmask file in `bbd100k.eval.testcases.mots <https://github.com/bdd100k/bdd100k/blob/master/bdd100k/eval/testcases/mots/example_bitmask.png>`_

Run Evaluation on Your Own
^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can evaluate your algorithms with public annotations by running
::

    python -m bdd100k.eval.run -t seg_track -g ${gt_path} -r ${res_path} 

- `gt_path`: the path to the ground-truch bitmask images folder.
- `res_path`: the path to the results bitmask images folder.
