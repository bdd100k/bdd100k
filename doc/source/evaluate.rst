Evaluation
===========


Detection
~~~~~~~~~

Submission format
^^^^^^^^^^^^^^^^^^^^^^

To evaluate your algorithms on BDD100K detection benchmark, you may prepare 
your prediction results as a list of bounding box predictions with the following format:
::

    {
        "name": str, name of the input image,
        "category": str, name of the predicted category,
        "score": float, confidence score of the prediction
        "box2d": List[float], [x1, y1, x2, y2]
    }

You can find an example result file in |bdd100k_testcase_det|_.

.. |bdd100k_testcase_det| replace:: ``bdd100k.eval.testcases``
.. _bdd100k_testcase_det: https://github.com/bdd100k/bdd100k/blob/master/bdd100k/eval/testcases/bbox_predictions.json

When you submit your results, save your results in a JSON file and then compress it into a zip file.

Run Evaluation on Your Own
^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can evaluate your algorithm with public annotations by running 
::
    
    python3 -m bdd100k.eval.run -t det -g ${gt_file} -r ${res_file} 

- `gt_file`: ground truth file in JSON, either in Scalabel format or COCO format. If using COCO format, add a flag `--ann-format coco`
- `res_file`: prediction results file in JSON, format as described above.

Other options.
- If you want to evaluate the detection performance on the BDD100K MOT set, 
you can add a flag `--mode track`. 
- You can also specify the output directory to save the evaluation results by updating `--out-dir ${out_dir}`.


Evaluation Metrics
^^^^^^^^^^^^^^^^^^^^^^
Similar to COCO evaluation, we report 12 scores as 
"AP", "AP_50", "AP_75", "AP_small", "AP_medium", "AP_large", "AR_max_1", "AR_max_10",
"AR_max_100", "AR_small", "AR_medium", "AR_large" across all the classes. 



Multiple Object Tracking
~~~~~~~~~~~~~~~~~~~~~~~~

Submission format
^^^^^^^^^^^^^^^^^^^^^^

To evaluate your algorithms on BDD100K multiple object tracking benchmark, there are two save/submit options:

- A zip file or a folder that contains JSON files of each video.

- A zip file or a file that contains a JSON file of the entire evaluation set.

The JSON file for each video should contain a list of per-frame result dictionaries with the following structure:
::

    {
        "video_name": str, name of the current sequence,
        "name": str, name of the current frame,
        "index": int, index of the current frame within the sequence,
        "labels": List[dict], List of predictions for the current frame
    }

The 'labels' list will contain the predictions for the current frame, each specified by another dict in `Scalabel Format <https://doc.scalabel.ai/format.html>`_:
::

    {
        "name": str, name of the input image,
        "category": str, name of the predicted category,
        "id": str, unique instance id of the prediction in the current sequence,
        "score": float, confidence score of the prediction,
        "box2d": dict[float], {x1, y1, x2, y2}
    }

You can find an example result file in |bdd100k_testcase_track|_.

.. |bdd100k_testcase_track| replace:: ``bdd100k.eval.testcases``
.. _bdd100k_testcase_track: https://github.com/bdd100k/bdd100k/blob/master/bdd100k/eval/testcases/track_predictions.json

Run Evaluation on Your Own
^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can evaluate your algorithms with public annotations by running
::

    python -m bdd100k.eval.run -t mot -g ${gt_file} -r ${res_file} 

To obtain results on val/test phase, submit your result files at `BDD100K 2D Multiple Object Tracking & Segmentation Challenge <https://competitions.codalab.org/competitions/29388>`_.



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
It is a fair game to pre-train your network with **ImageNet**, 
but if other datasets are used, please note in the submission description. 
We will rank the methods without using external datasets except **ImageNet**.

.. Jiangmiao: online or offline constrains??
.. Jiangmiao: ranking metric by mMOTA? KITTI said no ranking metric. 


Multi Object Tracking and Segmentation
~~~~~~~~~~~~~~~~~~~~~~~~

We use the same metrics set as MOT above. The only difference lies in the computation of distance matrixs.
Concretely, in MOT, it is computed using box IoU. While for MOTS, the mask IoU is used.

Submission format
^^^^^^^^^^^^^^^^^

The submission should be a zipped nested folder for bitmask images.
For each image, the results should be saved in an RGBA image with png format, and named as "${image_name}.png".
Moreover, images belonging to the same video should be places in the same folder, named by ${video_name}.

For the image, The first byte, R, is used for the category id range from 1 (instead of 0).
And G is for the instance attributes. Currently, four attributes are used, they are "truncated", "occluded", "crowd" and "ignore".
Note that boxes with "crowd" or "ignore" labels will not be considered during testing.
The above four attributes are stored in lowest important bits of G. Given this, ``G = 8 & truncated + 4 & occluded + 2 & crowd + ignore``
. Finally, the B channel and A channel together store the instance_id, which can be computed as ``B * 255 + A``.
You can also refer to the below image for reference.

.. figure:: ../images/bitmask.png
   :alt: Downloading buttons

Label format
^^^^^^^^^^^^^^^^^

We provide labels in both json and bitmask formats. Note that polygons stored in jsons is not the same format with COCO.
Instead, the "poly2d" entry in jsons stores vertices and control points of Bezeir Curves. 
We provides the scripts for convert BDD jsons into bitmask.
However, as the conversion process is not determinitic, we don't recommend converting it by yourself.
The evaluation scripts uses bitmasks as ground-truth, so we suggest using bitmasks as input all the way.

Run Evaluation on Your Own
^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can evaluate your algorithms with public annotations by running
::

    python -m bdd100k.eval.run -t mots -g ${gt_folder} -r ${res_folder} 

To obtain results on val/test phase, submit your result files at `BDD100K 2D Multiple Object Tracking Challenge <https://competitions.codalab.org/competitions/29388>`_.