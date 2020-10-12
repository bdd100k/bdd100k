Using Data
-------------

Dependency
~~~~~~~~~~~

The BDD100K toolkit depends on Python 3.7+. To install the python dependencies:

.. code-block:: bash

    pip3 install -r requirements.txt


Understanding the Data
~~~~~~~~~~~~~~~~~~~~~~~

After being unzipped, all the files will reside in a folder named ``bdd100k``. All
the original videos are in ``bdd100k/videos`` and labels in ``bdd100k/labels``.
``bdd100k/images`` contains the frame at 10th second in the corresponding video.


``bdd100k/labels`` contains json files based on `Scalabel Format
<https://www.scalabel.ai/doc/format.html>`_ for training and validation sets. |vis_labels|_ provides examples to parse and
visualize the labels.

.. |vis_labels| replace:: ``bdd100k.vis.labels``
.. _vis_labels: https://github.com/bdd100k/bdd100k/blob/master/bdd100k/vis/labels.py

For example, you can view training data one by one

.. code-block:: bash

    python3 -m bdd100k.vis.labels --image-dir bdd100k/images/100k/train \
        -l bdd100k/labels/bdd100k_labels_images_train.json


Or export the drivable area in segmentation maps:

.. code-block:: bash

    python3 -m bdd100k.vis.labels --image-dir bdd100k/images/100k/train \
        -l bdd100k/labels/bdd100k_labels_images_train.json \
        -s 1 -o bdd100k/out_drivable_maps/train --drivable


This exporting process will take a while, so we also provide ``Drivable Maps`` in
the downloading page, which will be ``bdd100k/drivable_maps`` after decompressing.
There are 3 possible labels on the maps: 0 for background, 1 for direct drivable
area and 2 for alternative drivable area.


Trajectories
~~~~~~~~~~~~~

To visualize the GPS trajectories provided in ``bdd100k/info``, you can run the
command below to produce an html file that displays a single trajectory and
output the results in folder ``out/``:


.. code-block:: bash

    python3 -m bdd100k.vis.trajectory \
        -i bdd100k/info/train/0000f77c-6257be58.json -o out/ -k {YOUR_API_KEY}


Or create html file for each GPS trajectory in a directory, for example:

.. code-block:: bash

    python3 -m bdd100k.vis.trajectory \
        -i bdd100k/info/train -o out/ -k {YOUR_API_KEY}


To create a Google Map API key, please follow the instruction
`here <https://developers.google.com/maps/documentation/embed/get-api-key>`_. The
generated maps will look like

.. figure:: ../media/images/trajectory_gmap.jpg
   :alt: Trajectory on Google Map

Semantic Segmentation
~~~~~~~~~~~~~~~~~~~~~~

At present time, instance segmentation is provided as semantic segmentation maps
and polygons in json will be provided in the future. The encoding of labels
should still be ``train_id`` defined in |bdd100k_label|_,
thus car should be 13.

.. |bdd100k_label| replace:: ``bdd100k.label.label``
.. _bdd100k_label: https://github.com/bdd100k/bdd100k/blob/master/bdd100k/label/label.py