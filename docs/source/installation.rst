.. _installation_ref:

**Installation Guide**
======================

To install SarcGraph follow the following steps:

**Prerequisites**
-----------------

SarcGraph needs the following packages to run:

+-------------------+---------+
| Package Name      | Version |
+===================+=========+
| ffmpeg            | 4.2.2   |
+-------------------+---------+
| matplotlib        | 3.5.2   |
+-------------------+---------+
| networkx          | 2.8.4   |
+-------------------+---------+
| numpy             | 1.23.5  |
+-------------------+---------+
| pandas            | 1.5.2   |
+-------------------+---------+
| scikit-image      | 0.19.3  |
+-------------------+---------+
| scikit-learn      | 1.2.1   |
+-------------------+---------+
| scipy             | 1.10.0  |
+-------------------+---------+
| sk-video          | 1.1.10  |
+-------------------+---------+
| trackpy           | 0.6.1   |
+-------------------+---------+

**Installation Steps**
----------------------

We recommend using Anaconda to install prerequisites.

**1. Install Anaconda**

Follow `Anaconda's installation instructions <https://docs.anaconda.com/anaconda/install/index.html>`_.

**2. Create a new environment**

Run the following command in Anaconda terminal to create a new conda environment:

.. code-block:: bash

    $ conda create -n sarcgraph python=3.10

Activate the ``sarcgraph`` environment:

.. code-block:: bash

    $ conda activate sarcgraph

**3. Install prerequisites**

Run the following commands to install prerequisites:

.. code-block:: bash

    $ conda install -c anaconda networkx=2.8.4 numpy=1.23.5 pandas=1.5.2 scikit-image=0.19.3 scikit-learn=1.2.1 scipy=1.10.0
    $ conda install -c conda-forge ffmpeg=4.2.2 matplotlib=3.5.2 sk-video=1.1.10 trackpy=0.6.1

**4. Install SarcGraph**

SarcGraph can be installed using the following command:

.. code-block:: bash

    $ pip install sarcgraph

**5. Verify installation**

Run the following command and check if it runs with no errors:

.. code-block:: bash

    $ python -c "from sarcgraph.sg import SarcGraph"

**Troubleshooting**
-------------------
