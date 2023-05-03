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

**Stable Version Installation**
-------------------------------

**From Conda**
**************

We recommend using Anaconda to install sarcgraph.

**1. Install Anaconda**

Follow `Anaconda's installation instructions <https://docs.anaconda.com/anaconda/install/index.html>`_. Alternatively you can install `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ (a minimal installer for conda).

**2. Create a new Conda environment and install sarcgraph**

Run the following command in Anaconda terminal to create a new conda environment
containing the sarcgraph package and all its dependencies:

.. code-block:: bash

    $ conda create --name sarcgraph-env -c conda-forge -c saeedmhz sarcgraph

Type ``y`` and press ``Enter`` when prompted. 

**3. Activate the environment**

.. code-block:: bash

    $ conda activate sarcgraph-env

**From PyPI**
*************

Since `ffmpeg <https://ffmpeg.org/>`_ is not availble on PyPI, you need to 
install it seperately. We recommend using Anaconda to install the correct 
version of ffmpeg.

**1. Install ffmpeg**

Install anaconda and run the following commands in Anaconda terminal to create 
and activate a new environment:

.. code-block:: bash

    $ conda create --name sarcgraph-env python=3.10

.. code-block:: bash

    $ conda activate sarcgraph-env

Run the following command to install ffmpeg:

.. code-block:: bash

    $ conda install -c conda-forge ffmpeg=4.2.2

**2. Install SarcGraph**

SarcGraph and its dependencies can be installed using the following command:

.. code-block:: bash

    $ pip install --upgrade sarcgraph

**Verify Installation**
***********************

Run the following command and check if it runs with no errors:

.. code-block:: bash

    $ python -c "from sarcgraph.sg import SarcGraph"

**Run Tutorial Notebooks**
--------------------------

To open and run tutorial demos in the 
`tutorials <https://github.com/Sarc-Graph/sarcgraph/tree/main/tutorials>`_ 
directory you may use 
`Jupyter <https://docs.jupyter.org/en/latest/index.html>`_ by following these 
steps:

1. Install the ``Jupyter`` package:

.. code-block:: bash

    $ conda activate sarcgraph
    $ pip install jupyter

2. Download the 
`SarcGraph repository <https://github.com/Sarc-Graph/sarcgraph>`_ and go to the 
``sarcgraph/tutorials/`` directory:

.. code-block:: bash

    $ git clone https://github.com/Sarc-Graph/sarcgraph
    $ cd sarcgraph/tutorials

3. You can open any of the demos by running the following command while in the 
``sarcgraph/tutorials/`` directory.:

.. code-block:: bash

    $ jupyter notebook demo_file_name.ipynb