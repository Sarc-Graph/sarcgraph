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

    conda create --name sarcgraph-env -c conda-forge -c saeedmhz sarcgraph

Type ``y`` and press ``Enter`` when prompted. 

**3. Activate the environment**

.. code-block:: bash

    conda activate sarcgraph-env

**From PyPI**
*************

Since `ffmpeg <https://ffmpeg.org/>`_ is not availble on PyPI, you need to 
install it seperately. We recommend using Anaconda to install the correct 
version of ffmpeg.

**1. Install ffmpeg**

Install anaconda and run the following commands in Anaconda terminal to create 
and activate a new environment:

.. code-block:: bash

    conda create --name sarcgraph-env python=3.10

.. code-block:: bash

    conda activate sarcgraph-env

Run the following command to install ffmpeg:

.. code-block:: bash

    conda install -c conda-forge ffmpeg=4.2.2

**2. Install SarcGraph**

SarcGraph and its dependencies can be installed using the following command:

.. code-block:: bash

    pip install --upgrade sarcgraph

**Verify Installation**
***********************

Run the following command and check if it runs with no errors:

.. code-block:: bash

    python -c "from sarcgraph.sg import SarcGraph"


**Run Tutorial Notebooks**
--------------------------

You have two options for running the tutorial demos in the 
`tutorials <https://github.com/Sarc-Graph/sarcgraph/tree/main/tutorials>`_ 
directory:

1. **Use Binder:**

You can run the tutorials directly in your browser using 
`Binder <https://mybinder.org/>`_. This option does not require you to set up 
the environment locally, but it may take some time for Binder to finish the 
initial setup. Click the badge below to launch the tutorials on Binder:

.. image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/Sarc-Graph/sarcgraph/main?filepath=tutorials

2. **Run Locally:**

If you prefer to run the tutorials on your local machine, follow these steps:

   a. Set up your local environment by activating the ``sarcgraph-env`` conda 
   environment:

      .. code-block:: bash

         conda activate sarcgraph-env

   b. Install the ``Jupyter`` package:

      .. code-block:: bash

         pip install jupyter

      If you're new to Jupyter Notebooks, you may find these introductory videos helpful:

      - `Jupyter Notebook Tutorial: Introduction, Setup, and Walkthrough <https://www.youtube.com/watch?v=HW29067qVWk>`_
      - `Jupyter Notebook Complete Beginner Guide 2023 <https://www.youtube.com/watch?v=5pf0_bpNbkw>`_

   c. Download the tutorial files by either:

      - Downloading the individual notebook files from the 
        `tutorials folder <https://github.com/Sarc-Graph/sarcgraph/tree/main/tutorials>`_
        in the GitHub repository. You can do this by clicking on each file, then
        clicking the "Raw" button, and finally right-clicking and selecting 
        "Save As" to save the file to your local machine. Make sure to save the
        file with the ``.ipynb`` extension. 

        Once you have downloaded the tutorial files, move them to a local 
        directory. You can use the following command to move to that directory:

        .. code-block:: bash

           cd path/to/local/directory

      - Cloning the entire GitHub repository using the following command:

        .. code-block:: bash

           git clone https://github.com/Sarc-Graph/sarcgraph

        Once you have cloned the repository, move to the tutorials directory:

        .. code-block:: bash

           cd sarcgraph/tutorials

   d. You can open and run any of the demos by executing the following command 
   while in the directory where the tutorial files are located:

      .. code-block:: bash

         jupyter notebook demo_file_name.ipynb