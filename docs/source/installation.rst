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

Type ``y`` and press ``Enter`` when prompted. 

Activate the ``sarcgraph`` environment:

.. code-block:: bash

    $ conda activate sarcgraph

**3. Install pre-requisites**

Run the following commands to install prerequisites:

.. code-block:: bash

    $ conda install -c anaconda networkx=2.8.4 numpy=1.23.5 pandas=1.5.2 scikit-image=0.19.3 scikit-learn=1.2.1 scipy=1.10.0

.. code-block:: bash

    $ conda install -c conda-forge ffmpeg=4.2.2 matplotlib=3.5.2 sk-video=1.1.10 trackpy=0.6.1

**4. Install SarcGraph**

SarcGraph can be installed using the following command:

.. code-block:: bash

    $ pip install sarcgraph

**5. Verify installation**

Run the following command and check if it runs with no errors:

.. code-block:: bash

    $ python -c "from sarcgraph.sg import SarcGraph"

**Running Tests**
-----------------

Tests in the `SarcGraph repository <https://github.com/Sarc-Graph/sarcgraph>`_ are 
written using the `Pytest framework <https://docs.pytest.org/en/7.2.x/>`_. To run 
tests:

1. Install the `pytest` package:

.. code-block:: bash

    $ conda activate sarcgraph
    $ pip install -U pytest

2. Download the `SarcGraph repository <https://github.com/Sarc-Graph/sarcgraph>`_ 
and go to the ``sarcgraph`` directory:

.. code-block:: bash

    $ git clone https://github.com/Sarc-Graph/sarcgraph
    $ cd sarcgraph

3. Run tests (may take a few minutes to finish):

.. code-block:: bash

    $ pytest tests/ --verbose --disable-warnings

You should see a similar report when done:

.. code-block:: bash

    ================ 53 passed, 1885 warnings in 184.62s (0:03:04) ================

**Run Jupyter Notebooks**
-------------------------

To open and run tutorial demos in the ``tutorials/`` directory you may use 
`Jupyter <https://docs.jupyter.org/en/latest/index.html>`_ by following these 
steps:

1. Download the `SarcGraph repository <https://github.com/Sarc-Graph/sarcgraph>`_ 
and go to the ``sarcgraph/tutorials/`` directory:

.. code-block:: bash

    $ git clone https://github.com/Sarc-Graph/sarcgraph
    $ cd sarcgraph/tutorials

3. Install the ``Jupyter`` package:

.. code-block:: bash

    $ conda activate sarcgraph
    $ pip install jupyter

4. You can open any of the demos by running the following command while in the 
``sarcgraph/tutorials/`` directory.:

.. code-block:: bash

    $ jupyter notebook demo_file_name.ipynb


**Make Documentation**
----------------------

This documentation is built with `Sphinx <https://www.sphinx-doc.org/en/master/>`_.
You can generate the documentation locally by following these steps:

1. Install required packages:

.. code-block:: bash

    $ conda activate sarcgraph
    $ pip install sphinx sphinx-design sphinx-rtd-theme sphinx-copybutton nbsphinx ipykernel
    $ conda install -c conda-forge pandoc

2. Download the `SarcGraph repository <https://github.com/Sarc-Graph/sarcgraph>`_ 
and go to the ``sarcgraph/docs`` directory to make the documentation:

.. code-block:: bash

    $ git clone https://github.com/Sarc-Graph/sarcgraph
    $ cd sarcgraph/docs
    $ make html

3. When the process is done with no errors the documentation will be in the
``sarcgraph/docs/build/html/`` directory. Open ``index.html`` to access the home page.
