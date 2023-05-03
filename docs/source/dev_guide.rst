.. _dev_ref:

**Developer's Guide**
=====================

Welcome to the Developer's Guide for SarcGraph! This guide is intended for 
developers, contributors, and maintainers who are interested in understanding, 
modifying, or extending the functionality of SarcGraph. Our goal is to create a 
welcoming and collaborative environment for everyone who wants to contribute to 
this project.

In this guide, we'll cover topics such as setting up your development 
environment, code style and conventions, testing, documentation, version control
and branching, the contribution process, release management, and community 
communication. We hope this information will help you become a valuable 
contributor to the SarcGraph project.

Let's dive in and get started with a project overview!

**Project Overview**
--------------------

SarcGraph is a specialized software tool designed to streamline the detection, 
tracking, and analysis of z-discs and sarcomeres in movies of beating human 
induced pluripotent stem cell-derived cardiomyocytes (hiPSC-CMs). The main 
objective is to offer researchers a user-friendly, adaptable, and efficient 
solution for studying sarcomere dynamics in hiPSC-CMs.

The project comprises two primary modules: sg and sg_tools.

1. :ref:`sg <sg.sarcgraph>`: This module encompasses the core functionality of SarcGraph, 
including image processing techniques for z-disc segmentation and tracking, as 
well as sarcomere detection. The module employs the Trackpy library to handle 
the tracking tasks effectively.

2. :ref:`sg_tools <sg_tools.sarcgraphtools>`: This module features three functional categories:

- :class:`TimeSeries <sarcgraph.sg_tools.SarcGraphTools.TimeSeries>`: Offers functions for handling time series data related to the 
  sarcomeres, such as length and position information.

- :class:`Analysis <sarcgraph.sg_tools.SarcGraphTools.Analysis>`: Provides post-processing analysis tools for extracting high-level 
  mechanical properties, like average deformation or orientation order 
  parameter.

- :class:`Visualization <sarcgraph.sg_tools.SarcGraphTools.Visualization>`: Supplies a range of visualization utilities for displaying 
  the results of detection, tracking, and analysis, enabling users to better 
  comprehend and interpret their findings.

SarcGraph's source code, written in Python, leverages various popular scientific
computing libraries like NumPy, SciPy, scikit-image, scikit-learn, pandas, and 
matplotlib, along with the Trackpy library for tracking purposes. The code 
organization is simple and efficient, with each component housed in its 
respective module.

In the sections that follow, we will outline the steps for setting up your 
development environment, working with the code, and contributing to the project.

**Setting Up Your Development Environment**
-------------------------------------------

Before you can start contributing to the SarcGraph project, you'll need to set 
up your development environment. This section will guide you through the 
necessary steps to prepare your environment for working with the SarcGraph 
codebase.

1. **System requirements and package dependencies**

SarcGraph is compatible with Linux, macOS, and Windows operating systems. We 
have tested the software on Python 3.10, and it is recommended to use this 
version for development. Below is a table listing the required packages and 
their respective versions:

.. list-table::
   :header-rows: 1

   * - Package Name
       - Version
   * - ffmpeg
       - 4.2.2
   * - matplotlib
       - 3.5.2
   * - networkx
   * - numpy
   * - pandas
   * - scikit-image
   * - scikit-learn
   * - scipy
   * - sk-video
   * - trackpy
   * - pytest
   * - pytest-cov
   * - flake8
   * - black
   * - sphinx
   * - sphinx_rtd_theme
   * - sphinx-design
   * - sphinx-copybutton
   * - nbsphinx
   * - ipykernel
   * - pandoc

.. note::

   These packages can be installed using Conda from the default channel or the 
   conda-forge channel. If you prefer to use pip, note that it cannot install 
   ffmpeg and pandoc. Installing instructions for ffmpeg and pandoc on different
   operating systems are different and can be challenging, so we recommend using
   Conda to install these packages.

2. **Clone the repository**

First, clone the SarcGraph repository from GitHub:

.. code-block:: bash

   git clone https://github.com/Sarc-Graph/sarcgraph.git
   cd sarcgraph

3. **Setting up the development environment with Conda**

We recommend using Conda to set up your development environment, as it 
simplifies the installation of the required packages. We have provided an 
``environment.yml`` file in the repository that you can use to create a Conda 
environment and install all necessary dependencies:

.. code-block:: bash

   conda env create -f environment.yml
   conda activate sarcgraph-dev

Install the package in editable mode:

.. code-block:: bash

    $ pip install -e .

4. **Additional tools and software**

To maintain consistency with the existing codebase, we use the following tools:

- Formatting and linting: ``flake8`` and ``black``
- Documentation generation: ``sphinx``
- Testing and code coverage: ``pytest`` and ``pytest-cov``

Ensure that you have these tools installed and configured in your development 
environment.

With your development environment set up, you can now start working with the 
SarcGraph codebase and contribute to the project.

Contributing Guidelines
-----------------------

We welcome contributions from the community and appreciate your efforts to 
improve SarcGraph. To ensure that your contributions are easily integrated into 
the project, please follow these guidelines:

1. **Fork the repository**

Create a fork of the SarcGraph repository on GitHub, and clone your fork to your
local machine:

.. code-block:: bash

   git clone https://github.com/<your-username>/sarcgraph.git

2. **Create a feature branch**

Always create a new branch for your changes. This helps to keep the codebase 
organized and ensures that your changes do not interfere with the main branch. 
Name your branch descriptively to indicate the purpose of the changes:

.. code-block:: bash

   git checkout -b my-feature-branch

3. **Commit your changes**

Commit your changes to the feature branch, and make sure that your commit 
messages are clear and informative. This helps others understand the purpose of 
your changes:

.. code-block:: bash

   git commit -m "Add a new feature to improve functionality"

4. **Stay up-to-date with the main branch**

Ensure that your fork is up-to-date with the main branch to minimize the chances
of conflicts when merging your changes:

.. code-block:: bash

   git remote add upstream https://github.com/Sarc-Graph/sarcgraph.git
   git fetch upstream
   git merge upstream/main

5. **Test your changes**

To ensure the reliability and stability of SarcGraph, we use a comprehensive 
test suite that covers various aspects of the software. We encourage you to run 
the test suite after making changes to the codebase to make sure your 
modifications do not introduce new issues or break existing functionality.

To run the tests, make sure you have ``pytest`` and ``pytest-cov`` installed in 
your environment. Then navigate to the root directory of the SarcGraph 
repository and run the following command:

.. code-block:: bash

   pytest

This will run the entire test suite, and the output will indicate whether the 
tests have passed or failed. If any tests fail, please review your changes and 
ensure that they are not causing the issues.

In addition to running the tests, you can also check the code coverage to see 
which parts of the code are not covered by the tests. This can help you identify
areas that may need additional testing. To check the code coverage, run the 
following command:

.. code-block:: bash

   pytest --cov=sarcgraph

The output will display the percentage of code covered by the tests for each 
module in the SarcGraph package.

6. **Adhere to code style guidelines**

To maintain a consistent and clean codebase, we follow a set of code style and 
formatting guidelines. We use ``flake8`` for linting and ``black`` for code 
formatting. Before submitting a pull request, make sure your code adheres to 
these guidelines by running ``flake8`` and ``black`` on your changes.

You can lint your code with ``flake8`` by running the following command in the 
root directory of the SarcGraph repository:

.. code-block:: bash

   flake8

If ``flake8`` identifies any issues, you should fix them before submitting your 
pull request.

To automatically format your code with ``black``, run the following command in 
the root directory of the SarcGraph repository:

.. code-block:: bash

   black .

This will reformat your code according to the project's formatting rules, 
ensuring that it is clean and consistent with the rest of the codebase.

7. **Update documentation**

Keeping the documentation up-to-date and comprehensive is crucial for the 
usability of SarcGraph. If you make changes to the codebase or add new features,
please update the documentation accordingly.

We use ``Sphinx`` to generate the documentation, which is written in 
reStructuredText (RST) format. To make changes to the documentation, you can 
edit the RST files located in the `docs/source` directory. If you are unfamiliar
with RST, you can refer to the 
`reStructuredText Primer <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_
for guidance.

With the required packages installed, navigate to the ``docs`` directory within 
the SarcGraph repository and run the following command:

.. code-block:: bash

   make html

This will generate the HTML documentation in the ``docs/build/html`` directory. 
Open the ``index.html`` file in your web browser to view the local version of the 
documentation.

After making changes to the documentation, please verify that it builds 
correctly and that your updates are accurately reflected in the generated HTML.

8. **Submit a pull request**

Once your changes are complete and tested, push your feature branch to your fork
on GitHub, and create a pull request against the main branch of the SarcGraph 
repository:

.. code-block:: bash

   git push origin my-feature-branch

Make sure that your pull request has a clear title and a detailed description of
the changes you have made. This makes it easier for maintainers to review your 
changes.

After submitting your pull request, the maintainers will review your changes and
provide feedback. Once your changes have been reviewed and approved, they will 
be merged into the main branch.

If you have any questions or need assistance, feel free to contact Emma Lejeune 
at elejeune@bu.edu.