# **SarcGraph**

[![python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/) ![os](https://img.shields.io/badge/os-ubuntu%20|%20macos%20|%20windows-blue.svg) [![license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/Sarc-Graph/sarcgraph#license)

[![flake8](https://github.com/Sarc-Graph/sarcgraph/actions/workflows/black_flake8.yml/badge.svg)](https://github.com/Sarc-Graph/sarcgraph/actions/workflows/black_flake8.yml) [![codecov](https://codecov.io/gh/Sarc-Graph/sarcgraph/branch/main/graph/badge.svg?token=XNE85EJ4GX)](https://codecov.io/gh/Sarc-Graph/sarcgraph) [![Documentation Status](https://readthedocs.org/projects/sarc-graph/badge/?version=latest)](https://sarc-graph.readthedocs.io/en/latest/?badge=latest)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Sarc-Graph/sarcgraph/main)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7963553.svg)](https://doi.org/10.5281/zenodo.7963553)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.05322/status.svg)](https://doi.org/10.21105/joss.05322)

## **Table of Contents**
* [Project Summary](#summary)
* [Installation Instructions](#install)
* [Contents](#contents)
* [Tutorial](#tutorial) - [Notebooks](https://github.com/Sarc-Graph/sarcgraph/tree/main/tutorials)
* [Validation](#validation)
* [References to Related Work](#references)
* [Contact Information](#contact)
* [Acknowledgements](#acknowledge)

## **Project Summary** <a name="summary"></a>

**SarcGraph** is a tool for automatic detection, tracking and analysis of
z-discs and sarcomeres in movies of beating *human induced pluripotent stem
cell-derived cardiomyocytes (hiPSC-CMs)*.

<br />
<center><img src="figures/intro.png" width=30%></center>
<br />

SarcGraph was initially introduced in [Sarc-Graph: Automated segmentation, tracking, and analysis of sarcomeres in hiPSC-derived cardiomyocytes](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009443).
This package is created to make SarcGraph more accessible to the broader
research community.

**For more information visit [SarcGraph documentation](https://sarc-graph.readthedocs.io/en/latest/).**

## **Installation Instructions** <a name="install"></a>

### **Stable Version** <a name="install-stable"></a>

#### From Conda

Follow the instructions to install [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

Create a new Conda envirnoment and install sarcgraph.

```bash
conda create --name sarcgraph-env -c conda-forge -c saeedmhz sarcgraph
```

**Note:** Type ``y`` and press ``Enter`` when prompted.

Activate the environment.

```bash
conda activate sarcgraph-env
```
#### From PyPI

Please check the [Getting Started](https://sarc-graph.readthedocs.io/en/latest/installation.html) section in the documentation.

### **Developer's Version** <a name="install-dev"></a>

#### **Get a copy of the [SarcGraph repository](https://github.com/Sarc-Graph/sarcgraph) on your local machine**

You can do this by clicking the green ``<> code`` button and selecting ``Download Zip`` or by running the following command in terminal and move to the directory of the ``sarcgraph`` repository.:

```bash
git clone https://github.com/Sarc-Graph/sarcgraph.git
cd sarcgraph
```

#### **Create and activate a conda virtual environment**

1. Install [Anaconda](https://docs.anaconda.com/anaconda/install/) on your local machine.

2. Type the following command in terminal to create a virtual envirnoment and install the required packages:

```bash
conda env create --file=environment.yml
```

3. Activate your virtual environment.

```bash
conda activate sarcgraph
```

#### **Install SarcGraph**

Install SarcGraph in editable mode:

```bash
pip install -e .
```

## **Contents** <a name="contents"></a>

```bash
|___ sarcgraph
|        |___ docs/
|        |___ figures/
|                |___ *.png
|        |___ samples/
|        |___ sarcgraph/
|                |___ __init__.py
|                |___ sg.py
|                |___ sg_tools.py
|        |___ tests/
|        |___ tutorials/
|                |___ *.ipynb
```

## **Tutorial** <a name="tutorial"></a>

This GitHub repository contains a folder called ``tutorials`` that contains demos to extensively show how this package can be used to analyze videos or images of hiPSC-CMs.

### **Package Contents** <a name="whats-in-package"></a>

The package contains two seperate modules: `sg` for sarcomere detection and tracking and `sg_tools` for running further analysis and visualizations.

#### **sarcgraph.sg** <a name="sarcgraph.py"></a>
`sarcgraph.sg` module takes a video/image file as input (more details in tutorials). This module then processes the input file to detect and track z-discs and sarcomeres through running 3 tasks:

 - Z-disc Segmentation,
 - Z-disc Tracking,
 - Sarcomere Detection.

Here is a list of functions developed for each task:

- `zdisc_segmentation`: Detect z-discs in each frame of the input video/image and saves the following information into a pandas `DataFrame`:

> - `frame`: (frame number) 
> - `x` and `y`: (X and Y position of the center of a z-disc)
> - `p1_x`, `p1_y` and `p2_x`, `p2_y`: (X and Y position of both ends of a z-disc)

- `zdisc_tracking`: Tracks detected z-discs in the input video over all frames and adds the following information to the pandas `DataFrame`:

> - `particle`: (z-disc id)
> - `freq`: (number of frames in which a z-discs is tracked)
frame,sarc_id,x,y,length,width,angle,z-discs

- `sarcomere_detection`: Detects sarcomeres in the input video/image using tracked z-discs `DataFrame` and saves the following information into a new pandas `DataFrame`:

> - `frame`: (frame number)
> - `sarc_id`: (sarcomere id)
> - `x` and `y`: (X and Y position of the center of a sarcomere)
> - `length`: (sarcomere length)
> - `width`: (sarcomere width)
> - `angle`: (sarcomere angle)
> - `zdiscs`: (ids of the two z-discs forming a sarcomere)


#### **sarcgraph.sg_tools** <a name="sarcgraph_tools.py"></a>

`sarcgraph.sg_tools` module consists of 3 subclasses:

- `TimeSeries`: Process timeseries of detected and tracked sarcomeres

> - `sarcomeres_gpr()`: Applies Gaussian Process Regression (GPR) on each recovered timeseries characteristic of all detected sarcomeres to reduce the noise and fill in the missing data

- `Analysis`: Extract more information from detected sarcomeres characteristics timeseries

> - `compute_F_J`: Computes the average deformation gradient (F) and its jacobian (J)
> - `compute_OOP`: Computes the Orientation Order Parameter (OOP)
> - `compute_metrics`: Computes {OOP, C_iso, C_OOP, s_til, s_avg} as defined in the [SarcGraph paper](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009443)
> - `compute_ts_params`: Computes timeseries constants (contraction time, relaxation time, flat time, period, offset)
> - `create_spatial_graph`: Generates a spatial graph of tracked z-discs where edges indicate sarcomeres and edge weights indicate the ratio of the frames in which each sarcomere is detected

- `Visualization`: Visualize detected sarcomeres information

> - `zdiscs_and_sarcs`: Visualizes detected z-discs and sarcomeres in the chosen frame
> - `contraction`:Visualizes detected sarcomeres in every frame as a gif file
> - `normalized_sarcs_length`: Plots normalized length of all detected sarcomeres vs frame number
> - `OOP`: Plots recovered Orientational Order Parameter
> - `F`: Plots recovered deformation gradient
> - `J`: Plots recovered deformation jacobian
> - `F_eigenval_animation`: Visualizes the eigenvalues of F vs frame number
> - `timeseries_params`: Visualizes time series parameters
> - `dendrogram`: Clusters timeseries and plots as a dendrogram of the clusters
> - `spatial_graph`: Visualizes the spatial graph
> - `tracked_vs_untracked`: Visualizes metrics that compare the effect of tracking sarcomeres in a video vs only detecting sarcomeres in each frame without tracking

To use this module an object of the class `SarcGraphTools` should be created by setting the `input_dir` to the folder that contains the output saved from running full sarcomere detection and timeseries processing on the input data.

## Validation <a name="validation"></a>

To validate our methods and ensure correct implementation, we generated challenging synthetic videos with characteristics similar to beating hiPSC-CMs. We used these videos to evaluate the sarcomere detection algorithm by comparing recovered metrics to their known ground truth. The figure shows this process for one of many tested validation examples.

<br />
<center><img src="figures/validation.png" width=75%></center>
<br />

## References to Related Work <a name="references"></a>

* Zhao, B., Zhang, K., Chen, C. S., & Lejeune, E. (2021). Sarc-graph: Automated segmentation, tracking, and analysis of sarcomeres in hiPSC-derived cardiomyocytes. PLoS Computational Biology, 17(10), e1009443.

* Allan, D. B., Caswell, T., Keim, N. C., Wel, C. M. van der, & Verweij, R. W. (2023). Soft-matter/trackpy: v0.6.1 (Version v0.6.1). Zenodo. https://doi.org/10.5281/zenodo.7670439

* Toepfer, C. N., Sharma, A., Cicconet, M., Garfinkel, A. C., Mücke, M., Neyazi, M., Willcox, J. A., Agarwal, R., Schmid, M., Rao, J., & others. (2019). SarcTrack: An adaptable software tool for efficient large-scale analysis of sarcomere function in hiPSC-cardiomyocytes. Circulation Research, 124(8), 1172–1183.

* Morris, T. A., Naik, 94 J., Fibben, K. S., Kong, X., Kiyono, T., Yokomori, K., & Grosberg, A. (2020). Striated myocyte structural integrity: Automated analysis of sarcomeric z-discs. PLoS Computational Biology, 16(3), e1007676.

* Pasqualin, C., Gannier, F., Yu, A., Malécot, C. O., Bredeloux, P., & Maupoil, V. (2016). SarcOptiM for ImageJ: High-frequency online sarcomere length computing on stimulated cardiomyocytes. American Journal of Physiology-Cell Physiology, 311(2), C277–C283.

* Ribeiro, A. J. S., Schwab, O., Mandegar, M. A., Ang, Y.-S., Conklin, B. R., Srivastava, D., & Pruitt, B. L. (2017). Multi-imaging method to assay the contractile mechanical output of micropatterned human iPSC-derived cardiac myocytes. Circulation Research, 120(10), 1572–1583. https://doi.org/10.1161/CIRCRESAHA.116.310363

## Contact Information <a name="contact"></a>

For information about this software, please get in touch with [Saeed Mohammadzadeh](mailto:saeedmhz@bu.edu) or [Emma Lejeune](mailto:elejeune@bu.edu).

## Acknowledgements <a name="acknowledge"></a>
