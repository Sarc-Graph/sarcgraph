# **SarcGraph**

[![license](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/Sarc-Graph/sarcgraph#license)

[![flake8](https://github.com/Sarc-Graph/sarcgraph/actions/workflows/black_flake8.yml/badge.svg)](https://github.com/Sarc-Graph/sarcgraph/actions/workflows/black_flake8.yml)

[![codecov](https://codecov.io/gh/Sarc-Graph/sarcgraph/branch/main/graph/badge.svg?token=XNE85EJ4GX)](https://codecov.io/gh/Sarc-Graph/sarcgraph)

## **Table of Contents**
* [Project Summary](#summary)
* [Installation Instructions](#install)
* [Tutorial](#tutorial) - [Notebooks](https://github.com/Sarc-Graph/sarcgraph/tree/main/tutorials)
* [Validation](#validation)
* [References to Related Work](#references)
* [Contact Information](#contact)
* [Acknowledgements](#acknowledge)

## **Project Summary** <a name="summary"></a>

**SarcGraph** is a tool for automatic detection, tracking and analysis of
z-discs and sarcomeres in movies of beating *human induced pluripotent stem
cell-derived cardiomyocytes (hiPSC-CMs)*.

SarcGraph was initially introduced in [Sarc-Graph: Automated segmentation, tracking, and analysis of sarcomeres in hiPSC-derived cardiomyocytes](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009443).
This package is created to make SarcGraph more accessible to the broader
research community.

## **Installation Instructions** <a name="install"></a>

### **Get a copy of the SarcGraph repository on your local machine**

You can do this by clicking the green ``<> code`` button and selecting ``Download Zip`` or by running the following command in terminal:

```bash
git clone https://github.com/Sarc-Graph/sarcgraph.git
```

### **Create and activate a conda virtual environment**

1. Install [Anaconda](https://docs.anaconda.com/anaconda/install/) on your local machine.

2. Open a terminal and move to the directory of the ``sarcgraph`` repository. Then, type the following command in terminal to create a virtual envirnoment and install the required packages:

```bash
cd sarcgraph
conda env create --file=environments.yml
```

3. Activate your virtual environment.

```bash
conda activate sarcgraph
```

### **Install SarcGraph**

SarcGraph can be installed using ``pip``:

```bash
pip install sarcgraph
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

## References to Related Work <a name="references"></a>

## Contact Information <a name="contact"></a>

## Acknowledgements <a name="acknowledge"></a>
