# **SarcGraph**

[![license](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/Sarc-Graph/SarcGraph-2.0#license)

[![flake8](https://github.com/Sarc-Graph/SarcGraph-2.0/actions/workflows/black_flake8.yml/badge.svg)](https://github.com/Sarc-Graph/SarcGraph-2.0/actions/workflows/black_flake8.yml)

[![codecov](https://codecov.io/gh/Sarc-Graph/SarcGraph-2.0/branch/main/graph/badge.svg?token=XNE85EJ4GX)](https://codecov.io/gh/Sarc-Graph/SarcGraph-2.0)

## **Table of Contents**
* [Project Summary](#summary)
* [Installation Instructions](#install)
* [Tutorial](#tutorial)
* [Validation](#validation)
* [To-Do List](#todo)
* [References to Related Work](#references)
* [Contact Information](#contact)
* [Acknowledgements](#acknowledge)

## **Project Summary** <a name="summary"></a>

**SarcGraph** is a tool for automatic detection, tracking and analysis of
zdiscs and sarcomeres in movies of beating *human induced pluripotent stem
cell-derived cardiomyocytes (hiPSC-CMs)*.

SarcGraph was initially introduced in [Sarc-Graph: Automated segmentation, tracking, and analysis of sarcomeres in hiPSC-derived cardiomyocytes](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009443).
This package is created to make SarcGraph more accessible to the broader
research community.

## **Installation Instructions** <a name="install"></a>

### **Get a copy of the SarcGraph repository on your local machine**

You can do this by clicking the green ``<> code`` button and selecting ``Download Zip`` or by running the following command in terminal:

```bash
git clone https://github.com/Sarc-Graph/SarcGraph-2.0.git
```

### **Create and activate a conda virtual environment**

1. Install [Anaconda](https://docs.anaconda.com/anaconda/install/) on your local machine.

2. Open a terminal and move to the directory of the ``sarcgraph`` repository. Then, type the following command in terminal to create a virtual envirnoment and install the required packages:

```bash
conda env create --file=environments.yml python=3.10
```

3. Activate your virtual environment.

```bash
conda activate sarcgraph-env
```

### **Install SarcGraph**

SarcGraph can be installed using ``pip``:

```bash
pip install sarcgraph
```

## **Tutorial** <a name="tutorial"></a>

This GitHub repository contains a folder called ``tutorials`` that contains demos to extensively show how this package an be used to analize videos or images of HiPSC-CMs.

### **What's in the package** <a name="whats-in-package"></a>

The package contains two seperate modules: `sg` for sarcomere detection and tracking and `sg_tools` for running further analysis and visualizations.

#### **sarcgraph.sg** <a name="sarcgraph.py"></a>
`sarcgraph.sg` module takes a video/image file as input (more details in tutorials). This module then processes the input file to detect and track zdiscs and sarcomeres through running 3 tasks:

 - Zdisc Segmentation,
 - Zdisc Tracking,
 - Sarcomere Detection.

Here is a list of functions developed for each task:

- `zdisc_segmentation`: Detect zdiscs in each frame of the input video/image and saves the following information into a pandas `DataFrame`:

> - `frame`: (frame number) 
> - `x` and `y`: (X and Y position of the center of a zdisc)
> - `p1_x`, `p1_y` and `p2_x`, `p2_y`: (X and Y position of both ends of a zdisc)

- `zdisc_tracking`: Tracks detected zdiscs in the input video over all frames and adds the following information to the pandas `DataFrame`:

> - `particle`: (zdisc id)
> - `freq`: (number of frames in which a zdiscs is tracked)
frame,sarc_id,x,y,length,width,angle,zdiscs
- `sarcomere_detection`: Detects sarcomeres in the input video/image using tracked zdiscs `DataFrame` and saves the following information into a new pandas `DataFrame`:

> - `frame`: (frame number)
> - `sarc_id`: (sarcomere id)
> - `x` and `y`: (X and Y position of the center of a sarcomere)
> - `length`: (sarcomere length)
> - `width`: (sarcomere width)
> - `angle`: (sarcomere angle)
> - `zdiscs`: (ids of the two zdiscs forming a sarcomere)


#### **sarcgraph.sg_tools** <a name="sarcgraph_tools.py"></a>

`sarcgraph.sg_tools` module consists of 3 subclasses:

- `TimeSeries` for processing timeseries of detected and tracked sarcomeres *it makes sense to move time series to the `sarcgraph` module since it is a step that has to be done.*
- `Analysis` for extracting desired information out of the original input file
- `Visualization` for plotting different types of extracted information

To use this module an object of the class `SarcGraphTools should be created by setting the `input_dir` to the folder that contains the info saved from running full sarcomere detection and timeseries processing on the input data.

##### `TimeSeries` <a name="sarcgraph_tools.TimeSeries"></a>
*This will be moved to `sarcgraph.py`

##### `Visualization` <a name="sarcgraph_tools.Visualization"></a>

- `zdiscs_and_sarcs
- `contraction
- plot_F
- plot_J
- F_eigenval_animation
- plot_dendrogram

##### `Analysis` <a name="sarcgraph_tools.Analysis"></a>

- compute_F

### Preparing data for analysis <a name="data_prep"></a>

The input data can be an image, a video, or a numpy array. 
### Preparing an input file<a name="input"></a>

### Running the code

### Understanding the output files

## Validation <a name="validation"></a>

## To-Do List <a name="todo"></a>

## References to Related Work <a name="references"></a>

## Contact Information <a name="contact"></a>

## Acknowledgements <a name="acknowledge"></a>
