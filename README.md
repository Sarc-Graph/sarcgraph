# SarcGraph-2.0

[![codecov](https://codecov.io/gh/Sarc-Graph/SarcGraph-2.0/branch/main/graph/badge.svg?token=XNE85EJ4GX)](https://codecov.io/gh/Sarc-Graph/SarcGraph-2.0)

<p align="center">
<a href="https://github.com/Sarc-Graph/SarcGraph-2.0/actions/workflows/black_flake8.yml">
  <img src="https://github.com/Sarc-Graph/SarcGraph-2.0/actions/workflows/black_flake8.yml/badge.svg?branch=main" />
</a>
</p>

## Table of Contents
* [Project Summary](#summary)
* [Project Roadmap](#roadmap)
* [Installation Instructions](#install)
* [Tutorial](#tutorial)
* [Validation](#validation)
* [To-Do List](#todo)
* [References to Related Work](#references)
* [Contact Information](#contact)
* [Acknowledgements](#acknowledge)

## Project Summary <a name="summary"></a>

## Project Roadmap <a name="roadmap"></a>

## Installation Instructions <a name="install"></a>

## Tutorial <a name="tutorial"></a>

This GitHub repository contains a folder called ``tutorials`` that contains an example dataset and python script for running the code.

### What's in the package <a name="whats-in-package"></a>

The package contains two seperate modules: `sarcgraph` for sarcomere detection and tracking and `sarcgraph_tools` for running further analysis and visualizations.

#### `sarcgraph` <a name="sarcgraph.py"></a>
`sarcgraph` module can take a video/image file as an input or a numpy array (more details in tutorials). This module then processes the input file into a numpy array and can run 3 functions on the data consecutively: `zdisc_segmentation`, `zdisc_tracking`, `sarcomere_detection`. Here is a summary of what each function does:

- *`zdisc_segmentation`:* Segment zdiscs in each frame of the input file and outputs a pandas `DataFrame` and saves the file in a folder defined by the user if `save_data=True`. Each row of the dataframe is a zdisc and columns are:
> - `frame`: (frame number) 
> - `x` and `y`: (X and Y position of the zdisc center in number of pixels)
> - `p1_x`, `p1_y` and p2_x`, `p2_y`: (X and Y position of both ends of a zdisc in number of pixels)

- *`zdisc_tracking`:* The input to this function could be either a video/image (or the numpy array of input file) or pandas dataframe of segmented zdiscs in the format explained before. The output adds `particle` (particle_id) and `freq` (number of frames in which a zdiscs is tracked).

- *`sarcomere_detection`:* The input to this function could be either a video/image (or the numpy array of input file) or a pandas dataframe tracked zdiscs in the format explained before. *The output is a 3d numpy array [5, number of tracked sarcomeres, number of frames] where on the first index the values are X anf Y position of the center of a sarcomere, sarcomere length, sarcomere width, and sarcomere angle. But, it is better to keep store this as a dataframe too.*

If `save_data` parameter is set to `True` each function saves outputs to a folder specified by the user.

#### `sarcgraph_tools` <a name="sarcgraph_tools.py"></a>

The `sarcgraph_tools` module consists of 3 classes of functions:

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
