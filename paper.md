---
title: 'SarcGraph: A Python package for automated segmentation, tracking and analysis of sarcomeres in hiPSC-CMs'
tags:
  - Python
  - Quantitative Methods
  - Biomechanics
  - Sarcomeres
  - Cardiomyocytes
authors:
  - name: Saeed Mohammadzadeh
    orcid: 0000-0001-9879-044X
    affiliation: 1
  - name: Emma Lejeune
    orcid: 0000-0001-8099-3468
    corresponding: true
    affiliation: 2
affiliations:
  - name: Department of Systems Engineering, Boston University, Massachusetts, the United States of America
    index: 1
  - name: Department of Mechanical Engineering, Boston University, Massachusetts, the United States of America
    index: 2
date: 02 March 2023
bibliography: paper.bib
---

# Summary

Heart disease remains the leading cause of death worldwide [@who_cvd_fs]. In response to this societal burden, scientific funding agencies and researchers have invested significant effort into understanding and controlling the functional behavior of heart cells. In particular, there has been a recent and growing focus on engineered heart cells and tissue to both better understand the complex interactions that drive disease, and to repair the damaged heart. An important component of these endeavors is the study of human induced pluripotent stem cell-derived cardiomyocytes (hiPSC-CMs), cells sampled non-invasively from living humans, transformed into stem cells, and subsequently differentiated into cardiomyocytes, i.e., cardiac muscle cells. These cardiomyocytes are composed of sarcomeres, sub-cellular contractile units, that can be fluorescently labeled and visualized via z-disc proteins (see \autoref{fig:intro}). One major challenge in studying hiPSC-CMs in this context is that the immaturity and structural nonlinearities of hiPSC-CMs (i.e., disordered sarcomere chain structure in comparison to the almost crystalline sarcomere structure of mature cardiomyocytes) causes significant complications for performing consistent analysis of their functional contractile characteristics. And, though multiple methods have recently been developed for analyzing images of hiPSC-CMs [@morris2020striated; @sutcliffe2018high; @pasqualini2015structural; @doi:10.1161/CIRCRESAHA.116.310363; @pasqualin2016sarcoptim; @TELLEY2006514], few are suitable for analyzing the asynchronous contractile behavior of beating cells [@toepfer2019sarctrack]. In our previous publication, we introduced a novel computational framework (`SarcGraph`) to perform this task, directly compared it to other methods, and demonstrated its state of the art efficacy and novel functionalities [@zhao2021sarc].

![An example frame of a beating hiPSC-CM movie with a schematic illustrations of labeled z-discs and sarcomeres.\label{fig:intro}](figures/intro.png){width=60%}

Here we introduce an open-source Python package to make the `SarcGraph` approach to performing automated quantitative analysis of information-rich movies of fluorescently labeled beating hiPSC-CMs broadly accessible. In contrast to the original version of the software released in conjunction with our previous publication [@zhao2021sarc], the updated version is better designed, more efficient, and significantly more user-friendly. In addition, there are multiple methodological and implementation updates to improve overall performance. In brief, our framework includes tools to automatically detect and track z-discs and sarcomeres in movies of beating cells, and to recover sarcomere-scale and cardiomyocyte-scale functional behavior metrics. In addition, SarcGraph includes additional functions to perform post-processing spatio-temporal analysis and data visualization to help extract rich biological information. With this further development of SarcGraph, we aim to make automated quantitative analysis of hiPSC-CMs more accessible to the broader research community. To make our framework more accessible, SarcGraph is capable of running various video and image formats and textures out of the box. And, SarcGraph is readily customizable and adaptable by the user. In addition to the ongoing maintenance of SarcGraph by our group, we expect a continuous contribution by other researchers to improve the software. 

# SarcGraph in Action

SarcGraph provides users with tools to process images and videos of hiPSC-CMs for z-disc and sarcomere segmentation and tracking. \autoref{fig:visualization} demonstrates tracked z-discs and sarcomeres in the first frames of two videos of beating cells. For z-disc and sarcomere segmentation, we build on our previously described work [@zhao2021sarc] and also implement a more efficient sarcomere detection algorithm, detailed in the [Appendix](#appendix). For tracking, we build on a previously developed Python package for particle tracking TrackPy [@allan_daniel_b_2023_7670439]. 

![This figure showcases segmented z-discs and detected sarcomeres in the first frame of two beating hiPSC-CM movies. Detected sarcomeres are marked by red stars, while blue contours indicate z-discs. Visualizations were created using the `SarcGraphTools.Visualization.zdiscs_and_sarcs()` function built into the SarcGraph package.\label{fig:visualization}](figures/sample_vis.png){width=75%}

After initial segmentation and tracking, SarcGraph offers several post-processing analysis and visualization functions. 
\autoref{fig:features} showcases some of these features. Notably, there are multiple demos and tutorials that further explain these capabilities in the SarcGraph repository.

![These plots illustrate some of the key post-processing features of the SarcGraph package on a sample video of hiPSC-CMs: a) a spatial graph visualization of segmented z-discs and sarcomeres; b) the average normalized sarcomere length; and c) the components of the approximate deformation gradient.\label{fig:features}](figures/features.png){width=95%}

To validate our methods and ensure correct implementation, we generated challenging synthetic videos with characteristics similar to beating hiPSC-CMs. We used these videos to evaluate the sarcomere detection algorithm by comparing recovered metrics to their known ground truth. \autoref{fig:validation} shows this process for one of many tested validation examples.

![These plots show the performance of SarcGraph on a synthetically generated sample with known ground truth behavior: a) the first frame of the video with z-discs marked in blue and tracked sarcomeres marked by red stars; b) the average normalized sarcomere length with good agreement between recovered and ground truth behavior; and c) principal stretches from the approximate deformation gradient with good agreement between recovered and ground truth behavior. As a brief note, the updated version of SarcGraph is better able to recover the ground truth from this synthetic example in comparison to the previous version of the framework [@zhao2021sarc].\label{fig:validation}](figures/validation.png){width=95%}

# Acknowledgements

This work was made possible through the support of the Boston University David R. Dalton Career Development Professorship, the Boston University Hariri Institute Junior Faculty Fellowship, the National Science Foundation CELL-MET ERC EEC-1647837, and the American Heart Association Career Development Award 856354. This support is gratefully acknowledged.

# Appendix

## Sarcomere Detection Algorithm

In this implementation of SarcGraph, we introduce a novel and further customizable algorithm for sarcomere detection, which replaces the ghost points-based approach used in our previous work [@zhao2021sarc]. Our new method works by first constructing a spatial graph from the segmented z-discs in a given frame of the movie. In this initial spatial graph, each node represents a z-disc and each edge represents a potential sarcomere location where each node is initially connected to its three nearest neighbors. To score each edge, we define a function $\mathcal{S}_i$ that takes into account the length of the edge $l_i$, the angle between the edge and its neighboring edges $\theta_{i,j}$, the maximum allowable length for a sarcomere $l_{max}$, an initial guess for the average length of sarcomeres $l_{avg}$, and three user-defined functions $f_k$.

The scoring function is defined as:

$$ \mathcal{S}_i = \mathbb{1}_{l_i < l_{max}} \Big(\max_{j\in\{1,\dots,n_i\}} \big(c_1 \times f_1(\theta_{i,j}) + c_2 \times f_2(l_i, l_j)\big) + c_3 \times f_3(l_i, l_{avg})\Big) $$

where $n_i$ is the number of edges connected to edge $i$ and,

$$ f_1(\theta_{i,j}) = \mathbb{1}_{\theta_{i,j} \leq \pi/2} \big( 1 - \theta_{i,j} / (\pi/2) \big) ^ 2 $$
$$ f_2(l_i, l_j) = (1 + |l_j - l_i| / l_i) ^ {-1} $$
$$ f_3(l_i) = e ^ {-\pi(1 - l_i / l_{avg}) ^ 2} $$

and $c_1$, $c_2$, and $c_3$ are user-defined constants that weigh the importance of each link feature. The default values of $c_k$ and functions for $f_k$ in SarcGraph were selected to produce accurate results on all of the samples that we tested (both real and synthetic data) without an example-specific parameter tuning. However, it is possible to customize these values and functions to suit different scenarios where the defaults may not perform optimally. In future development of SarcGraph, automated parameter tuning will be implemented as needed. 

To prune this spatial graph such that only accurately detected sarcomeres remain, pruning follows three rules: (1) each node can have at most two edges, (2) the angle between edges attached to a node must be greater than a threshold $\theta_{max}$, and (3) the edge score must be greater than a threshold $s_{max}$. The new algorithm offers improved effectiveness and flexibility over the previous method and is a key component of the new SarcGraph package's sarcomere detection capabilities.

![In the initial graph (left) the edges show the potential sarcomeres. After scoring and pruning, the remaining edges that represent the detected sarcomeres are shown in blue lines in the pruned graph (right). The edges that were deleted during pruning are shown in red dashed lines.\label{fig:appendix}](figures/appendix.png){width=95%}

# References
