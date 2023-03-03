.. currentmodule:: sarcgraph.sg

.. _sg.sarcgraph:

*******************************************
Detection and Tracking (:class:`SarcGraph`)
*******************************************

Provides tools to detect and track zdiscs and sarcomeres.

Parent Class
============

.. autosummary::
   :toctree: generated/

   ~SarcGraph

Methods
=======

.. autosummary::
   :toctree: generated/

   ~SarcGraph.zdisc_segmentation
   ~SarcGraph.zdisc_tracking
   ~SarcGraph.sarcomere_detection

Private Methods
===============

.. autosummary::
   :toctree: generated/

   ~SarcGraph._data_loader
   ~SarcGraph._to_gray
   ~SarcGraph._save_numpy
   ~SarcGraph._save_dataframe
   ~SarcGraph._filter_frames
   ~SarcGraph._process_input
   ~SarcGraph._detect_contours
   ~SarcGraph._process_contour
   ~SarcGraph._zdiscs_to_pandas
   ~SarcGraph._merge_tracked_zdiscs
   ~SarcGraph._zdisc_to_graph
   ~SarcGraph._score_graph
   ~SarcGraph._prune_graph