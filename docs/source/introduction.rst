Introduction
============

**SarcGraph** is developed for automatic detection, tracking and analysis of
zdiscs and sarcomeres in movies of beating **human induced pluripotent stem
cell-derived cardiomyocytes (hiPSC-CMs)**.

Overview
--------

**SarcGraph** was initially introduced in
`Sarc-Graph: Automated segmentation, tracking, and analysis of sarcomeres in
hiPSC-derived cardiomyocytes <https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009443>`_.
This package is created to make **SarcGraph** more accessible to the broader
research community.

Background
----------

Key Features
------------

**SarcGraph** includes functions to detect and track z-discs and sarcomeres in
beating cells, construct spatial graphs of z-discs and sarcomeres for network
distance computation, and perform automated spatiotemporal analysis and data
visualization.

Explain the problem your package solves: Next, explain the problem or pain point
that your package solves. This should be a few sentences that describe the
challenge or issue that users might be facing, and how your package provides a solution.

Provide some background information: After explaining the problem, provide some
background information on the context and domain of your package. This could
include information on the industry or field where your package is most useful,
or any relevant trends or developments that are driving demand for your package.

Highlight key features and benefits: After providing context, highlight the key
features and benefits of your package. This could include information on performance,
ease of use, flexibility, or any other features that make your package stand out from
others in the same category.

Summarize the documentation: Finally, provide a brief summary of what readers can
expect to find in the rest of your documentation. This could include information
on the different sections or topics covered, or a list of the key resources or
references that are included.


Introduction: Start with a brief introduction that explains what your package does, who it's intended for, and why someone might want to use it. This will help set the context for the rest of your documentation.

Getting started: After the introduction, provide a section that walks users through a simple example of how to use your package. This could include code snippets or screenshots to help illustrate the steps.

API reference: If your package has an API or public interface, provide documentation for each of the classes, functions, and methods that users can interact with. This section should be organized in a way that makes it easy for users to find the information they need.

Tutorials: In addition to a simple getting-started example, provide more detailed tutorials that walk users through common use cases for your package. This could include multi-step examples that build on the getting-started example, or more complex scenarios that require several components of your package to work together.

FAQ: Include a list of frequently asked questions and their answers. This can help users quickly find solutions to common issues or questions that they might encounter.

Contributing: If you're open to contributions from the community, provide information on how users can contribute to your package. This could include guidelines for submitting bug reports or pull requests, or instructions on how to set up a development environment.

License and attribution: Finally, include information on the license that your package is released under, as well as any attribution requirements or acknowledgements that you'd like users to include when they use your package. This can help ensure that your package is used ethically and responsibly.

Write your documentation: With Sphinx configured, you can start writing your documentation in reStructuredText format. You can create a new file for each section of your documentation, such as installation instructions, usage examples, and API references. Sphinx provides a rich set of markup language to create documentation with.

Generate HTML documentation: To generate HTML pages from your documentation files, run the following command in your terminal:

Add the HTML documentation to your package: You can add the HTML files to your package by creating a docs directory in the root of your package and copying the contents of the build/html directory to it.

Upload your package to PyPI: Once you've added the documentation to your package, you can upload it to PyPI using the twine command-line tool. Make sure to include the docs directory in your package when you create the distribution package.

View your documentation: After uploading your package to PyPI, your documentation will be available on the project's page on PyPI. You can also host your documentation on a separate website, such as Read the Docs.
