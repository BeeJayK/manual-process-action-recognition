# Overview
This repository contains code developed for the paper "Proving the Potential of Skeleton Based Action Recognition to Automate the Analyses of Manual Processes".

Scripts that contain major functionlaties are showcased under '_0-code_showcases'.

Note: Since the code was sold, it can unfortunately not be shown in full. Especially the scripts "custom layers" and "mt_hlps" were thinned out by half.

Desciption of the scripts showcased:
- Conv_1D_Basic.py ------------ expamle of the Pipeline for the Conv-1d Models
- Create_Train_Data.py -------- creation of train-ready-data from raw video inputs
- custom_layers.py ------------ implementations of the (ML) preprocessing pipeline via custom layers
- Model_Analyzer.py ----------- automates the plot creations, metrics analysations etc.
- motion_tracking_helpers.py -- general helper functions
- mt_neural_helpers.py -------- DataBuilder, Inhibiting functions, Callbacks and related functions
- mt_plot_helpers.py ---------- custom plot functions
- VisT.py --------------------- used to analyse footage with given tf models


# Introduction
The work was embedded at the Digital Capibility Center (DCC) Aachen. The DCC is a cooperation between an institut of the RWTH Aachen and McKinsey, with the goal to bring digitalisation and AI into the production industry.

The work topic to be solved with the thesis is the classification of motion classes, performed by a worker, during the packaging process in a production line, based on a video stream.

The stream is processed by two neural networks, of which the first extracts the handcoordinates from images (using google's mediapipe) and the second takes these coordinate representations along with aditional information and predicts the current motion class.

A visualized models output look like:
<img width="1347" alt="github-repo-main-visualization" src="https://user-images.githubusercontent.com/75037677/162510608-eb3cb1ed-db22-4f67-b8f6-1cba1971a2a6.png">



Based on reliable motion class predictions (action recognition), one can easily develop toplevel routines, that can solve tasks which are currently time consuming, complicated and expensive. These tasks include (dynamic) cycle time calculations, tracking of station utilization, worker guidance, on-the-fly quality control and many more

Developing the second net is the main subject of the presented work. The work included everything from setting up an efficient training environment, write robust input pipelines and workflows, explore different architectures, simulate different hardware parameters and find a reliable, generalising solution.

Nets are trained on a simulation-PC with a RTX3080Ti & 16 CPU-cores, wandb is used for sweeping different architectures and input pipelines/operations, models are shared between different agents.

The models are constructed using tensorflow and include custom layers for preprocessing.

The project is embedded in and interacts with a larger code-environment, that is not public.

The following provides an overview on the IT connections used for developement:
<img width="1244" alt="github-repo-it-connections" src="https://user-images.githubusercontent.com/75037677/162510194-474bff88-38c5-47fd-9762-10e9b4ea7880.png">

