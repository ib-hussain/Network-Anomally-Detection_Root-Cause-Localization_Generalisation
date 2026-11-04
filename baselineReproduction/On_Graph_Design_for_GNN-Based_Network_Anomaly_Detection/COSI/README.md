

# Here is a short explanation for using COSI model for anomaly detection


Code implementation for : GNN graph structures in network anomaly detection (NOMS 2025)

Requirements for GDN :

    Python >= 3.6
    cuda == 10.2 (not working on cuda 12.x, if so, use cpu instead)
    Pytorch==1.5.1
    PyG: torch-geometric==1.5.0

  1) Short.ipynb is the file that is used to create the graph structure. Here we used a preprocessed dataset from the 5G3E dataset : https://github.com/cedric-cnam/5G3E-dataset
     Short Cisco and SWaT are two others file that are doing the same for CISCO and SWaT dataset

  3) Tools_first_step.py is a utilitary file used in the notebook, we can find all function that I am using here, with anterior version for some of them.
  
  4) Start.py is the main file for the GDN training algorithm from the code provided here : https://github.com/d-ailin/GDN. This version is not using bash for launching script.
  
  5) test.py is the main test file for testing the GDN
  
  6) Util is a set of files used from Start/test
  
  7) models regroup GDN,LSTM-AE and GDN utilitary file
  
  8) plot.ipynb is the file used for ploting 3D figure in the article

Code implementation for : journal extension

Important part of the code was developped in this repository : https://github.com/GraGODs/GraGOD

please follow the project initialization process for using gradgod.

