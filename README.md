# LANL-earthquake-prediction-project

May 13th 2019
Author: Praveer Nidamaluri

Udacity ML Nanodegree Capstone Project - LANL Earthquake Prediction


README - Project Files Description
-----------------------------------
Please read the report first for an understanding of the project.

======================================================================

Files description:
-------------------

"proposal.pdf":
Capstone project proposal

"Capstone Project Report.pdf":
Report write-up

"Benchmark Model_fset1/3/4.ipynb":
Benchmark model implementation notebooks, with various feature sets

"Data Exploration for Project Proposal - ... .ipynb":
Notebooks that document data exploration. Only the 
"Data Exploration for Project Proposal.ipynb" notebook contains results 
shown in the report.

"Explore Labquake Regions of Dataset.ipynb":
Notebook that plots neighborhoods around the labquakes in the dataset.
For exploration only, no results shown in the report.

"gen_features.py":
Main helper module with custom functions used in the various project jupyter notebooks

"Linear Regression_fset3.ipynb":
Notebook with implementation of linear model for prediction of the 'time_to_failure'.

"SVR_fset4.ipynb":
Notebook with implmentation of support vector regressor for prediction of the 'time_to_failure'.

"Catboost_fset5.ipynb":
Notebook with implementation of the Catboost gradient boosting regressor for prediction
of the 'time_to_failure'

========================================================================

Folder description:
--------------------

"Data":
Should contain the training data. This must be downloaded from:
https://www.kaggle.com/c/LANL-Earthquake-Prediction/data

The folder will contain a 9+gb csv file and separate 'test' folder. 
The "Data Exploration for Project Proposal.ipynb" notebook should then be used. 
The second cell of the notebook splits up the large csv file into 140 more manageable chunks.
The other notebooks operate on the training chunks.

"Features":
This folder contains csv files with features generated in the various notebooks.

"Submissions":
Any submission csv files for the Kaggle competition are stored in this folder.

"temp":
This is a folder used to store temporary csv files during data processing by the 
other notebooks.
