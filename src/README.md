# Source Code
Code used for training was written to run on the High Performance Computing clusters (HPRC) at Texas A&M University so it will need to be modified (i.e. save file locations, reference folders, etc.).

## Autoencoder
The python file '*gram-train.py*' must be modified before running on your local machine. It contains large chunks of code which have been commented out based on changing project goals. These are left in for a possible future project involving the clique structure of the model.

This program trains the model on multiple GPUs at once which requires saving the data in separate '*data shards*'. The first time the program runs it will need to create these shards which are saved in a separate directory which it will create. An additional directory will be created '*../models/gram-autoencoder*' where the model will be stored.

**Requires:** '*gram-maxfeatures.csv*', '*gram-minfeatures.csv*', and '*GramMatrices/*' 

**Note:** Data Shards only need to be created once, unless you wish to change how the data is imported during training.

## Classifiers
Python code '*train-classifier.py*' uses the normalized Gram matrices of pre-labeled models. This data is found in the subfolder '*pre-labeled/*' under the associated classifier. This code also must be modified before running on a local machine to specify which classifier to train and where it should be saved.

The file '*class-predict-label.py*' imports the trained classifier and uses the Data Shards (*created by the autoencoder training program*) and saves the resulting predictions in a separate folder. This must be modified for your local machine.

## Clustering
This folder contains the following programs:

* '*gram-predict-coords.py*' predicts the latent layer coordinates from the autoencoder for each model.)
* '*cluster-models.py*' and '*cluster-models-lc01.py*' imports the predicted latent layer coodinates and performs the fastHDBSCAN clustering algorithm.
