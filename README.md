# Machine Learning the 6d SUGRA Landscape

Machine Learning has been a useful tool for analysing large, and often complex, sets of data. They can reveal patterns hidden within the data providing a deeper level of understanding and information about the data. In [[2505.xxxxx](arxiv.com/hep-th/)] we apply these techniques to the set of admissable and anomaly-free 6-dimensional supergravity models with 8 supercharges generated in [[1]](#1). 

We find that both supervised and unsupervised training techniques are surprisingly effective at classification and auto-clustering the highly complex features associated with this patch of the landscape. Provided in this repository are the machine learning models and results referenced in our associated paper. Results from our analysis can be found in the [companion website](https://nait400.github.io/ML-6d-sugra-landscape/), also linked below. 

## Contents
### Machine Learning Models
* [Autoencoder](https://github.com/nait400/ML-6d-sugra-landscape/tree/ed5ff3bac3702e56fd47acdb9a4e04e9b0728d99/models/autoencoder)
* [Classifier-0](https://github.com/nait400/ML-6d-sugra-landscape/tree/2cf57d227ee385b5f6d4dca624d816ac2cc5e431/models/classifier-0)
* [Classifier-1](https://github.com/nait400/ML-6d-sugra-landscape/tree/2cf57d227ee385b5f6d4dca624d816ac2cc5e431/models/classifier-1)
### Data (format: .mx and .wdx)
* [Clustered Model Data (Autoencoder)](https://github.com/nait400/ML-6d-sugra-landscape/tree/2cf57d227ee385b5f6d4dca624d816ac2cc5e431/data/clusters)
* [Consistent Models (Classifier-0)](https://github.com/nait400/ML-6d-sugra-landscape/tree/2cf57d227ee385b5f6d4dca624d816ac2cc5e431/data/classifier-0)
* [Inconsistent Models (Classifier-1)](https://github.com/nait400/ML-6d-sugra-landscape/tree/2cf57d227ee385b5f6d4dca624d816ac2cc5e431/data/classifier-1)
### Training Code
**Note:** Code used for training was written to run on the High Performance Computing clusters (HPRC) at Texas A&M University so it will need to be modified (i.e. save file locations, reference folders, etc.). For convenience we provide the set of [Gram Matrices](https://github.com/nait400/ML-6d-sugra-landscape/tree/2cf57d227ee385b5f6d4dca624d816ac2cc5e431/src/GramMatrices) (in their original randomized order) that were used to train the autoencoder.
* [Autoencoder Training](https://github.com/nait400/ML-6d-sugra-landscape/tree/2cf57d227ee385b5f6d4dca624d816ac2cc5e431/src/autoencoder) (requires gram-maxfeatures.csv and gram-minfeatures.csv)
* [Classifiers](https://github.com/nait400/ML-6d-sugra-landscape/tree/2cf57d227ee385b5f6d4dca624d816ac2cc5e431/src/classifiers) (requires gram-maxfeatures.csv and gram-minfeatures.csv)
* [Clustering and Plot Generation](https://github.com/nait400/ML-6d-sugra-landscape/tree/2cf57d227ee385b5f6d4dca624d816ac2cc5e431/src/clustering)
#### Dependencies
* Python (>=3.8)
* Numpy (1.22<= ver <1.27)
* matplotlib (3.7.2)
* TensorFlow (2.13.0)
* numba (0.58.1)
* scikit-learn (1.3.1)
** joblib (>= 1.1.1)
** threadpoolctl (>= 2.0.0)


## Access to Analysis Results
Our results from the autoencoder can all be found in the companion GitHub page [Machine Learning the 6d SUGRA Landscape](https://nait400.github.io/ML-6d-sugra-landscape/). To aid with navigation we provide a clickable image with links to the associated clusters as well as a "Comparison View" tool where separate clusters and/or multiple instances of the same cluster can be viewed one a single webpage side-by-side.

## References
<a id="1">[1]</a>
Y. Hamada and G. J. Loges,
"*Towards a complete classification of 6D supergravities*,"
JHEP **02**, 095 (2024)
DOI:[10.1007/JHEP02(2024)095](https://doi.org/10.1007/JHEP02(2024)095)
[[arXiv:2311.00868 [hep-th]](https://arxiv.org/abs/)].
