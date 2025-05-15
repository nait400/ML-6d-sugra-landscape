'''
The following program will import the predicted coordinates of the 
autoencoder then using the HDBSCAN algorithm from scikit-learn it will
assign each model to a cluster with labels starting from 0 to n_clusters.

Models labeled with cluster_id = -1 are treated as noise and do not fall
into any of the non-negative clusters.
'''


import os
from sys import exit
import numpy as np
import numpy.random as rand
import scipy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
import fast_hdbscan
# from sklearn.cluster import HDBSCAN


model_dir = 'gram-ae/'
cluster_min_size = 80
cluster_select_eps = 0.12 #prev: 0.15, 0.18, 0.2, 0.12
num_samples = 5*10**5


def plot(X, labels, probabilities=None, parameters=None, ground_truth=False, ax=None):
    '''The following function was taken from the scikit-learn website, 
    "Demo of HDBSCAN clustering algorithm".'''
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    labels = labels if labels is not None else np.ones(X.shape[0])
    probabilities = probabilities if probabilities is not None else np.ones(X.shape[0])
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    # colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    colors1 = [plt.cm.tab20b(each) for each in rand.permutation(np.linspace(0, 1, int(np.floor(len(unique_labels)/2))))]
    colors2 = [plt.cm.tab20c(each) for each in rand.permutation(np.linspace(0, 1, int(np.ceil(len(unique_labels)/2))))]
    colors = np.vstack((colors1, colors2))
    # The probability of a point belonging to its labeled cluster determines
    # the size of its marker
    proba_map = {idx: probabilities[idx] for idx in range(len(labels))}
    cl_handles = []
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_index = np.where(labels == k)[0]
        cl_handles.append(mpatches.Patch(color=tuple(col), label='C-{:02n}'.format(k)))
        for ci in class_index:
            ax.plot(
                X[ci, 0],
                X[ci, 1],
                "x" if k == -1 else "o",
                markerfacecolor=tuple(col),
                markeredgecolor="k",
                markersize=4 if k == -1 else 1 + 5 * proba_map[ci],
            )
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    preamble = "True" if ground_truth else "Estimated"
    title = f"{preamble} number of clusters: {n_clusters_}"
    if parameters is not None:
        parameters_str = ", ".join(f"{k}={v}" for k, v in parameters.items())
        title += f" | {parameters_str}"
    ax.set_title(title, fontsize='large')
    ax.legend(handles=cl_handles, ncols=2)
    plt.tight_layout()


def gram_import(fnames):
    '''Import Gram Matrix data.'''
    temp = []
    split_loc = []
    sp_rows = 0
    key_names = list(scipy.io.loadmat(fnames[0]).keys())[-1]
    for fln in fnames:
        # temp.append(scipy.io.loadmat(fln)[key_names])
        sp_rows = sp_rows + scipy.io.loadmat(fln)[key_names].shape[0]
        split_loc.append(sp_rows)
    # return scipy.sparse.vstack(temp), split_loc
    return split_loc


# Find npy file names of predicted coordinates
cluster_dir = 'clusters/model-clusters/' + model_dir
cor_dir = 'clusters/model-coords/gram-shards_predict/'
files = [cor_dir + fnm for fnm in sorted(next(os.walk(cor_dir))[2]) 
         if '.npy' in fnm]
print('Files loaded in the following order: ')

# Load coordinate data into files
all_coords = []
for fname in files:
    print(fname)
    all_coords.append(np.load(fname, mmap_mode='r'))

# Find Gram matrix file names
gram_dir = '../VertexLabeledCliques/AllGroups/GramMatrices/'
gm_files = [gram_dir + fnm for fnm in sorted(next(os.walk(gram_dir))[2])]
# gm_data, partitions = gram_import(gm_files)
partitions = gram_import(gm_files)
print('Splitting data into ' + str(len(gm_files)) + 
      ' partitions at index locations: ')
print(partitions)

# Combine all coordinate data into a single numpy array
all_coords = np.vstack(all_coords)
print('Loaded data has final shape: ' + str(all_coords.shape))

#  with joblib.parallel_backend('loky', n_jobs=-1):
hdb = fast_hdbscan.HDBSCAN(min_cluster_size=cluster_min_size, 
                           cluster_selection_epsilon=cluster_select_eps, 
                          ).fit(all_coords)

concat_data = np.split(np.hstack((all_coords, 
                                  hdb.labels_.reshape((partitions[-1],1))
                                 )), partitions)
print('Partitioned data for matching with their associated Gram Matrices...')

print('Saving clustered data...')
for idx, data in enumerate(concat_data[:-1]):
    print('Saving part-{0:2} of data'.format(idx))
    np.save(cluster_dir + 'model-clusters_part-{:02}.npy'.format(idx), data) 

print('Data saved.\nPlotting clusters...')
fig, axis = plt.subplots(1, 1, figsize=(30,26), dpi=200)

idx = np.random.choice(np.arange(len(all_coords)), num_samples)

plot(all_coords[idx], hdb.labels_[idx], hdb.probabilities_[idx], ax=axis)
fig.savefig(cluster_dir + 'hdb_clustering_grams', dpi=fig.dpi)
print('Figure Saved!\nProgram Complete!')
