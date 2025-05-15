''' This file trains an autoencoder on only gram matrix data now.
(...continue later...)'''


import io
import os
import time
import csv
import scipy.io
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

# This file trains an autoencoder on gram matrix data.


# Here are variables one might want to change

# Number of files to import when creating data shards
nfiles = 'all'
# Number of solutions to use from each file
# (set this to 'all' to use all data)
num_data = 'all'
# Choose data from files beginning at, (shift_data - 1)*num_data
# (this is ignored if num_data = 'all')
shift_data = 1
# Train for this many epochs
max_epochs = 10 # prev: 10, 20
# Set batch size (prev: 12000, 8000*, 4000)
batches = 11000
bpk = 0

# To continue training an existing model each epoch change to False
overwrite = False

# Which activation function to use
af = 'relu'
# Relative weighting for the losses
# clique_loss_weight = 5
gm_loss_weight = 1

# Folder to save network snapshots + images
folder = 'gram-classifier'
version = '1'
# clique_file_dir = '../VertexLabeledCliques/AllGroups/Vertices/'
gram_file_dir = '../VertexLabeledCliques/AllGroups/GramMatrices/'

# gram_features from gram_features_max
with open('gram-maxfeatures.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file, delimiter=' ')
    feature_data = list(reader)
gm_features = len(feature_data)
print("Number of Gram Features: " + str(gm_features))

# Folders, split into (nsplit) segments, where data-shards are stored
#     If training ends early at some incomplete shard index (n) find 
#     last printed "Data shard (n)/(nsplit)" from std.out file and 
#     change, shard_id = 0 --> shard_id = (n)-1 
# To generate new data-shards replace with: shard_id = 'new' (last val: 44)
shard_id = 0
init_shard_id = 0
nsplit = 98 # prev: 86, 70
# clq_shard_path = 'data-shards/clique-shards/'
gm_shard_path = 'data-shards/class-gram-shards/'
# if not os.path.exists(clq_shard_path):
#     os.makedirs(clq_shard_path)
if not os.path.exists(folder):
    os.makedirs(folder)
if not os.path.exists(gm_shard_path):
    os.makedirs(gm_shard_path)

# Set AutoShard Policy to Data for MirroredStrategy (multi-GPU)
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = \
    tf.data.experimental.AutoShardPolicy.DATA

# Folder for checkpoints used for identifying clusters (TODO need to fix)
checkpoint_dir = "./ckpt"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Cluster coordinate save directory
clusters_dir = 'clusters'
predict_dir = 'clusters/class-labels/gram-shards_predict/ae-03/'
if not os.path.exists(clusters_dir):
    os.makedirs(clusters_dir)
if not os.path.exists(predict_dir):
    os.makedirs(predict_dir)


def fig2img(fig):
    '''Function for converting Matplotlib figures to PIL images'''
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def custom_import(fnames, nelem=1000, pnid=1):
    '''Import data in segments or all at once by setting nelem='all'.'''
    temp = []
    key_names = list(scipy.io.loadmat(fnames[0]).keys())[-1]
    for fln in fnames:
        if nelem != 'all':
            temp.append(
                scipy.io.loadmat(fln)[key_names][(pnid-1)*nelem:pnid*nelem])
        else:
            temp.append(scipy.io.loadmat(fln)[key_names])
    return scipy.sparse.vstack(temp)


def gram_make_or_restore_model(owrite=False, itid=0, sv1='7e', sv2='20e'):
    ''' Either restore the latest model, or create a fresh one if no
    checkpoint is available.'''
    checkpoints = [folder + "/" + name for name in next(os.walk(folder))[1]]
    if checkpoints and not owrite:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        save_flag = latest_checkpoint.split('_')[-1]
        if save_flag in (sv1, sv2):
            os.system("cp -r " + latest_checkpoint + " " + checkpoint_dir + "/"
                      + folder + "_" + save_flag + "_iteration_" + str(itid))
        if not model:
            print("Restoring from", latest_checkpoint)
            latest_model = tf.keras.models.load_model(latest_checkpoint)

            gm_feat = latest_model.get_layer('gm_input').input_shape[-1]

            gm_in = tf.keras.layers.Input(shape=(gm_feat[-1]))
            encoder = latest_model.get_layer('l1')(gm_in)
            encoder = latest_model.get_layer('l2')(encoder)
            encoder = latest_model.get_layer('l3')(encoder)
            encoder = latest_model.get_layer('l4')(encoder)
            # encoder = latest_model.get_layer('l5')(encoder)
            latent_lyr = latest_model.get_layer('latent_layer')(encoder)

            encode_gif = tf.keras.Model(
                inputs=gm_in,
                outputs=latent_lyr
            )
            return latest_model, encode_gif
        print("Continuing with training...")
        encode_gif = tf.keras.Model(
            inputs=model.inputs,
            outputs=model.get_layer('latent_layer').output
        )
        return model, encode_gif
    print("Creating a new model")
    return gram_anomalyfree_autoencoder(gm_features)


def class_make_or_restore_model(owrite=False, itid=0, sv1='7e', sv2='20e'):
    ''' Either restore the latest model, or create a fresh one if no
    checkpoint is available.'''
    checkpoints = [folder + "/" + name for name in next(os.walk(folder))[1]]
    if checkpoints and not model:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print("Restoring from", latest_checkpoint)
        latest_model = tf.keras.models.load_model(latest_checkpoint)
        return latest_model


def gram_custom_load_dataset(gm_path, start=0, nshards=nsplit):
    '''Loads generated datasets for gram sparse tensors.'''
    gm_shard_files = [gm_path+fname for
                      fname in sorted(os.listdir(gm_path))[:nshards]]
    print("Loading Gram Datashards: " + gm_shard_files[start] + 
          " through " + gm_shard_files[nshards-1])

    gm_data = tf.data.Dataset.load(gm_shard_files[start])
    for dat in gm_shard_files[start+1:]:
        temp = tf.data.Dataset.load(dat)
        gm_data = gm_data.concatenate(temp)
    return gm_data


def sparse_to_dense(gm_dat):
    '''Function used for mapping across datasets.'''
    tensors = tf.data.Dataset.from_tensor_slices(
        tf.sparse.to_dense(tf.sparse.reorder(gm_dat)),
        name='gm_input')
    return tf.data.Dataset.zip((tensors, tensors)).batch(batches)


if shard_id == 'new':
    print("Generating Data Shards...")
    # Import max features for one-hot encoding clique vectors
    # max_features = np.genfromtxt('clique-maxfeatures.csv', dtype='i')

    # Data files
    # clique_files = []
    # gram_files = []

    # This populates the file lists with all 32 chunks of the data.
    # To use only specific files, comment out these loops and do manually.
    # for i in range(32):
    #     clique_files.append(clique_file_dir +
    #     		    'AllGrp-VLabel-Randomized_Part-' +
    #                     f'{i:02d}' + '.mat')
    # for i in range(32):
    #     gram_files.append(gram_file_dir +
    #                       'AllGrp-Gram-Randomized_Part-' +
    #                       f'{i:02d}' + '.mat')
    gram_files = [gram_file_dir + fname for
                      fname in sorted(os.listdir(gram_file_dir)) 
                      if '.mat' in fname]
    if nfiles != 'all': gram_files = gram_files[:nfiles]

    # clique_data = []
    gram_data = []

    # time_load = time.time()

    # clique_data = custom_import(clique_files, num_data, shift_data)

    # print(clique_data.shape)
    # print("Clique files loaded\n--- %s seconds ---" % (time.time()-
    #                                                    time_load))
    time_load = time.time()

    # One-hot encode clique data in sparse matrix format
    # clique_data_onehot = []
    # temp = []
    # for cl in range(len(max_features)):
    #     temp.append(scipy.sparse.csc_matrix(
    #             (np.ones(clique_data[:, cl].getnnz()),
    #              (clique_data[:, cl].indices, clique_data[:, cl].data-1)),
    #             shape=(clique_data.shape[0], max_features[cl]), dtype='int'))

    # clique_data_onehot = scipy.sparse.hstack(temp).tocoo()
    # temp = []
    # del clique_data

    # print(clique_data_onehot.shape)
    # print("One-hot Encoding Done!\n--- %s seconds ---" % (time.time()-
    #                                                       time_load))
    # time_load = time.time()

    gram_data = custom_import(gram_files, num_data, shift_data)

    # Load in maximum and minimum Gram matrix features
    gram_features_max = []
    gram_features_min = []
    with open('gram-maxfeatures.csv', 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=' ')
        feature_data = list(reader)
    for fe in feature_data:
        gram_features_max.append(int(fe[0]))
    with open('gram-minfeatures.csv', 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=' ')
        feature_data = list(reader)
    for fe in feature_data:
        gram_features_min.append(int(fe[0]))

    # SparseTensor with Gram normalization factors
    gmax_m_gmin = np.array(gram_features_max)-np.array(gram_features_min)
    renorm_factors = scipy.sparse.diags([1/gmax_m_gmin[id]
                                         if gmax_m_gmin[id] != 0 else 0
                                         for id in range(len(
                                                         gram_features_max))])
    del gmax_m_gmin

    # Normalize the Gram matrix data to be in unit range
    gram_data_norm = scipy.sparse.coo_matrix(gram_data @ renorm_factors)
    del gram_data, renorm_factors, gram_features_max, gram_features_min

    print("Gram Matrices Prepared\n--- %s seconds ---" % (time.time()-
                                                          time_load))
    time_load = time.time()

    # Size of clique vector feature space
    # features = clique_data_onehot.shape[1]
    # Size of Gram matrix feature space
    gm_features = gram_data_norm.shape[1]

    # Converting coordinates from scipy's coo_matrix to tf's SparseTensor form
    # onehot_coords = np.array([clique_data_onehot.row, 
    #                           clique_data_onehot.col]).T
    renorm_coords = np.array([gram_data_norm.row, gram_data_norm.col]).T

    # spr_clq = tf.sparse.SparseTensor(onehot_coords, 
    #                                      clique_data_onehot.data,
    #                                      clique_data_onehot.shape)
    spr_grams = tf.sparse.SparseTensor(renorm_coords, gram_data_norm.data,
                                       gram_data_norm.shape)
    # del clique_data_onehot, gram_data_norm
    del gram_data_norm

    # Save to Dataset shard files
    # itt = 0
    # for spclq in tf.sparse.split(sp_input=spr_clq, num_split=nsplit, axis=0):
    #     tf.data.Dataset.from_tensors(spclq).save(
    #         clq_shard_path+"/part_{:0{wt}}".format(itt, wt=len(str(nsplit))))
    #     itt = itt+1
    itt = 0
    for spgm in tf.sparse.split(sp_input=spr_grams, num_split=nsplit, axis=0):
        tf.data.Dataset.from_tensors(spgm).save(
            gm_shard_path+"/part_{:0{wt}}".format(itt, wt=len(str(nsplit))))
        itt = itt+1
    # del onehot_coords, renorm_coords, spr_cliques, spr_grams
    del renorm_coords, spr_grams

# Initializing model
model = None

# Count of Labels
nzeros = []
nones = []

# model, encoder_gif = gram_make_or_restore_model(overwrite, shard_id)
model = class_make_or_restore_model(overwrite, shard_id)

# Uncomment this if you want it to print the architecture
model.summary()

dataset = gram_custom_load_dataset(gm_shard_path, init_shard_id)
shard_id = init_shard_id

for shard in dataset.map(sparse_to_dense): 
    # shard = shard.cache().with_options(options)
    shard = shard.with_options(options)
    shard_id = shard_id + 1
    print('Data Shard ' + str(shard_id) + '/' + str(nsplit))

    # Predict the latent space values
    output = model.predict(shard, verbose=0)
    tempshape = output.shape
    # vals, counts = np.unique(output, return_counts=True)
    # nzeros.append(counts[0])
    # nones.append(counts[1])
    print('output shape: ' + str(tempshape))
    # print('number of predictions with ' + str(vals[0]) + ': ' + str(counts[0]))
    # print('number of predictions with ' + str(vals[1]) + ': ' + str(counts[1]))
    # print('current total distribution of 0 and 1 labels: ' + str(np.sum(nzeros)/tempshape[0]) + str(np.sum(nones)/tempshape[1]))
 
    # Save predicted latent space coordinates
    np.save(predict_dir + 'predict-class_' + 
            'shard-{:0{wdt}}.npy'.format(shard_id-1, wdt=len(str(nsplit))), 
            output)

    # # Predict the latent space values
    # pred = model.predict(shard, verbose=0)
    # loss = model.loss(shard, pred)
    # print('output shape: ' + str(pred.shape))
 
    # # Save predicted latent space coordinates
    # np.save(predict_dir + 'loss-predict_' + 
    #         'shard-{:0{wdt}}.npy'.format(shard_id-1, wdt=len(str(nsplit))), 
    #         output)
