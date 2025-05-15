''' This file predicts the coordinates of each model and saves them
to '../clusters/model-coords/gram-shards_predict/'
Dependent Files:
	../data-shards/gram-shards/* 
	GramMatrices/*.mtx
	../gram-maxfeatures.csv
	../gram-minfeatures.csv

Commented sections of code were from testing or older versions
of the program. 
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
folder = '../models/gram-autoencoder'
version = '2'
# clique_file_dir = '../VertexLabeledCliques/AllGroups/Vertices/'
gram_file_dir = '../GramMatrices/'

# gram_features from gram_features_max
with open('../gram-maxfeatures.csv', 'r', encoding='utf-8') as file:
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
gm_shard_path = '../data-shards/gram-shards/'
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
# checkpoint_dir = "./ckpt"
# if not os.path.exists(checkpoint_dir):
#     os.makedirs(checkpoint_dir)

# Cluster coordinate save directory
clusters_dir = '../clusters'
predict_dir = '../clusters/model-coords/gram-shards_predict/'
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


# Initializing model
model = None

model, encoder_gif = gram_make_or_restore_model(overwrite, shard_id)

# Uncomment this if you want it to print the architecture
model.summary()

dataset = gram_custom_load_dataset(gm_shard_path, shard_id)

for shard in dataset.map(sparse_to_dense): 
    # shard = shard.cache().with_options(options)
    shard = shard.with_options(options)
    shard_id = shard_id + 1
    print('Data Shard ' + str(shard_id) + '/' + str(nsplit))

    # Predict the latent space values
    output = encoder_gif.predict(shard, verbose=0)
    print('output shape: ' + str(output.shape))
 
    # Save predicted latent space coordinates
    np.save(predict_dir + 'predict-coords_' + 
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
