''' This file trains an autoencoder on clique + gram matrix data.
It reads in .mat data and one-hot-encodes it based on a maximum clique size
for a given clique vertex.
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

# Number of solutions to use from each file
# (set this to 'all' to use all data)
num_data = 'all'
# Choose data from files beginning at, (shift_data - 1)*num_data
# (this is ignored if num_data = 'all')
shift_data = 1
# Train for this many epochs
max_epochs = 20
# Set batch size
batches = 4000

# To continue training an existing model each epoch change to False
overwrite = False

# Which activation function to use
af = 'relu'
# Relative weighting for the losses
clique_loss_weight = 5
gm_loss_weight = 1

# Folder to save network snapshots + images
folder = 'autoencoder'
clique_file_dir = '../VertexLabeledCliques/AllGroups/Vertices/'
gram_file_dir = '../VertexLabeledCliques/AllGroups/GramMatrices/'
# clique_file_dir = 'clique-data/select-group/all-groups/vectors/'
# gram_file_dir = 'clique-data/select-group/all-groups/gram-matrices/'

# Folders, split into nsplit segments, where data-shards are stored
nsplit = 70
clq_shard_path = 'data-shards/clique-shards/'
gm_shard_path = 'data-shards/gram-shards/'
if not os.path.exists(clq_shard_path):
    os.makedirs(clq_shard_path)
if not os.path.exists(gm_shard_path):
    os.makedirs(gm_shard_path)

# Set AutoShard Policy to Data for MirroredStrategy (multi-GPU)
# options = tf.data.Options()
# options.experimental_distribute.auto_shard_policy = \
#     tf.data.experimental.AutoShardPolicy.DATA

# Folder for checkpoints used for identifying clusters
checkpoint_dir = "ckpt"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


def fig2img(fig):
    '''Function for converting Matplotlib figures to PIL images'''
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def make_or_restore_model(owrite=False, itid=0, sv1='7e', sv2='20e'):
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

            lyr_in = tf.keras.layers.Input(shape=(features))
            gm_in = tf.keras.layers.Input(shape=(gm_features))
            encoder = latest_model.get_layer('l1')(lyr_in)
            concat = tf.keras.layers.concatenate([encoder, gm_in])
            encoder = latest_model.get_layer('l2')(concat)
            encoder = latest_model.get_layer('l3')(encoder)
            latent_lyr = latest_model.get_layer('latent_layer')(encoder)

            encode_gif = tf.keras.Model(
                inputs=[lyr_in, gm_in],
                outputs=latent_lyr
            )
            return latest_model, encode_gif
        print("Continuing with training...")
        encode_gif = tf.keras.Model(
            inputs=model.inputs,
            outputs=model.get_layer('latent_layer').output
        )
        return model, encode_gif


def custom_load_dataset(clq_path, gm_path, nshards=nsplit):
    '''Loads generated datasets for clique and gram sparse tensors.'''
    clq_shard_files = [clq_path+fname for
                       fname in sorted(os.listdir(clq_path))[:nshards]]

    gm_shard_files = [gm_path+fname for
                      fname in sorted(os.listdir(gm_path))[:nshards]]

    clq_data = tf.data.Dataset.load(clq_shard_files[0])
    for dat in clq_shard_files[1:]:
        temp = tf.data.Dataset.load(dat)
        clq_data = clq_data.concatenate(temp)

    gm_data = tf.data.Dataset.load(gm_shard_files[0])
    for dat in gm_shard_files[1:]:
        temp = tf.data.Dataset.load(dat)
        gm_data = gm_data.concatenate(temp)
    return tf.data.Dataset.zip((clq_data, gm_data))


def sparse_to_dense(clq_dat, gm_dat):
    '''Function used for mapping across datasets.'''
    tensors = tf.data.Dataset.zip((
        tf.data.Dataset.from_tensor_slices(
            tf.sparse.to_dense(tf.sparse.reorder(clq_dat)),
            name='clique_input'),
        tf.data.Dataset.from_tensor_slices(
            tf.sparse.to_dense(tf.sparse.reorder(gm_dat)),
            name='gm_input')))
    return tf.data.Dataset.zip((tensors, tensors)).batch(batches)


def plot_latent_gif(epoch_id=1, itid=0, features=15041, gm_features=136):
    '''Plot latent layer of encoder after model training'''
    checkpoints = [[folder + "/" + name, name.split('_')[-2]] for name in
                   next(os.walk(folder))[1]]
    epoch_checkpoint = next(ckpoint[0] for ckpoint in
                            checkpoints if ckpoint[1] == str(epoch_id) + 'e')
    print("Restoring from", epoch_checkpoint)
    latest_model = tf.keras.models.load_model(epoch_checkpoint)

    lyr_in = tf.keras.layers.Input(shape=(features))
    gm_in = tf.keras.layers.Input(shape=(gm_features))
    encoder = latest_model.get_layer('l1')(lyr_in)
    concat = tf.keras.layers.concatenate([encoder, gm_in])
    encoder = latest_model.get_layer('l2')(concat)
    encoder = latest_model.get_layer('l3')(encoder)
    latent_lyr = latest_model.get_layer('latent_layer')(encoder)

    encode_gif = tf.keras.Model(
        inputs=[lyr_in, gm_in],
        outputs=latent_lyr
    )
    return latest_model, encode_gif


model = None
frames = []
epochs = []
clique_losses = []
gm_losses = []
shard_id = 0

# dataset = custom_load_dataset(clq_shard_path, gm_shard_path)

# model, encoder_gif = plot_latent_gif(20, shard_id)

cluster_dir = 'data-shards/cluster/'
coord_files = 'predicted-coords/predict-coords'

# for shard in dataset.map(sparse_to_dense):
for shard in range(66): 
    # shard = shard.cache().with_options(options)
    shard_id = shard_id + 1
    print('Data Shard ' + str(shard_id) + '/' + str(nsplit))

    # Predict the latent space values
    # output = encoder_gif.predict(shard, verbose=0)
    output = np.load(cluster_dir + coord_files +
            '_shard-{:0{wdt}}.npy'.format(shard_id,
                                      wdt=len(str(nsplit))))

    # Plot the latent space
    figg = plt.figure()
    plt.plot(output[:, 0], output[:, 1], 'b,')

    ImF = fig2img(figg)
    plt.close()
    frames.append(ImF)

    # Save a copy of the network (labeled by number of epochs)
    # model.save(folder + '/autoencoder_' + str(ep+1) + 'e')

# After training is complete, export the latent space plots as a gif
frames[0].save(folder + '/' + str(round(time.time())) + '.gif', save_all=True,
               append_images=frames[1:], loop=0)
