''' This file trains an autoencoder on only gram matrix data.
Dependent Files: 
	GramMatrices/*.mtx
	../gram-maxfeatures.csv
	../gram-minfeatures.csv

Commented sections of code were from testing or older versions
of the program. 
(...continue later...)'''


import io
import os
import sys
import time
import csv
import scipy.io
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

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
# (change to True for testing purposes)
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

# # Folder for checkpoints used for identifying clusters (TODO need to fix)
# checkpoint_dir = "./ckpt"
# if not os.path.exists(checkpoint_dir):
#     os.makedirs(checkpoint_dir)
 

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

# Old version for initializing model architecture
# def anomalyfree_autoencoder(ftrs, gm_ftrs):
#     '''Define the newtwork architecture'''
#     input_lyr = tf.keras.layers.Input(shape=(ftrs), name='clique_input')
#     encoder = tf.keras.layers.Dense(150, name='l1', activation=af)(input_lyr)
#     gm_input = tf.keras.layers.Input(shape=(gm_ftrs), name='gm_input')
#     conc = tf.keras.layers.concatenate([encoder, gm_input], name='concat')
#     encoder = tf.keras.layers.Dense(50, name='l2', activation=af)(conc)
#     encoder = tf.keras.layers.Dense(10, name='l3', activation=af)(encoder)
#     latent = tf.keras.layers.Dense(2, name='latent_layer')(encoder)
#     decoder = tf.keras.layers.Dense(10, name='l4', activation=af)(latent)
#     decoder = tf.keras.layers.Dense(50, name='l5', activation=af)(decoder)
#     gm_output = tf.keras.layers.Dense(gm_features, name='gm')(decoder)
#     decoder = tf.keras.layers.Dense(150, name='l6', activation=af)(decoder)
#     output = tf.keras.layers.Dense(features, name='clique')(decoder)
# 
#     model = tf.keras.Model(
#         inputs=[input_lyr, gm_input],
#         outputs=[output, gm_output]
#     )
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(),
#         loss=tf.keras.losses.MeanSquaredLogarithmicError(reduction='sum_over_batch_size'),
#         loss_weights=[clique_loss_weight, gm_loss_weight]
#     )
#     encode_gif = tf.keras.Model(
#         inputs=[input_lyr, gm_input],
#         outputs=latent
#     )
#     return model, encode_gif

# Old version for restoring or initializing models
# def make_or_restore_model(owrite=False, itid=0, sv1='7e', sv2='20e'):
#     ''' Either restore the latest model, or create a fresh one if no
#     checkpoint is available.'''
#     checkpoints = [folder + "/" + name for name in next(os.walk(folder))[1]]
#     if checkpoints and not owrite:
#         latest_checkpoint = max(checkpoints, key=os.path.getctime)
#         save_flag = latest_checkpoint.split('_')[-1]
#         if save_flag in (sv1, sv2):
#             os.system("cp -r " + latest_checkpoint + " " + checkpoint_dir + "/"
#                       + folder + "_" + save_flag + "_iteration_" + str(itid))
#         if not model:
#             print("Restoring from", latest_checkpoint)
#             latest_model = tf.keras.models.load_model(latest_checkpoint)
# 
#             feat = latest_model.get_layer('clique_input').input_shape[-1]
#             gm_feat = latest_model.get_layer('gm_input').input_shape[-1]
# 
#             lyr_in = tf.keras.layers.Input(shape=(feat[-1]))
#             gm_in = tf.keras.layers.Input(shape=(gm_feat[-1]))
#             encoder = latest_model.get_layer('l1')(lyr_in)
#             concat = tf.keras.layers.concatenate([encoder, gm_in])
#             encoder = latest_model.get_layer('l2')(concat)
#             encoder = latest_model.get_layer('l3')(encoder)
#             latent_lyr = latest_model.get_layer('latent_layer')(encoder)
# 
#             encode_gif = tf.keras.Model(
#                 inputs=[lyr_in, gm_in],
#                 outputs=latent_lyr
#             )
#             return latest_model, encode_gif
#         print("Continuing with training...")
#         encode_gif = tf.keras.Model(
#             inputs=model.inputs,
#             outputs=model.get_layer('latent_layer').output
#         )
#         return model, encode_gif
#     print("Creating a new model")
#     return anomalyfree_autoencoder(features, gm_features)


def gram_anomalyfree_autoencoder(gm_ftrs):
    '''Define the newtwork architecture'''
    gm_input = tf.keras.layers.Input(shape=(gm_ftrs), name='gm_input')
    encoder = tf.keras.layers.Dense(512, name='l1', activation=af)(gm_input)
    encoder = tf.keras.layers.Dense(256, name='l2', activation=af)(encoder)
    encoder = tf.keras.layers.Dense(64, name='l3', activation=af)(encoder)
    encoder = tf.keras.layers.Dense(16, name='l4', activation=af)(encoder)
    latent = tf.keras.layers.Dense(2, name='latent_layer')(encoder)
    decoder = tf.keras.layers.Dense(16, name='l6', activation=af)(latent)
    decoder = tf.keras.layers.Dense(64, name='l7', activation=af)(decoder)
    decoder = tf.keras.layers.Dense(256, name='l8', activation=af)(decoder)
    decoder = tf.keras.layers.Dense(512, name='l9', activation=af)(decoder)
    gm_output = tf.keras.layers.Dense(gm_features, name='gm')(decoder)

    model = tf.keras.Model(
        inputs=gm_input,
        outputs=gm_output
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.MeanSquaredLogarithmicError(
            reduction='sum_over_batch_size'),
        loss_weights=gm_loss_weight
    )
    encode_gif = tf.keras.Model(
        inputs=gm_input,
        outputs=latent
    )
    return model, encode_gif


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
                                  tf.sparse.to_dense(
                                      tf.sparse.reorder(gm_dat)),
                                  name='gm_input')
    return tf.data.Dataset.zip((tensors, 
                                tensors
                               )).shuffle(tensors.cardinality(),
                                          reshuffle_each_iteration=True
                                         ).batch(batches)


# def custom_load_dataset(clq_path, gm_path, start=0, nshards=nsplit):
#     '''Loads generated datasets for clique and gram sparse tensors.'''
#     clq_shard_files = [clq_path+fname for
#                        fname in sorted(os.listdir(clq_path))[:nshards]]
#     print("Loading Clique Datashards: " + clq_shard_files[start] + 
#           " through " + clq_shard_files[nshards-1])
# 
#     gm_shard_files = [gm_path+fname for
#                       fname in sorted(os.listdir(gm_path))[:nshards]]
#     print("Loading Gram Datashards: " + gm_shard_files[start] + 
#           " through " + gm_shard_files[nshards-1])
# 
#     clq_data = tf.data.Dataset.load(clq_shard_files[start])
#     for dat in clq_shard_files[start+1:]:
#         temp = tf.data.Dataset.load(dat)
#         clq_data = clq_data.concatenate(temp)
# 
#     gm_data = tf.data.Dataset.load(gm_shard_files[start])
#     for dat in gm_shard_files[start+1:]:
#         temp = tf.data.Dataset.load(dat)
#         gm_data = gm_data.concatenate(temp)
#     return tf.data.Dataset.zip((clq_data, gm_data))
# 
# 
# def sparse_to_dense(clq_dat, gm_dat):
#     '''Function used for mapping across datasets.'''
#     tensors = tf.data.Dataset.zip((
#         tf.data.Dataset.from_tensor_slices(
#             tf.sparse.to_dense(tf.sparse.reorder(clq_dat)),
#             name='clique_input'),
#         tf.data.Dataset.from_tensor_slices(
#             tf.sparse.to_dense(tf.sparse.reorder(gm_dat)),
#             name='gm_input')))
#     return tf.data.Dataset.zip((tensors, tensors)).batch(batches)


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

# Testing how to implement TFRecords... was a waste of time
# ------------------------------------------------------------------------------
# serialized_cliques = [tf.io.serialize_sparse(el).numpy() for el in
#                       tf.sparse.SparseTensor(onehot_coords,
#                                              clique_data_onehot.data,
#                                              clique_data_onehot.shape)]

# serialized_cliques = tf.io.serialize_sparse(
#     tf.sparse.SparseTensor(onehot_coords,
#                            clique_data_onehot.data,
#                            clique_data_onehot.shape)).numpy()

# with tf.io.TFRecordWriter('ckpt/clique_one_hot_test.tfrecords') as
# file_writer:
#     for sspar in serialized_cliques:
#         serial_clq = tf.train.Example(features=tf.train.Features(
#             feature={'sparse_tensor': tf.train.Feature(
#                 int64_list=tf.train.Int64List(value=sspar))}))
#         file_writer.write(serial_clq.SerializeToString())


# with tf.io.TFRecordWriter('ckpt/gram_renorm_test.tfrecords') as file_writer:
#     file_writer.write(tf.io.serialize_sparse(
#         tf.sparse.SparseTensor(renorm_coords, gram_data_norm.data,
#                                gram_data_norm.shape)))
# ------------------------------------------------------------------------------

# Old tensor building method for cliques and gram matrices
# ------------------------------------------------------------------------------
# clique_tensor = tf.sparse.to_dense(tf.sparse.reorder(
#     tf.sparse.SparseTensor(onehot_coords, clique_data_onehot.data,
#                            clique_data_onehot.shape)))
# del clique_data_onehot

# # Make sure the tf.sparse.reorder() doesn't change row order
# gram_tensor = tf.sparse.to_dense(tf.sparse.reorder(
#     tf.sparse.SparseTensor(renorm_coords, gram_data_norm.data,
#                            gram_data_norm.shape)))
# del gram_data_norm
# ------------------------------------------------------------------------------


# Trying to get Sparse Tensors to work
# # -----------------------------------------------------------------------
# input_lyr = tf.keras.layers.Input(shape=(features), sparse=True)
#                                   # batch_size=batches)
# # assert input_lyr.shape.as_list() == [None, features]
# encoder = tf.keras.layers.Dense(150, activation=af)(input_lyr)
# gm_input = tf.keras.layers.Input(shape=(gm_features), sparse=True)
#                                  # batch_size=batches)

# # assert gm_input.shape.as_list() == [None, gm_features]
# # gm_input.set_shape([None, gm_features])
# # print(encoder)
# # print(gm_input)
# # # conc = tf.keras.layers.Concatenate(axis=-1)([encoder, gm_input])
# # conc = tf.concat([encoder, gm_input], axis=1)
# # print(conc.shape)
# id_mat_sp_list = [tf.keras.KerasTensor(tf.sparse.eye(len(onehot_coords))),
#                   gm_input]
# print(id_mat_sp_list)
# gm_aug_in = tf.keras.layers.Concatenate()(1, id_mat_sp_list)

# print(gm_aug_in)

# conc = tf.sparse.sparse_dense_matmul()

# encoder = tf.keras.layers.Dense(50, activation=af)(conc)
# encoder = tf.keras.layers.Dense(10, activation=af)(encoder)
# latent = tf.keras.layers.Dense(2)(encoder)
# decoder = tf.keras.layers.Dense(10, activation=af)(latent)
# decoder = tf.keras.layers.Dense(50, activation=af)(decoder)
# gm_output = tf.keras.layers.Dense(gm_features)(decoder)
# decoder = tf.keras.layers.Dense(150, activation=af)(decoder)
# output = tf.keras.layers.Dense(features)(decoder)

# model = tf.keras.Model(
#     inputs=[input_lyr, gm_input],
#     outputs=[output, gm_output]
# )

# model.compile(
#     optimizer=tf.keras.optimizers.Adam(),
#     loss=tf.keras.losses.MeanSquaredLogarithmicError(),
#     loss_weights=[clique_loss_weight, gm_loss_weight]
# )
# # -----------------------------------------------------------------------

# Uncomment this if you want it to print the architecture
# model.summary()

# Implementing Distributed Training on Multi-GPU Single Machine
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# Initializing model
model = None
frames = []
epochs = []
# clique_losses = []
gm_losses = []
# batch_sizes = [8000,4000,2000,1000]

# dataset = custom_load_dataset(clq_shard_path, gm_shard_path, shard_id)
dataset = gram_custom_load_dataset(gm_shard_path, init_shard_id)

for ep in range(max_epochs):

    print('Epoch ' + str(ep+1) + '/' + str(max_epochs))
    shard_id = init_shard_id

    # if ep+1 % 2 == 0:
    #     if bpk < len(batch_sizes)-1: bpk += 1
    #     batches = batch_sizes[bpk]
    #     print('Changing Batch Size to ' + str(batches))

    with strategy.scope():
        model, encoder_gif = gram_make_or_restore_model(overwrite, shard_id)

    for shard in dataset.map(sparse_to_dense).prefetch(tf.data.AUTOTUNE):
        shard = shard.with_options(options)
        shard_id = shard_id + 1
        print('Data Shard ' + str(shard_id) + '/' + str(nsplit))

        # Train
        history = model.fit(
            shard,
            epochs=1,
            verbose=1
        )

        # Save the loss values from this epoch
        # clique_losses.append(
        #     history.history['clique_loss'][0]*clique_loss_weight)
        # gm_losses.append(history.history['gm_loss'][0]*gm_loss_weight)
        gm_losses.append(history.history['loss'])
        # epochs.append(max_epochs*(shard_id-1)+ep+1)
        epochs.append(ep*(nsplit-1)+shard_id-1)

        # Predict the latent space values
        output = encoder_gif.predict(shard, verbose=0)

        # Plot the latent space
        figg = plt.figure()
        plt.plot(output[:, 0], output[:, 1], 'b,')

        ImF = fig2img(figg)
        plt.close()
        frames.append(ImF)

        # Save a copy of the network (labeled by number of epochs)
        model.save(folder + '/gram_autoencoder_' + str(ep+1) + 'e')

# After training is complete, export the latent space plots as a gif
frames[0].save(folder + '/' + str(round(time.time())) + '.gif', save_all=True,
               append_images=frames[1:], loop=0)



# for shard in dataset.map(sparse_to_dense).prefetch(tf.data.AUTOTUNE):
#     shard = shard.cache().with_options(options)
#     shard_id = shard_id + 1
#     print('Data Shard ' + str(shard_id) + '/' + str(nsplit))
#     for ep in range(max_epochs):
# 
#         print('Epoch ' + str(ep+1) + '/' + str(max_epochs))
# 
#         with strategy.scope():
#             model, encoder_gif = gram_make_or_restore_model(overwrite, shard_id)
# 
#         # Train
#         history = model.fit(
#             shard,
#             epochs=1,
#             verbose=1
#         )
# 
#         # Save the loss values from this epoch
#         # clique_losses.append(
#         #     history.history['clique_loss'][0]*clique_loss_weight)
#         # gm_losses.append(history.history['gm_loss'][0]*gm_loss_weight)
#         gm_losses.append(history.history['loss'])
#         epochs.append(max_epochs*(shard_id-1)+ep+1)
# 
#         # Predict the latent space values
#         output = encoder_gif.predict(shard, verbose=0)
# 
#         # Plot the latent space
#         figg = plt.figure()
#         plt.plot(output[:, 0], output[:, 1], 'b,')
# 
#         ImF = fig2img(figg)
#         plt.close()
#         frames.append(ImF)
# 
#         # Save a copy of the network (labeled by number of epochs)
#         model.save(folder + '/gram_autoencoder_' + str(ep+1) + 'e')
# 
# # After training is complete, export the latent space plots as a gif
# frames[0].save(folder + '/' + str(round(time.time())) + '.gif', save_all=True,
#                append_images=frames[1:], loop=0)

# model.save(folder + '/gram_autoencoder.h5', save_format='h5')

# Output a plot of the losses
# plt.plot(epochs, clique_losses, 'bo')
plt.plot(epochs, gm_losses, 'bs')
#plt.legend(['clique_losses', 'gm_losses'], loc='upper right')
plt.legend(['gm_losses'], loc='upper right')
# plt.savefig(folder + '/loss_plot.png')
plt.savefig(folder + '/gram_loss_plot.png')
#plt.legend(['clique_losses', 'gm_losses'], loc='upper right')
plt.legend(['gm_losses'], loc='upper right')
# plt.savefig(folder + '/loss_plot.png')
plt.savefig(folder + '/gram_loss_plot.png')


# Save Model in HDF5 format
new_model = tf.keras.models.load_model(folder + '/gram_autoencoder_' + str(max_epochs) + 'e')
new_model.summary()
new_model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.MeanSquaredLogarithmicError(reduction='sum_over_batch_size')
)
new_model.save(folder + '/gram_autoencoder_v' + version + '.h5', save_format='h5')
