###
# Import libraries
###

import os
import random
import warnings
import numpy as np
import tensorflow as tf
from model import AttentiveMOS
from params import hyperparams
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from callbacks import SRCCCallback, TopKModelCheckpoint

###
# User parameters
###

user_seed = 999
ckpt_path = None # Provide path to a checkpoint if you want to resume training from a previous checkpoint
tfds_path = f'Data/Speech/MOS_datasets' # Replace with your own path to the tfds datasets
ds_names = ['bvcc', 'somos', 'singmos', 'nisqa', 'tmhintqi', 'pstn', 'tencent'] # datasets to be used for training and validation

###
# Fix seed
###

random.seed(user_seed)
np.random.seed(user_seed)                     # keras seed fixing
tf.random.set_seed(user_seed)                 # tensorflow seed fixing
warnings.filterwarnings("ignore")

ckpt_dir = f'checkpoints/seeds/{user_seed}/'
os.makedirs(ckpt_dir, exist_ok=True)          # create directory if none exists

print(f'\nSeed value: {user_seed}')
for key, value in hyperparams.items():
    print(f'{key}:{value}')
    
print(f"GPU information: {tf.config.list_physical_devices('GPU')}")

###
# Fetch datasets
###

all_train_ds, all_dev_ds = [], []
for ds_name in ds_names:
    tfds_dir = f'{tfds_path}/{ds_name}/tf_version'

    [train_set, dev_set], ds_info = tfds.load(
            name = ds_name,
            with_info = True, 
            split = ['train', 'dev'],
            data_dir = tfds_dir, 
            shuffle_files = True
        )
    print(f"Loaded {ds_name} info:\n{ds_info}")

    all_train_ds.append(train_set)
    all_dev_ds.append(dev_set)

train_set = all_train_ds[0]
for ds in all_train_ds[1:]:
    train_set = train_set.concatenate(ds)

dev_set = all_dev_ds[0]
for ds in all_dev_ds[1:]:
    dev_set = dev_set.concatenate(ds)

print("Final merged datasets:")
print("Training examples:", len(train_set))
print("Validation examples:", len(dev_set))

###
# Data pipeline
###

def preprocess(record, replay_or_silence = hyperparams['padding']):
    audio_waveform = record['utterance']
    max_duration = hyperparams['max_len']
    
    if len(audio_waveform) > max_duration:
        input_waveform = audio_waveform[:max_duration]          # retain final segment of desired length
    else:
        if replay_or_silence == 'replay':
            input_waveform = tf.repeat(
                audio_waveform, 
                (max_duration // len(audio_waveform)) + 1,
            )                                                   # repeat audio to exceed required duration 
            input_waveform = input_waveform[:max_duration]      # chop audio to retain desired length
        else:
            input_waveform = tf.pad(
                audio_waveform, 
                [[0, max_duration - len(audio_waveform)]]
            )                                                   # trailing silence

    input_waveform.set_shape((hyperparams['max_len'], ))
    mos_label = tf.reshape(record['mos'], (1, ))
    std_label = tf.reshape(record['std'], (1, ))
        
    return input_waveform, mos_label, std_label

train_set = train_set.map(
        preprocess
    ).shuffle(
        seed = user_seed,
        buffer_size = 512, 
        reshuffle_each_iteration=True
    ).batch(
        batch_size = hyperparams['batch_size'],
        drop_remainder = False,
    ).prefetch(
        buffer_size = tf.data.AUTOTUNE
    )

dev_set = dev_set.map(
        preprocess
    ).batch(
        batch_size = hyperparams['batch_size'],
        drop_remainder = False,        
    ).prefetch(
        buffer_size = tf.data.AUTOTUNE
    )

###
# Model architecture & compilation
###

attentive_mos = AttentiveMOS(hyperparams)   
if ckpt_path:                               # resume from a previous checkpoint 
    attentive_mos.build(input_shape = (hyperparams['max_len'], ))
    attentive_mos.load_weights(ckpt_path)

attentive_mos.compile(
    optimizer = tf.keras.optimizers.AdamW(  # optimization difficulty. deeper vs wider networks. 
        learning_rate = hyperparams['lr'], 
        weight_decay = hyperparams['weight_decay'],
        global_clipnorm = hyperparams['global_clipnorm']
    ),
)

###
# Callbacks & Model Training
###

srcc_callback = SRCCCallback(dev_set)

top_k_callback = TopKModelCheckpoint(   
    mode="max",
    monitor="val_srcc",
    save_dir = ckpt_dir,
    save_weights_only=True,
    topK=hyperparams['topK']
)

early_stop = tf.keras.callbacks.EarlyStopping(
    mode='max',
    monitor='val_srcc',
    restore_best_weights=True,
    patience=hyperparams['patience'],
    verbose=1
)

history = attentive_mos.fit(x = train_set,
    epochs = hyperparams['epochs'], 
    validation_data = dev_set,
    callbacks = [srcc_callback, top_k_callback, early_stop], 
    verbose = 2,
)

###
# Visualize training
###

plt.figure() 
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend(loc='upper right')
plt.savefig(os.path.join(ckpt_dir, 'loss_curves.png'))