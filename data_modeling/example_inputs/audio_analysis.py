###
# Import libraries
###

import os
import warnings
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

###
# User parameters
###
ds_name = 'somos'
split_names = ['train', 'dev']
tfds_dir = '/fs/ess/PAS2301/Data/Speech/MOS_datasets/somos_ood/tf_version'

ckpt_dir = '/users/PAS2301/kibria5/Research/quality_assessment/generalization/AttentiveMOS/checkpoints/somos'

###
# Fix seed
###

np.random.seed(6830)                    # keras seed fixing
tf.random.set_seed(6830)                # tensorflow seed fixing
warnings.filterwarnings("ignore")

print(tf.config.list_physical_devices('GPU'))

###
# Fetch dataset 
###

[train_set, dev_set], ds_info = tfds.load(
        name = ds_name,
        with_info = True, 
        split = split_names,
        data_dir = tfds_dir, 
        shuffle_files = False
    )

print(f'Total training records = {len(train_set)}')        
print(f'Total validation records = {len(dev_set)}')  

###
# Data pipeline
###

max_len = 262144
def preprocess(record, rewind_or_silence = 'silence'):
    audio_waveform = record['utterance']
    max_duration = max_len
    
    if len(audio_waveform) > max_duration:
        input_waveform = audio_waveform[:max_duration]          # retain final segment of desired length
    else:
        if rewind_or_silence == 'rewind':
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

    input_waveform.set_shape((max_len, ))
    mos_label = record['mos']
    std_label = record['std']
    
    return input_waveform, mos_label, std_label

train_set = train_set.map(
        preprocess
    )

dev_set = dev_set.map(
         preprocess
    )

###
# Visualize audio
###
max_count = 3

for i, record in enumerate(train_set):
    if i < max_count:
        audio, mos = record[0], record[1]
        
        plt.figure() 
        plt.plot(audio)
        plt.xlabel('time')
        plt.ylabel('magnitude')
        plt.title(f'MOS = {mos}')
        plt.savefig(f'sample_{i}.png')
    
    else:
        break
    