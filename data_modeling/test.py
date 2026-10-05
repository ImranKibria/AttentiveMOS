import os
import glob
import scipy
import random
import warnings
import openpyxl
import numpy as np
import tensorflow as tf
from model import AttentiveMOS
from params import hyperparams
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from collections import defaultdict

###
# User parameters
###

user_seed = 999
tfds_path = f'Data/Speech/MOS_datasets' # Replace with your own path to the tfds datasets

###
# Fix Seed & Log Recepie
###

warnings.filterwarnings("ignore")

random.seed(user_seed)
np.random.seed(user_seed)                    # keras seed fixing
tf.random.set_seed(user_seed)                # tensorflow seed fixing

ckpt_dir = f'checkpoints/seeds/{user_seed}/'
eval_dir = f'evaluation/seeds/{user_seed}/'

print(f'\nSeed value: {user_seed}')
print(f'Checkpoint directory: {ckpt_dir}')
print(f'Evaluation directory: {eval_dir}')
for key, value in hyperparams.items():
    print(f'{key}: {value}')
print(f"GPU information: {tf.config.list_physical_devices('GPU')}")

best_ckpt = os.path.join(ckpt_dir, 'best_ckpt.weights.h5')

###
# Fetch Datasets 
###

val_ds = dict()
ds_names = ['bvcc', 'somos', 'singmos', 'nisqa', 'tmhintqi', 'pstn', 'tencent']

print('\nLoading validation sets...')
for ds_name in ds_names:
    val_ds[ds_name] = tfds.load(
            name = ds_name,
            split = 'dev',
            with_info = False, 
            shuffle_files = False,
            data_dir = os.path.join(tfds_path, f'{ds_name}/tf_version') 
        )
    print(f"Loaded {ds_name}_val")

print('\nLoading test sets...')
split_names = ['bvcc_test', 'somos_test', 'singmos_test', 'nisqa_FOR', 'nisqa_LIVETALK', 'nisqa_P501', 'tmhintqi_test', 'bc19_test', 'vmc23_track1a_test', 'vmc23_track1b_test', 'vmc23_track2_test', 'vmc23_track3_test']
ds, ds_info = tfds.load(
        with_info = True, 
        name = 'benchmark',
        split = split_names,
        shuffle_files = False,
        data_dir = os.path.join(tfds_path, 'benchmark') , 
    )
print(f'{ds_info}')    

###
# Input Preprocessing
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
    return input_waveform

os.makedirs(eval_dir, exist_ok=True)                        # create directory if none exists

###
# Load Model
###

print(f'\nLoading best model checkpoint...')
attentive_mos = AttentiveMOS(hyperparams)   
attentive_mos.build((None, hyperparams['max_len']))

attentive_mos.load_weights(best_ckpt)
attentive_mos.trainable = False

###
# Metric Evaluation & Plotting
###

print('\nComputing metrics and plotting results...')
def calculate(
    true_mean_scores, predict_mean_scores, true_sys_mean_scores, predict_sys_mean_scores
):

    utt_MSE = np.mean((true_mean_scores - predict_mean_scores) ** 2)
    utt_LCC = np.corrcoef(true_mean_scores, predict_mean_scores)[0][1]
    utt_SRCC = scipy.stats.spearmanr(true_mean_scores, predict_mean_scores)[0]
    utt_KTAU = scipy.stats.kendalltau(true_mean_scores, predict_mean_scores)[0]
    sys_MSE = np.mean((true_sys_mean_scores - predict_sys_mean_scores) ** 2)
    sys_LCC = np.corrcoef(true_sys_mean_scores, predict_sys_mean_scores)[0][1]
    sys_SRCC = scipy.stats.spearmanr(true_sys_mean_scores, predict_sys_mean_scores)[0]
    sys_KTAU = scipy.stats.kendalltau(true_sys_mean_scores, predict_sys_mean_scores)[0]

    return {
        "utt_MSE": utt_MSE,
        "utt_LCC": utt_LCC,
        "utt_SRCC": utt_SRCC,
        "utt_KTAU": utt_KTAU,
        "sys_MSE": sys_MSE,
        "sys_LCC": sys_LCC,
        "sys_SRCC": sys_SRCC,
        "sys_KTAU": sys_KTAU,
    }

def log_metrics_and_plot(name, eval_set, log_sheet, plot_results=True):
    print('Evaluation set: ', name)
    
    eval_results = defaultdict(list)
    eval_sys_results = defaultdict(lambda: defaultdict(list))

    for item in eval_set:
        sys_name = item["system_id"].numpy().decode('utf-8')
        
        audio = tf.expand_dims(preprocess(item), axis=0)
        answer = attentive_mos.predict(audio, verbose=0).item()
        
        avg_score = float(item["mos"])

        eval_results["pred_mean_scores"].append(answer)
        eval_results["true_mean_scores"].append(avg_score)
        
        eval_sys_results["pred_mean_scores"][sys_name].append(answer)
        eval_sys_results["true_mean_scores"][sys_name].append(avg_score)

    eval_results["true_mean_scores"] = np.array(eval_results["true_mean_scores"])
    eval_results["pred_mean_scores"] = np.array(eval_results["pred_mean_scores"])
    eval_sys_results["true_mean_scores"] = np.array(
        [np.mean(scores) for scores in eval_sys_results["true_mean_scores"].values()]
    )
    eval_sys_results["pred_mean_scores"] = np.array(
        [np.mean(scores) for scores in eval_sys_results["pred_mean_scores"].values()]
    )

    # calculate metrics
    results = calculate(
        eval_results["true_mean_scores"],
        eval_results["pred_mean_scores"],
        eval_sys_results["true_mean_scores"],
        eval_sys_results["pred_mean_scores"],
    )
    print(
        f'[UTT][ MSE = {results["utt_MSE"]:.3f} | LCC = {results["utt_LCC"]:.3f} | SRCC = {results["utt_SRCC"]:.3f} | KTAU = {results["utt_KTAU"]:.3f} ] [SYS][ MSE = {results["sys_MSE"]:.3f} | LCC = {results["sys_LCC"]:.3f} | SRCC = {results["sys_SRCC"]:.3f}  | KTAU = {results["sys_KTAU"]:.3f} ]'
    )
    
    # Save Results to Excel File
    synthetic = {"bvcc_val", "somos_val", "singmos_val", "bvcc_test", "somos_test", "singmos_test",
    "bc19_test", "vmc23_track1a_test", "vmc23_track1b_test", "vmc23_track2_test"}
    
    if name in synthetic:
        mse = round(results["sys_MSE"], 4)
        srcc = round(results["sys_SRCC"], 3)
    else:
        mse = round(results["utt_MSE"], 4)
        srcc = round(results["utt_SRCC"], 3)

    log_sheet.append([name, mse, srcc])

    # SCATTER PLOT
    if plot_results:
        plt.figure()
        plt.scatter(eval_results["pred_mean_scores"], eval_results["true_mean_scores"])
        plt.title(name)
        plt.xlim([1,5]); plt.xlabel('Predicted Score')
        plt.ylim([1,5]); plt.ylabel('True MOS Label')
        plt.savefig(os.path.join(eval_dir, name))  

wb = openpyxl.Workbook()
ws = wb.active
ws.title = "Metrics"
ws.append(["Dataset", "MSE", "SRCC"])

for ds_name in ds_names:
    val_set = val_ds[ds_name]
    ds_name = f'{ds_name}_val'
    log_metrics_and_plot(ds_name, val_set, log_sheet=ws, plot_results=False)
    
for idx, split_name in enumerate(split_names):
    test_set = ds[idx]
    log_metrics_and_plot(split_name, test_set, log_sheet=ws)

wb.save(os.path.join(eval_dir, "evaluation_results.xlsx"))
