"""benchmark dataset."""

import os
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds

csv_dir = 'data_preparation/testing_datasets/benchmark/data'
fs = 16000

class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for benchmark dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(benchmark): Specifies the tfds.core.DatasetInfo object
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset
            'mos': tfds.features.Scalar(dtype=tf.float32),          # mean opinion score  
            'utterance': tfds.features.Tensor(shape=(None, ), dtype=tf.float32),  

            'wav_path': tfds.features.Scalar(dtype=tf.string),      # system synthesizing speech
            'system_id': tfds.features.Scalar(dtype=tf.string),     # system synthesizing speech
            'sample_id': tfds.features.Scalar(dtype=tf.string),     # system synthesizing speech
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('utterance', 'mos'),  
        # homepage='https://dataset-homepage/',
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(benchmark): Downloads the data and defines the splits
    # path = dl_manager.download_and_extract('https://todo-data-url')

    # TODO(benchmark): Returns the Dict[split names, Iterator[Key, Example]]
    return {
      'bc19_test': self._generate_examples(
          csv_path = os.path.join(csv_dir, 'bc19_test.csv'),
      ),
          
      'bvcc_test': self._generate_examples(
          csv_path = os.path.join(csv_dir, 'bvcc_test.csv'),
      ),
          
      'nisqa_FOR': self._generate_examples(
          csv_path = os.path.join(csv_dir, 'FOR.csv'),
      ),
      
      'nisqa_LIVETALK': self._generate_examples(
          csv_path = os.path.join(csv_dir, 'LIVETALK.csv'),
      ),
      
      'nisqa_P501': self._generate_examples(
          csv_path = os.path.join(csv_dir, 'P501.csv'),
      ),
      
      'singmos_test': self._generate_examples(
          csv_path = os.path.join(csv_dir, 'singmos_test.csv'),
      ),
       
      'somos_test': self._generate_examples(
          csv_path = os.path.join(csv_dir, 'somos_test.csv'),
      ),
      
      'tmhintqi_test': self._generate_examples(
          csv_path = os.path.join(csv_dir, 'tmhintqi_test.csv'),
      ),
      
      'vmc23_track1a_test': self._generate_examples(
          csv_path = os.path.join(csv_dir, 'vmc23_track1a_test.csv'),
      ),
      
      'vmc23_track1b_test': self._generate_examples(
          csv_path = os.path.join(csv_dir, 'vmc23_track1b_test.csv'),
      ),
      'vmc23_track2_test': self._generate_examples(
          csv_path = os.path.join(csv_dir, 'vmc23_track2_test.csv'),
      ),
      
      'vmc23_track3_test': self._generate_examples(
          csv_path = os.path.join(csv_dir, 'vmc23_track3_test.csv'),
      ),
    }

  def _generate_examples(self, csv_path):
    """Yields examples."""
    df = pd.read_csv(csv_path)
    
    for i in range(len(df)):
      record = df.iloc[i]
      
      sample_id = record['sample_id']
      system_id = record['system_id']

      mos = record['avg_score']
      
      wav_path = record['wav_path']
      waveform, _ = librosa.load(wav_path, sr=fs)

      yield sample_id, {
          'mos': float(mos),
          'utterance': waveform.astype(np.float32),
          
          'wav_path': str(wav_path),   
          'system_id': str(system_id),   
          'sample_id': str(sample_id),   
      }

