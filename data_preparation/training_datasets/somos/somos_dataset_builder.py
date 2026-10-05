"""somos dataset."""

import os
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds

csv_dir = 'data_preparation/training_datasets/somos/data'
fs = 16000

class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for somos dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(somos): Specifies the tfds.core.DatasetInfo object
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset
            'mos': tfds.features.Scalar(dtype=tf.float32),          # mean opinion score  
            'std': tfds.features.Scalar(dtype=tf.float32),          # standard deviation in listener ratings
            'utterance': tfds.features.Tensor(shape=(None, ), dtype=tf.float32),  

            'wav_path': tfds.features.Scalar(dtype=tf.string),      # system synthesizing speech
            'system_id': tfds.features.Scalar(dtype=tf.string),     # system synthesizing speech
            'sample_id': tfds.features.Scalar(dtype=tf.string),     # system synthesizing speech
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('utterance', 'score'),  # Set to `None` to disable
        # homepage='https://dataset-homepage/',
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(somos): Downloads the data and defines the splits
    # path = dl_manager.download_and_extract('https://todo-data-url')

    # TODO(somos): Returns the Dict[split names, Iterator[Key, Example]]
    return {
      'train': self._generate_examples(
          csv_path = os.path.join(csv_dir, 'somos_train.csv'),
      ),
      'dev': self._generate_examples(
          csv_path = os.path.join(csv_dir, 'somos_dev.csv'),
      ),
      'test': self._generate_examples(
          csv_path = os.path.join(csv_dir, 'somos_test.csv'),
      ),
    }

  def _generate_examples(self, csv_path):
    """Yields examples."""
    # TODO(somos): Yields (key, example) tuples from the dataset
    df = pd.read_csv(csv_path)

    grouped = df.groupby('sample_id')
    
    for sample_id, group in grouped:
      mos = group['score'].mean()
      std = group['score'].std()
      
      wav_path = group['wav_path'].iloc[0]
      waveform, _ = librosa.load(wav_path, sr=fs)
      system_id = group['system_id'].iloc[0]
      
      yield sample_id, {
          'mos': float(mos),
          'std': float(std),
          'utterance': waveform.astype(np.float32),
          
          'wav_path': str(wav_path),   
          'system_id': str(system_id),   
          'sample_id': str(sample_id),   
      }
