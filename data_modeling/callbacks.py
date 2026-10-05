import os
import tensorflow as tf
from scipy.stats import spearmanr

class SRCCCallback(tf.keras.callbacks.Callback):

    def __init__(self, val_dataset):
        super().__init__()
        self.val_dataset = val_dataset

    def on_epoch_end(self, epoch, logs=None):

        preds = []
        labels = []

        for x, y, _ in self.val_dataset:
            y_pred = self.model(x, training=False)

            preds.extend(y_pred.numpy().flatten())
            labels.extend(y.numpy().flatten())

        srcc, _ = spearmanr(labels, preds)

        print(f"Validation SRCC: {srcc:.4f}")

        logs["val_srcc"] = srcc
        
class TopKModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, 
                 save_dir, 
                 save_weights_only = True, 
                 monitor = 'val_srcc', 
                 mode = 'max', 
                 topK=5):
        
        super().__init__()
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True) # create directory if none exists
        
        self.save_weights_only = save_weights_only
        self.metric = monitor
        self.mode = mode
        self.topK = topK
        self.best_ckpts = [] # contains (metric, filepaths)
        
    def on_epoch_end(self, epoch, logs = None):
        
        # record loss/rmse/srcc value
        score = logs.get(self.metric)

        # save checkpoint
        if self.save_weights_only: 
            ckpt_name = f'cp-{epoch+1:04d}.weights.h5'
            filepath = os.path.join(self.save_dir, ckpt_name)
            self.model.save_weights(filepath)
        
        # sort all checkpoints
        self.best_ckpts.append((score, filepath))
        
        if self.mode == 'max':
            self.best_ckpts.sort(reverse=True) # largest to smallest
        else: 
            self.best_ckpts.sort() # smallest to largest

        # delete every checkpoint except topK
        while len(self.best_ckpts) > self.topK:
            _, pop_fpath = self.best_ckpts.pop(-1)
            
            if os.path.exists(pop_fpath):
                os.remove(pop_fpath)
            
        return 