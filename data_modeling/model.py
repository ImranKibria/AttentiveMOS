import keras
import numpy as np
from keras import ops
import tensorflow as tf
from keras import layers
from scipy.stats import spearmanr
from keras_nlp.layers import TransformerEncoder


def frames_to_context(x, frames_in_context):
    _, frames, embedding_per_frame = x.shape
    tot_contexts = frames // frames_in_context
    x = ops.reshape(
        x,
        (
            -1,
            tot_contexts,
            frames_in_context,
            embedding_per_frame,
        ),
    )
    context_windows = ops.reshape(x, 
        (-1, frames_in_context, embedding_per_frame)
    )
    return context_windows

def context_to_frames(x, frames_in_audio):
    _, frames_in_context, embedding_per_frame = x.shape
    x = ops.reshape(
        x,
        (
            -1,
            frames_in_audio,
            embedding_per_frame,
        ),
    )
    return x

@keras.saving.register_keras_serializable()
class StandardTransformer(layers.Layer):
    def __init__(self, hidden_dim, num_heads, interm_dim, dropout_rate=0.0, activation='gelu', name='global_modeling', **kwargs):
        super().__init__(name=name, **kwargs)

        self.hidden_dim = hidden_dim        # embedding dimension (across all heads)
        self.num_heads = num_heads          # number of attention heads
        self.intermediate_dim = interm_dim  # number of MLP nodes
        self.dropout = dropout_rate
        self.layer_norm_epsilon = 0.01
        self.activation = activation
        return
     
    def build(self, inputs_shape):
        hidden_dim = inputs_shape[-1]
        key_dim = int(hidden_dim // self.num_heads)
        
        self.attention_layer = keras.layers.MultiHeadAttention(
            num_heads = self.num_heads,
            key_dim = key_dim,
            dropout = self.dropout,
            # kernel_initializer = 'he_uniform',
            # bias_initializer = 'zeros',
            name = 'self_attention_layer',
        )
        
        self.attention_layer_norm = keras.layers.LayerNormalization(
            epsilon = self.layer_norm_epsilon,
            name = 'self_attention_layer_norm',
        )
        
        self.attention_layer_norm.build(inputs_shape)
        
        self.attention_dropout = keras.layers.Dropout(
            rate = self.dropout,
            name = 'self_attention_dropout',
        )

        self.mlp_layer_norm = keras.layers.LayerNormalization(
            epsilon = self.layer_norm_epsilon,
            name = "feedforward_layer_norm",
        )
        self.mlp_layer_norm.build(inputs_shape)

        self.mlp_intermediate_dense = keras.layers.Dense(
            self.intermediate_dim,
            activation = self.activation,
            name="mlp_intermediate_dense",
        )
        
        self.mlp_intermediate_dense.build(inputs_shape)
        
        self.mlp_output_dense = keras.layers.Dense(
            hidden_dim,
            name="mlp_output_dense",
        )
        
        intermediate_shape = list(inputs_shape)
        intermediate_shape[-1] = self.intermediate_dim
        self.mlp_output_dense.build(tuple(intermediate_shape))
        
        self.mlp_dropout = keras.layers.Dropout(
            rate = self.dropout,
            name = "mlp_dropout",
        )
        self.built = True
        return

    def call(self, x, training=False):
        residual = x 
        x = self.attention_layer_norm(
            x
            ) 
        x = self.attention_layer(
            query = x,
            value = x,
            attention_mask = None,
            training = training,
            )
        # x = self.attention_dropout(x, training=training)
        x = x + residual

        residual = x
        x = self.mlp_layer_norm(x)
        x = self.mlp_intermediate_dense(x)
        x = self.mlp_output_dense(x)
        # x = self.mlp_dropout(x, training=training)
        x = x + residual

        return x

    def get_config(self):
        base_config = super().get_config()
        base_config.update({
            "dropout": self.dropout,
            "num_heads": self.num_heads,
            "hidden_dim": self.hidden_dim,
            "activation": self.activation,
            "intermediate_dim": self.intermediate_dim,
            "layer_norm_epsilon": self.layer_norm_epsilon,
        })
        return base_config


# swin transformer component
@keras.saving.register_keras_serializable()
class TransformerBlock(layers.Layer): 
    def __init__(self, hidden_dim, num_heads, interm_dim, num_frames, context_size, name='transformer', dropout_rate=0.0, activation='gelu', **kwargs):
        super().__init__(name=name, **kwargs)

        self.hidden_dim = hidden_dim        # embedding dimension (across all heads)
        self.num_frames = num_frames        # number of frames in audio
        self.num_heads = num_heads          # number of attention heads
        self.context_size = context_size    # size of context in number of frames
        self.intermediate_dim = interm_dim  # number of MLP nodes
        self.dropout = dropout_rate
        self.layer_norm_epsilon = 0.01
        self.activation = activation
        
        return
    
    def build(self, inputs_shape):   
        hidden_dim = inputs_shape[-1]
        key_dim = int(hidden_dim // self.num_heads)
        
        self.attention_layer = keras.layers.MultiHeadAttention(
            num_heads = self.num_heads,
            key_dim = key_dim,
            dropout = self.dropout,
            # kernel_initializer = 'he_uniform',
            # bias_initializer = 'zeros',
            name = 'self_attention_layer',
        )
        
        self.attention_layer_norm = keras.layers.LayerNormalization(
            epsilon = self.layer_norm_epsilon,
            name = 'self_attention_layer_norm',
        )
        
        self.attention_layer_norm.build(inputs_shape)
        
        self.attention_dropout = keras.layers.Dropout(
            rate = self.dropout,
            name = 'self_attention_dropout',
        )

        self.mlp_layer_norm = keras.layers.LayerNormalization(
            epsilon = self.layer_norm_epsilon,
            name = "feedforward_layer_norm",
        )
        self.mlp_layer_norm.build(inputs_shape)

        self.mlp_intermediate_dense = keras.layers.Dense(
            self.intermediate_dim,
            activation = self.activation,
            name="mlp_intermediate_dense",
        )
        
        self.mlp_intermediate_dense.build(inputs_shape)
        
        self.mlp_output_dense = keras.layers.Dense(
            hidden_dim,
            name="mlp_output_dense",
        )
        
        intermediate_shape = list(inputs_shape)
        intermediate_shape[-1] = self.intermediate_dim
        self.mlp_output_dense.build(tuple(intermediate_shape))
        
        self.mlp_dropout = keras.layers.Dropout(
            rate = self.dropout,
            name = "mlp_dropout",
        )
        self.built = True

        if self.num_frames < self.context_size:
            self.context_size = self.num_frames
            
        return

    def call(self, x, mask, training=False):             
        residual = x 
        x = self.attention_layer_norm(
            x
            ) 
        x = frames_to_context(
            x, 
            self.context_size
            ) 
        x = self.attention_layer(
            query = x,
            value = x,
            attention_mask = mask,
            training = training,
            )
        x = context_to_frames(
            x, 
            self.num_frames
            )
        # x = self.attention_dropout(x, training=training)
        x = x + residual

        residual = x
        x = self.mlp_layer_norm(x)
        x = self.mlp_intermediate_dense(x)
        x = self.mlp_output_dense(x)
        # x = self.mlp_dropout(x, training=training)
        x = x + residual

        return x

    def get_config(self):
        base_config = super().get_config()
        base_config.update({
            "dropout": self.dropout,
            "num_heads": self.num_heads,
            "hidden_dim": self.hidden_dim,
            "activation": self.activation,
            "num_frames": self.num_frames,
            "context_size": self.context_size,
            "intermediate_dim": self.intermediate_dim,
            "layer_norm_epsilon": self.layer_norm_epsilon,
        })
        return base_config


@keras.saving.register_keras_serializable()
class SwinTransformer(layers.Layer):
    def __init__(self, batch_size, projection_dim, num_heads, intermediate_dim, total_frames, frames_per_context, shift_in_context, dropout_rate=0.0, activation='gelu', name ='swin_block', **kwargs):
        super().__init__(name=name, **kwargs)
        
        # self.mask_count = batch_size
        
        self.projection_dim = projection_dim 
        self.num_heads = num_heads 
        self.intermediate_dim = intermediate_dim
        
        self.total_frames = total_frames 
        self.frames_per_context = frames_per_context 
        self.shift_in_context = shift_in_context 
        
        self.dropout = dropout_rate
        self.activation = activation        
        self.layer_norm_epsilon = 0.001

        self.context_modeling = TransformerBlock(
            hidden_dim = self.projection_dim,
            num_heads = self.num_heads,
            interm_dim = self.intermediate_dim,
            num_frames = self.total_frames,
            context_size = self.frames_per_context,
            dropout_rate = self.dropout,
            activation = self.activation,
            name = 'window_transformer',
        )

        self.shifted_context_modeling = TransformerBlock(
            hidden_dim = self.projection_dim,
            num_heads = self.num_heads,
            interm_dim = self.intermediate_dim,
            num_frames = self.total_frames,
            context_size = self.frames_per_context,
            dropout_rate = self.dropout,
            activation = self.activation,
            name = 'shifted_window_transformer',
        )
        
        return
                
    def shifted_context_mask(self):
        mask_array = np.zeros((1, self.total_frames, 1))
        
        mask_array[:, 0:-self.shift_in_context] = 0
        mask_array[:, -self.shift_in_context: ] = 1

        mask_array = ops.convert_to_tensor(mask_array)

        # mask array to windows
        mask_windows = frames_to_context(
            mask_array, 
            self.frames_per_context
            )

        mask_windows = ops.reshape(
            mask_windows, 
            [-1, self.frames_per_context]
            )

        attn_mask = ops.expand_dims(
            mask_windows, axis=1) - ops.expand_dims(mask_windows, axis=2)

        attn_mask = ops.where(attn_mask != 0, -100.0, attn_mask)
        attn_mask = ops.where(attn_mask == 0, 0.0, attn_mask)
        
        return attn_mask

    def call(self, x, training=False):
        
        x = self.context_modeling(
            x, 
            training = training,
            mask = None,
            )
        
        attention_mask = self.shifted_context_mask()
        # attention_mask = tf.broadcast_to(
        #                     attention_mask,
        #                     (
        #                         tf.shape(x)[0],
        #                         self.frames_per_context,
        #                         self.frames_per_context
        #                     )
        #                 )
        # attention_mask = tf.repeat(attention_mask, self.mask_count, axis=0)    
        attention_mask = tf.repeat(attention_mask, tf.shape(x)[0], axis=0)    
        
        shifted_x = ops.roll(x, shift=-self.shift_in_context, axis=1)         
        shifted_x = self.shifted_context_modeling(
            shifted_x, 
            training = training,
            mask = attention_mask,
            )
        
        x = ops.roll(shifted_x, shift=self.shift_in_context, axis=1) 
        
        return x

    def get_config(self):
        base_config = super().get_config()
        base_config.update({
            "dropout": self.dropout,
            "num_heads": self.num_heads, 
            "activation": self.activation,
            "total_frames": self.total_frames, 
            "projection_dim": self.projection_dim,
            "intermediate_dim": self.intermediate_dim,
            "shift_in_context": self.shift_in_context,
            "frames_per_context": self.frames_per_context,
            "layer_norm_epsilon": self.layer_norm_epsilon,
        })
        return base_config


@keras.saving.register_keras_serializable()
class Head_Block(layers.Layer):
    def __init__(self, units, name='head_block', **kwargs):
        super().__init__(name=name, **kwargs)
        self.name = name
        self.units = units
        
        self.layers = [
            keras.layers.Dense(
                units = num_neurons, 
                activation = "gelu", 
            ) for num_neurons in self.units
        ]
        self.out_layer = keras.layers.Dense(units=1)
        return

    def call(self, x):
        for layer in self.layers:
            x = layer(x)   
        x = self.out_layer(x)
        return x     

    def get_config(self):
        base_config = super().get_config()
        base_config.update({
            "units": self.units,
        })
        return base_config


@keras.saving.register_keras_serializable()
class AttentiveMOS(keras.Model):
    def __init__(self, hyperparams, name='AttentiveMOS', **kwargs):
        super().__init__(name=name, **kwargs)
        self.hyperparams = hyperparams

        self.loss_tracker = keras.metrics.Mean()
        self.val_loss_tracker = keras.metrics.Mean()
        
        self.rmse_metric = keras.metrics.RootMeanSquaredError()
        self.val_rmse_metric = keras.metrics.RootMeanSquaredError()

        # wave framing block
        self.wave_framing = tf.keras.layers.Lambda(
            lambda x: tf.signal.frame(
                signal = x,
                frame_length = self.hyperparams['frame_size'],
                frame_step = self.hyperparams['frame_step'],
                pad_end = True,
            ), 
            name = 'wave_framing',
        )
        
        # local block 0
        initializer = tf.keras.initializers.TruncatedNormal(stddev=0.02)
        self.embedding_layer = layers.Dense(
            self.hyperparams['embed_dim'],
            name = 'linear_embedding',
            kernel_initializer = initializer,
        )
        self.swin_transformer_0 = SwinTransformer(
            name = 'swin_0',
            batch_size = self.hyperparams['batch_size'],
            projection_dim = self.hyperparams['embed_dim'],
            num_heads = self.hyperparams['num_heads'],
            intermediate_dim = self.hyperparams['interm_dim'],
            total_frames = self.hyperparams['tot_frames'][0],
            frames_per_context = self.hyperparams['context_size'][0],
            shift_in_context = self.hyperparams['context_size'][0]//2,
        )

        # local block 1 to K
        for i in range(1, self.hyperparams['K'] + 1):
            setattr(self, f'frame_merging_{i}', layers.MaxPool1D(
                name=f'merger_{i}',
                pool_size=self.hyperparams['merge_size'][i], 
                strides=self.hyperparams['merge_size'][i],
            ))

            setattr(self, f'swin_transformer_{i}', SwinTransformer(
                name=f'swin_{i}',
                batch_size = self.hyperparams['batch_size'],
                projection_dim=self.hyperparams['embed_dim'],
                num_heads=self.hyperparams['num_heads'],
                intermediate_dim=self.hyperparams['interm_dim'],
                total_frames=self.hyperparams['tot_frames'][i],
                frames_per_context=self.hyperparams['context_size'][i],
                shift_in_context=self.hyperparams['context_size'][i] // 2,
            ))
        
        # MOS token                               
        self.label_token = tf.Variable(
            initial_value = tf.zeros_initializer()(
                shape=(1, self.hyperparams['global_embed_dim']), 
                dtype="float32"
            ), 
            trainable = True,
            name='label_token'
        )
        
        self.broadcast_layer = tf.keras.layers.Lambda(
            lambda inputs: tf.broadcast_to(
                inputs[0],
                [tf.shape(inputs[1])[0], 1, self.hyperparams['global_embed_dim']]
            ), 
            name = 'broadcast_layer'
        )
        
        self.concat_layer = keras.layers.Concatenate(
            axis = 1, 
            name='concat_embeddings'
        )
        
        # Global blocks 1 to M
        for j in range(1, self.hyperparams['M'] + 1):
            setattr(self, f'global_transformer_{j}', StandardTransformer(
                num_heads=self.hyperparams['num_heads'],
                hidden_dim=self.hyperparams['global_embed_dim'],
                interm_dim=4 * self.hyperparams['global_embed_dim'],
                name=f'global_transformer_{j}',
            ))

        # MLP
        self.label_head = Head_Block(
            units=self.hyperparams['head_units'], 
            name='label_head'
            )
        
        print(self.model_summary(input_shape=(self.hyperparams['max_len'], )))        
        return

    def call(self, x, training=False):

        x = self.wave_framing(x)
        
        x = self.embedding_layer(x)
        x = self.swin_transformer_0(
            x, 
            training=training,
            )

        for i in range(1, self.hyperparams['K'] + 1):
            frame_merging = getattr(self, f'frame_merging_{i}')
            swin_transformer = getattr(self, f'swin_transformer_{i}')
            
            x = frame_merging(x, training=training)
            x = swin_transformer(x, training=training)

        label_token = self.broadcast_layer([self.label_token, x]) 
        x = self.concat_layer([label_token, x])
        
        for j in range(1, self.hyperparams['M'] + 1):
            transformer = getattr(self, f'global_transformer_{j}')
            x = transformer(x, training=training)
        
        label_estimate = self.label_head(x[:, 0, :])   
        return label_estimate

    def model_summary(self, input_shape):
        x = tf.keras.Input(shape = input_shape)
        my_model = tf.keras.Model(inputs=[x], outputs=self.call(x))
        return my_model.summary()

    def custom_loss(self, label, prediction, label_noise):  
        # MSE
        l2_distance = tf.reduce_mean(tf.square(label - prediction), axis=-1, keepdims=True)
        
        # MAE
        # l1_distance = tf.reduce_mean(tf.abs(label - prediction), axis=-1, keepdims=True)
        
        # Proposed Error
        # normalized_error = tf.divide(l1_distance, label_noise + 0.01)
        # log_normalized_error = tf.math.log1p(normalized_error)
        
        return tf.reduce_sum(l2_distance)

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [
            self.rmse_metric, 
            self.loss_tracker,
            self.val_rmse_metric,
            self.val_loss_tracker,
        ]

    def train_step(self, data):
        # Unpack the data. 
        x, y, y_std = data
          
        with tf.GradientTape() as tape:
            # Forward pass
            y_pred = self(x, training=True)  
            
            # Compute the loss value 
            loss = self.custom_loss(y, y_pred, y_std)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Update metrics (includes the metric that tracks the loss)
        self.loss_tracker.update_state(loss)
        self.rmse_metric.update_state(y, y_pred)
        
        # Return a dict mapping metric names to current value
        return {"rmse": self.rmse_metric.result(),
                "loss": self.loss_tracker.result()}

    def test_step(self, data):
        # Unpack the data
        x, y, y_std = data
        
        # Compute predictions
        y_pred = self(x, training=False)
        
        # Calculate the loss.
        val_loss = self.custom_loss(y, y_pred, y_std)

        # Update the metrics.
        self.val_loss_tracker.update_state(val_loss)
        self.val_rmse_metric.update_state(y, y_pred)

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {"rmse": self.val_rmse_metric.result(),
                "loss": self.val_loss_tracker.result()}

    def get_config(self):
        base_config = super().get_config()
        base_config.update({
            "hyperparams": self.hyperparams,
        })
        return base_config

    def get_latent_embeddings(self, x):
        x = self.wave_framing(x)
        
        x = self.embedding_layer(x)
        x = self.swin_transformer_0(
            x, 
            training=False,
            )

        for i in range(1, self.hyperparams['K'] + 1):
            frame_merging = getattr(self, f'frame_merging_{i}')
            swin_transformer = getattr(self, f'swin_transformer_{i}')
            
            x = frame_merging(x, training=False)
            x = swin_transformer(x, training=False)

        label_token = self.broadcast_layer([self.label_token, x]) 
        x = self.concat_layer([label_token, x])
        
        for j in range(1, self.hyperparams['M'] + 1):
            transformer = getattr(self, f'global_transformer_{j}')
            x = transformer(x, training=False)

        return x