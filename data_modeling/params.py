hyperparams = {
    'max_len' : 196608,                         # 12.288s duration
    'padding': 'replay',                        # 'silence' or 'replay'
    
    'K': 6,
    'tot_frames' : [12288, 4096, 2048, 1024, 512, 256, 128], 
                                                
    
    'frame_size' : 32,                          # 2ms frame      
    'frame_step' : 16,                          # 1ms frame      
    
    'context_size': [8, 4, 4, 4, 2, 2, 2],      # receptive field
    
    'merge_size': [0, 3, 2, 2, 2, 2, 2],        # number of merging frames
    
    'interm_dim' : 16*4,                        # transformer: intermediate dimension
    'num_heads' : 4,                            # transformer: self attention heads
    'embed_dim' : 16,                           # transformer: embedding dimension
    
    'M': 12,
    'global_embed_dim': 16, 
    
    'head_units': [16, 16],                     # mos head: dense layer units
    
    'lr' : 3e-4,   
    'weight_decay': 0,     
    'global_clipnorm': 1.0,                        
    
    'topK': 5,                                  # top-K model weights to save
    'epochs': 750,
    'patience': 25,
    'batch_size' : 512,                         # lower -> good regularization. 512 in training, 1 in testing.     
}