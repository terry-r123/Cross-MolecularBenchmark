import ml_collections as mlc

c_z = mlc.FieldReference(128, field_type=int)   ## pair embedding dimmension
c_m = mlc.FieldReference(256, field_type=int)   ## msa embedding dimmension
c_s = mlc.FieldReference(256, field_type=int)   ## single embedding dimmension

# tune_chunk_size = mlc.FieldReference(True, field_type=bool)
tune_chunk_size = False
prodnafold_config = mlc.ConfigDict(
    {
        "loss":{
            'diffusion': 4.0,
            'distogram': 0.03,
        },
        "data":{
            "batch_size": 1,
            "num_workers": 0,
            "train":{
                "crop_size": 256,
            },
            "eval":{
                "crop_size": None,
            }
        },
        "globals": {
            "c_z": c_z,
            "c_m": c_m,
            "c_s": c_s,
            "c_t": 64,
            "c_e": 64,
            "eps": 1e-8,
        },
        "backbone": {
            "vocab_size": 47,
            "hidden_size": 768,
            "pad_token_id": 0,
            "type_vocab_size": 2,
            "layer_norm_eps": 1e-8,
        },
        "diffusion": {
            "num_blocks_dec": 2,
            "num_blocks_enc": 2,
            "num_blocks_dit": 6,
        },
        "heads":{
            "c_z": c_z,
            "no_bins": 64,
        }
    }
)

# if __name__ == '__main__':
#     my_config = prodnafold_config
#     print(my_config)
    # print(my_config.learning_rate)  # 输出: 0.01
    # print(my_config.hidden_units)   # 输出: [128, 64, 32]