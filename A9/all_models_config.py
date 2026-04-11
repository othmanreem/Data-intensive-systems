from models import (
    N_INPUT, N_OUTPUT, N_JOINTS, JOINTS,
    load_all_sequences, flatten_sequences, make_windowed_sequences,
    build_dense_model, build_conv1d_model, build_lstm_model, build_gru_model
)

ALL_MODEL_CONFIGS = {
    # 'dense': {
    #     'build_fn': build_dense_model,
    #     'params': {
    #         'hidden_units': (256, 128, 64),
    #         'activation': 'relu',
    #         'dropout_rate': 0.3,
    #         'l2_reg': 1e-4,
    #     },
    #     'data_type': 'flat',
    # },
    'conv1d': {
        'build_fn': build_conv1d_model,
        'params': {
            'filters': (64, 128),
            'kernel_size': 3,
            'pool_size': 2,
            'dense_units': (64,),
            'activation': 'relu',
            'dropout_rate': 0.3,
        },
        'data_type': 'windowed',
    },
    'conv1d_v2': {
        'build_fn': build_conv1d_model,
        'params': {
            'filters': (32, 64, 128),
            'kernel_size': 5,
            'pool_size': 2,
            'dense_units': (128, 64),
            'activation': 'relu',
            'dropout_rate': 0.4,
        },
        'data_type': 'windowed',
    },
    'conv1d_v3': {
        'build_fn': build_conv1d_model,
        'params': {
            'filters': (128, 256),
            'kernel_size': 3,
            'pool_size': 3,
            'dense_units': (256, 128, 64),
            'activation': 'relu',
            'dropout_rate': 0.2,
        },
        'data_type': 'windowed',
    },
    # 'lstm': {
    #     'build_fn': build_lstm_model,
    #     'params': {
    #         'lstm_units': (64, 32),
    #         'dense_units': (32,),
    #         'activation': 'tanh',
    #         'dropout_rate': 0.3,
    #         'recurrent_dropout': 0.1,
    #     },
    #     'data_type': 'windowed',
    # },
    # 'gru': {
    #     'build_fn': build_gru_model,
    #     'params': {
    #         'gru_units': (64, 32),
    #         'dense_units': (32,),
    #         'dropout_rate': 0.3,
    #         'recurrent_dropout': 0.1,
    #     },
    #     'data_type': 'windowed',
    # },
}
