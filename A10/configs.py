from models import (
    N_INPUT, N_OUTPUT, N_JOINTS, JOINTS,
    load_all_sequences, flatten_sequences, make_windowed_sequences,
    build_dense_model, build_conv1d_model, build_lstm_model, build_gru_model
)
from tensorflow import keras

# Optimizer configurations with learning rate settings
OPTIMIZER_CONFIGS = {
    'sgd': {
        'name': 'SGD',
        'optimizer_fn': lambda lr: keras.optimizers.SGD(
            learning_rate=lr,
            momentum=0.9,
            nesterov=True
        ),
        'default_lr': 0.01,
        'description': 'Stochastic Gradient Descent with momentum (0.9) and Nesterov acceleration'
    },
    'rmsprop': {
        'name': 'RMSprop',
        'optimizer_fn': lambda lr: keras.optimizers.RMSprop(
            learning_rate=lr,
            rho=0.9,
            epsilon=1e-7
        ),
        'default_lr': 0.001,
        'description': 'Root Mean Square propagation with rho=0.9'
    },
    'adam': {
        'name': 'Adam',
        'optimizer_fn': lambda lr: keras.optimizers.Adam(
            learning_rate=lr,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        ),
        'default_lr': 0.001,
        'description': 'Adaptive Moment Estimation with default beta parameters'
    },
}

# Model configurations
MODEL_CONFIGS = {
    'dense': {
        'build_fn': build_dense_model,
        'params': {
            'hidden_units': (256, 128, 64),
            'activation': 'relu',
            'dropout_rate': 0.3,
            'l2_reg': 1e-4,
        },
        'data_type': 'flat',
    },
    'conv1d_v3': {
        'build_fn': build_conv1d_model,
        'params': {
            'filters': (128, 256),
            'kernel_size': 3,
            'pool_size': 3,
            'dense_units': (256, 128, 64),
            'activation': 'relu',
            'dropout_rate': 0.05,
        },
        'data_type': 'windowed',
    },
    'lstm': {
        'build_fn': build_lstm_model,
        'params': {
            'lstm_units': (64, 32),
            'dense_units': (32,),
            'activation': 'tanh',
            'dropout_rate': 0.2,
            'recurrent_dropout': 0.0,
        },
        'data_type': 'windowed',
    },
    'gru': {
        'build_fn': build_gru_model,
        'params': {
            'gru_units': (64, 32),
            'dense_units': (32,),
            'dropout_rate': 0.2,
            'recurrent_dropout': 0.0,
        },
        'data_type': 'windowed',
    },
}

GRU_MODEL_CONFIGS = {
    'gru_drop005': {
        'build_fn': build_gru_model,
        'params': {
            'gru_units': (64, 32),
            'dense_units': (32,),
            'dropout_rate': 0.05,
            'recurrent_dropout': 0.0,
        },
        'data_type': 'windowed',
    },
    'gru_drop01': {
        'build_fn': build_gru_model,
        'params': {
            'gru_units': (64, 32),
            'dense_units': (32,),
            'dropout_rate': 0.1,
            'recurrent_dropout': 0.0,
        },
        'data_type': 'windowed',
    },
    'gru_drop02': {
        'build_fn': build_gru_model,
        'params': {
            'gru_units': (64, 32),
            'dense_units': (32,),
            'dropout_rate': 0.2,
            'recurrent_dropout': 0.0,
        },
        'data_type': 'windowed',
    },
    'gru_drop03': {
        'build_fn': build_gru_model,
        'params': {
            'gru_units': (64, 32),
            'dense_units': (32,),
            'dropout_rate': 0.3,
            'recurrent_dropout': 0.0,
        },
        'data_type': 'windowed',
    },
    'gru_drop05': {
        'build_fn': build_gru_model,
        'params': {
            'gru_units': (64, 32),
            'dense_units': (32,),
            'dropout_rate': 0.5,
            'recurrent_dropout': 0.0,
        },
        'data_type': 'windowed',
    },
    'gru_recurrent01': {
        'build_fn': build_gru_model,
        'params': {
            'gru_units': (64, 32),
            'dense_units': (32,),
            'dropout_rate': 0.1,
            'recurrent_dropout': 0.1,
        },
        'data_type': 'windowed',
    },
    'gru_recurrent02': {
        'build_fn': build_gru_model,
        'params': {
            'gru_units': (64, 32),
            'dense_units': (32,),
            'dropout_rate': 0.1,
            'recurrent_dropout': 0.2,
        },
        'data_type': 'windowed',
    },
    'gru_large': {
        'build_fn': build_gru_model,
        'params': {
            'gru_units': (128, 64, 32),
            'dense_units': (64, 32,),
            'dropout_rate': 0.2,
            'recurrent_dropout': 0.0,
        },
        'data_type': 'windowed',
    },
    'gru_small': {
        'build_fn': build_gru_model,
        'params': {
            'gru_units': (32, 16),
            'dense_units': (16,),
            'dropout_rate': 0.1,
            'recurrent_dropout': 0.0,
        },
        'data_type': 'windowed',
    },
    'gru_high_dropout': {
        'build_fn': build_gru_model,
        'params': {
            'gru_units': (64, 32),
            'dense_units': (32,),
            'dropout_rate': 0.4,
            'recurrent_dropout': 0.2,
        },
        'data_type': 'windowed',
    },
}

# Loss functions to test
LOSS_FUNCTIONS = {
    'mse': {
        'name': 'Mean Squared Error',
        'description': 'MSE penalizes larger errors more heavily'
    },
    'mae': {
        'name': 'Mean Absolute Error',
        'description': 'MAE treats all errors equally'
    }
}
