from tensorflow import keras

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
