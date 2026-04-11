"""
Cross-Validation Training Script for Kinect Movement Prediction Models

This script trains all models from models.py using k-fold cross-validation
and provides comprehensive evaluation metrics and visualizations.

1. **K-Fold Cross-Validation** - Uses 5-fold CV with shuffled splits
2. **Multiple Model Training** - Trains Dense, Conv1D, LSTM, and GRU models
3. **Comprehensive Metrics** - Calculates MSE, RMSE, MAE, and R² scores
4. **Early Stopping** - Prevents overfitting with patience-based stopping
5. **Learning Rate Reduction** - Automatically reduces LR on plateau
6. **Model Checkpointing** - Saves best model weights for each fold
7. **Visualization** - Creates training history plots and prediction analysis
8. **Result Export** - Saves all results to JSON and text files

Data Flow:
    179 files → 24,005 frames (flat) / 3,831 sequences (windowed)
             ↓
        90% train+val   |   10% test
             ↓
        5-fold CV on train+val
             ↓
        Select best fold (by val RMSE)
             ↓
        Evaluate best model on test set

"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Import tensorflow
import tensorflow as tf
from tensorflow import keras

# Add the current directory to path to import models
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import (
    N_INPUT, N_OUTPUT, N_JOINTS, JOINTS,
    load_all_sequences, flatten_sequences, make_windowed_sequences,
    build_dense_model, build_conv1d_model, build_lstm_model, build_gru_model
)

# Import scikit-learn for cross-validation
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Enable XLA Compilation (optional)
tf.config.optimizer.set_jit(True)  # Enable XLA

# Configuration
N_SPLITS = 5
BATCH_SIZE = 32
EPOCHS = 100
PATIENCE = 10
WINDOW_SIZE = 30

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
    }
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


def create_callbacks(model_name, fold, save_dir):
    """Create early stopping and model checkpoint callbacks."""
    checkpoint_path = os.path.join(save_dir, f'{model_name}_fold{fold}_best.h5')

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1
    )

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-7,
        verbose=1
    )

    return [early_stopping, model_checkpoint, reduce_lr]


def evaluate_model(y_true, y_pred, prefix=""):
    """Calculate comprehensive evaluation metrics."""
    # Flatten for frame-level metrics
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)

    metrics = {
        f'{prefix}mse': float(mean_squared_error(y_true_flat, y_pred_flat)),
        f'{prefix}rmse': float(np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))),
        f'{prefix}mae': float(mean_absolute_error(y_true_flat, y_pred_flat)),
        f'{prefix}r2': float(r2_score(y_true_flat, y_pred_flat)),
    }

    # Per-joint metrics
    for i, joint in enumerate(JOINTS):
        joint_mse = float(mean_squared_error(y_true[:, i], y_pred[:, i]))
        joint_rmse = float(np.sqrt(joint_mse))
        joint_mae = float(mean_absolute_error(y_true[:, i], y_pred[:, i]))
        joint_r2 = float(r2_score(y_true[:, i], y_pred[:, i]))

        metrics[f'{prefix}{joint}_mse'] = joint_mse
        metrics[f'{prefix}{joint}_rmse'] = joint_rmse
        metrics[f'{prefix}{joint}_mae'] = joint_mae
        metrics[f'{prefix}{joint}_r2'] = joint_r2

    return metrics


def train_model_with_cv(X, y, model_name, model_config, save_dir, verbose=1, test_size=0.1):
    """
    Train a model using k-fold cross-validation with a test split.

    Args:
        X: Input data
        y: Target data
        model_name: Name of the model
        model_config: Configuration dictionary for the model
        save_dir: Directory to save results
        verbose: Verbosity level
        test_size: Proportion of data to use for testing (default 0.2)

    Returns:
        Dictionary containing cross-validation results
    """
    print(f"\n{'='*60}")
    print(f"Training {model_name.upper()} with {N_SPLITS}-Fold Cross-Validation")
    print(f"{'='*60}")

    # Prepare data based on data type
    data_type = model_config['data_type']

    if data_type == 'flat':
        X_cv = X['flat']
        y_cv = y['flat']
    else:  # windowed
        X_cv = X['windowed']
        y_cv = y['windowed_last']

    print(f"Data shape: X={X_cv.shape}, y={y_cv.shape}")

    # Split data into train+val and test sets
    from sklearn.model_selection import train_test_split
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_cv, y_cv, test_size=test_size, random_state=42, shuffle=True
    )

    print(f"Training+Validation samples: {len(X_trainval)}, Test samples: {len(X_test)}")

    # Initialize KFold on training+validation set
    kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    # Storage for results
    fold_results = []
    history_list = []
    best_models = []
    best_val_metrics = []  # Store best validation metrics for each fold

    print(f"\nStarting {N_SPLITS}-fold cross-validation on training+validation set...")

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_trainval), 1):
        print(f"\n{'='*40}")
        print(f"Fold {fold}/{N_SPLITS}")
        print(f"{'='*40}")

        # Split data
        X_train, X_val = X_trainval[train_idx], X_trainval[val_idx]
        y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]

        print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

        # Build model
        model = model_config['build_fn'](**model_config['params'])

        # Get loss function from config, default to 'mse'
        loss_function = model_config.get('loss_function', 'mse')

        model.compile(
            optimizer='adam',
            loss=loss_function,
            metrics=['mae', 'mse']
        )

        # Print model summary
        if fold == 1 and verbose > 1:
            model.summary()

        # Create callbacks
        callbacks = create_callbacks(model_name, fold, save_dir)

        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=verbose
        )

        history_list.append(history)
        best_models.append(model)

        # Evaluate on validation set
        y_val_pred = model.predict(X_val, verbose=0)
        val_metrics = evaluate_model(y_val, y_val_pred, prefix='val_')
        best_val_metrics.append(val_metrics)  # Store for later test evaluation

        # Evaluate on training set
        y_train_pred = model.predict(X_train, verbose=0)
        train_metrics = evaluate_model(y_train, y_train_pred, prefix='train_')

        fold_result = {
            'fold': fold,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'best_epoch': len(history.history['loss']),
        }
        fold_results.append(fold_result)

        print(f"\nValidation Results:")
        print(f"  RMSE: {val_metrics['val_rmse']:.6f}")
        print(f"  MAE:  {val_metrics['val_mae']:.6f}")
        print(f"  R²:   {val_metrics['val_r2']:.6f}")

    # Calculate aggregate statistics
    aggregate_results = calculate_aggregate_results(fold_results)

    # Find best fold based on validation RMSE and evaluate on test set
    best_fold_idx = np.argmin([m['val_rmse'] for m in best_val_metrics])
    best_model = best_models[best_fold_idx]
    best_fold_num = best_fold_idx + 1

    print(f"\n{'='*40}")
    print(f"Best Fold: {best_fold_num}")
    print(f"{'='*40}")

    # Evaluate best model on test set
    print(f"\nEvaluating best model on test set...")
    y_test_pred = best_model.predict(X_test, verbose=0)
    test_metrics = evaluate_model(y_test, y_test_pred, prefix='test_')

    print(f"\nTest Results:")
    print(f"  RMSE: {test_metrics['test_rmse']:.6f}")
    print(f"  MAE:  {test_metrics['test_mae']:.6f}")
    print(f"  R²:   {test_metrics['test_r2']:.6f}")

    # Save results
    save_cv_results(fold_results, aggregate_results, model_name, save_dir, test_metrics=test_metrics)

    # Plot training history
    plot_training_history(history_list, model_name, save_dir)

    # Plot predictions vs actual
    plot_predictions(best_models, X_cv, y_cv, model_name, save_dir)

    # Plot test predictions
    plot_test_predictions(best_model, X_test, y_test, model_name, save_dir)

    return {
        'fold_results': fold_results,
        'aggregate_results': aggregate_results,
        'history_list': history_list,
        'best_models': best_models,
        'best_fold': best_fold_num,
        'test_metrics': test_metrics,
    }


def calculate_aggregate_results(fold_results):
    """Calculate aggregate statistics across all folds."""
    aggregate = {
        'n_folds': len(fold_results),
        'train_metrics': {},
        'val_metrics': {},
    }

    # Collect all metric keys
    train_keys = fold_results[0]['train_metrics'].keys()
    val_keys = fold_results[0]['val_metrics'].keys()

    # Calculate mean and std for each metric
    for key in train_keys:
        values = [fr['train_metrics'][key] for fr in fold_results]
        aggregate['train_metrics'][key] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
        }

    for key in val_keys:
        values = [fr['val_metrics'][key] for fr in fold_results]
        aggregate['val_metrics'][key] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
        }

    # Best fold (based on val_loss)
    best_fold_idx = np.argmin([fr['val_metrics']['val_mse'] for fr in fold_results])
    aggregate['best_fold'] = best_fold_idx + 1
    aggregate['best_fold_metrics'] = fold_results[best_fold_idx]

    return aggregate


def save_cv_results(fold_results, aggregate_results, model_name, save_dir, test_metrics=None):
    """Save cross-validation results to JSON file."""
    results_path = os.path.join(save_dir, f'{model_name}_cv_results.json')

    # Convert to serializable format
    output = {
        'model_name': model_name,
        'n_splits': N_SPLITS,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'patience': PATIENCE,
        'aggregate_results': aggregate_results,
        'fold_details': fold_results,
    }

    # Add test metrics if available
    if test_metrics is not None:
        output['test_metrics'] = test_metrics

    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {results_path}")


def plot_training_history(history_list, model_name, save_dir):
    """Plot training history for all folds."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{model_name.upper()} - Training History across Folds', fontsize=14)

    colors = plt.cm.tab10(np.linspace(0, 1, len(history_list)))

    for i, history in enumerate(history_list):
        color = colors[i]

        # Loss
        axes[0, 0].plot(history.history['loss'], label=f'Fold {i+1} (train)',
                       color=color, linestyle='-')
        axes[0, 0].plot(history.history['val_loss'], label=f'Fold {i+1} (val)',
                       color=color, linestyle='--')

        # MAE
        axes[0, 1].plot(history.history['mae'], label=f'Fold {i+1} (train)',
                       color=color, linestyle='-')
        axes[0, 1].plot(history.history['val_mae'], label=f'Fold {i+1} (val)',
                       color=color, linestyle='--')

    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss (MSE)')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend(loc='upper right', fontsize=8)
    axes[0, 0].grid(True)

    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].set_title('Training and Validation MAE')
    axes[0, 1].legend(loc='upper right', fontsize=8)
    axes[0, 1].grid(True)

    # Learning rate (if available)
    if 'lr' in history.history:
        axes[1, 0].plot(history.history['lr'])
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].grid(True)

    # Combined plot
    for i, history in enumerate(history_list):
        color = colors[i]
        axes[1, 1].plot(history.history['val_loss'], label=f'Fold {i+1}',
                       color=color, alpha=0.7)

    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Validation Loss')
    axes[1, 1].set_title('Validation Loss per Fold')
    axes[1, 1].legend(loc='upper right', fontsize=8)
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_training_history.png'), dpi=150)
    plt.close()


def plot_predictions(best_models, X, y, model_name, save_dir):
    """Plot predicted vs actual values."""
    # Use the best model (first fold's best model as representative)
    model = best_models[0]

    # Predict
    y_pred = model.predict(X, verbose=0)

    # Flatten for scatter plot
    y_true_flat = y.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{model_name.upper()} - Predictions vs Actual', fontsize=14)

    # Scatter plot
    axes[0, 0].scatter(y_true_flat, y_pred_flat, alpha=0.3, s=1)
    axes[0, 0].plot([y_true_flat.min(), y_true_flat.max()],
                   [y_true_flat.min(), y_true_flat.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual')
    axes[0, 0].set_ylabel('Predicted')
    axes[0, 0].set_title('Predicted vs Actual (All Joints)')
    axes[0, 0].grid(True)

    # Per-joint scatter
    for i, joint in enumerate(JOINTS[:6]):  # Plot first 6 joints
        axes[0, 1].scatter(y[:, i], y_pred[:, i], alpha=0.5, s=1, label=joint)
    axes[0, 1].plot([y[:, :6].min(), y[:, :6].max()],
                   [y[:, :6].min(), y[:, :6].max()], 'r--', lw=2)
    axes[0, 1].set_xlabel('Actual')
    axes[0, 1].set_ylabel('Predicted')
    axes[0, 1].set_title('Predicted vs Actual (First 6 Joints)')
    axes[0, 1].legend(loc='upper right', fontsize=8)
    axes[0, 1].grid(True)

    # Residual plot
    residuals = y_true_flat - y_pred_flat
    axes[1, 0].scatter(y_pred_flat, residuals, alpha=0.3, s=1)
    axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Residual')
    axes[1, 0].set_title('Residual Plot')
    axes[1, 0].grid(True)

    # Histogram of residuals
    axes[1, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Residual')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title(f'Residual Distribution (Mean: {residuals.mean():.4f}, Std: {residuals.std():.4f})')
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_predictions.png'), dpi=150)
    plt.close()


def plot_test_predictions(model, X_test, y_test, model_name, save_dir):
    """Plot test predictions vs actual values."""
    # Predict on test set
    y_test_pred = model.predict(X_test, verbose=0)

    # Flatten for scatter plot
    y_true_flat = y_test.reshape(-1)
    y_pred_flat = y_test_pred.reshape(-1)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{model_name.upper()} - Test Set Predictions', fontsize=14)

    # Scatter plot
    axes[0, 0].scatter(y_true_flat, y_pred_flat, alpha=0.3, s=1)
    axes[0, 0].plot([y_true_flat.min(), y_true_flat.max()],
                   [y_true_flat.min(), y_true_flat.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual')
    axes[0, 0].set_ylabel('Predicted')
    axes[0, 0].set_title('Test Set: Predicted vs Actual (All Joints)')
    axes[0, 0].grid(True)

    # Per-joint scatter (first 6 joints)
    for i, joint in enumerate(JOINTS[:6]):
        axes[0, 1].scatter(y_test[:, i], y_test_pred[:, i], alpha=0.5, s=1, label=joint)
    axes[0, 1].plot([y_test[:, :6].min(), y_test[:, :6].max()],
                   [y_test[:, :6].min(), y_test[:, :6].max()], 'r--', lw=2)
    axes[0, 1].set_xlabel('Actual')
    axes[0, 1].set_ylabel('Predicted')
    axes[0, 1].set_title('Test Set: Predicted vs Actual (First 6 Joints)')
    axes[0, 1].legend(loc='upper right', fontsize=8)
    axes[0, 1].grid(True)

    # Residual plot
    residuals = y_true_flat - y_pred_flat
    axes[1, 0].scatter(y_pred_flat, residuals, alpha=0.3, s=1)
    axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Residual')
    axes[1, 0].set_title('Test Set: Residual Plot')
    axes[1, 0].grid(True)

    # Histogram of residuals
    axes[1, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Residual')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title(f'Test Set: Residual Distribution (Mean: {residuals.mean():.4f}, Std: {residuals.std():.4f})')
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_test_predictions.png'), dpi=150)
    plt.close()



def main():
    """Main function to run cross-validation for all models and optimizer variants."""
    print("="*60)
    print("Cross-Validation Training for Kinect Movement Prediction")
    print("Testing Multiple Optimizer Variants")
    print("="*60)

    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           f"cv_results_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nResults will be saved to: {save_dir}")

    # Load data
    print("\n" + "="*60)
    print("Loading Data")
    print("="*60)

    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    DATA_DIR = os.path.join(REPO_ROOT, 'Datasets_all')
    KINECT_DATA_PATH = os.path.join(DATA_DIR, 'kinect_good_preprocessed')

    sequences, file_names = load_all_sequences(KINECT_DATA_PATH)

    # Prepare data
    X_flat, y_flat = flatten_sequences(sequences)
    X_seq, y_seq = make_windowed_sequences(sequences, window_size=WINDOW_SIZE, stride=5)
    y_seq_last = y_seq[:, -1, :]

    X = {
        'flat': X_flat,
        'windowed': X_seq,
    }
    y = {
        'flat': y_flat,
        'windowed_last': y_seq_last,
    }

    print(f"\nFlat dataset:      X={X_flat.shape}  y={y_flat.shape}")
    print(f"Windowed dataset:  X={X_seq.shape}  y_last={y_seq_last.shape}")

    # Define optimizer variants to test
    optimizers_to_test = ['sgd', 'rmsprop', 'adam']

    # Define loss functions to test
    loss_functions_to_test = ['mse', 'mae']

    print(f"\n{'='*60}")
    print("Optimizer Variants to Test:")
    print("="*60)
    for opt_name in optimizers_to_test:
        opt_config = OPTIMIZER_CONFIGS[opt_name]
        print(f"  - {opt_config['name']}: {opt_config['description']}")
        print(f"    Default LR: {opt_config['default_lr']}")

    print(f"\n{'='*60}")
    print("Loss Functions to Test:")
    print("="*60)
    for loss_name in loss_functions_to_test:
        loss_config = LOSS_FUNCTIONS[loss_name]
        print(f"  - {loss_name.upper()}: {loss_config['name']}")
        print(f"    {loss_config['description']}")

    # Train each model with each optimizer and loss function variant
    all_results = {}

    for model_name, model_config in MODEL_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Testing Model: {model_name.upper()}")
        print(f"{'='*60}")

        for optimizer_name in optimizers_to_test:
            for loss_function in loss_functions_to_test:
                # Create a copy of model config with optimizer and loss function specified
                config_with_optimizer = model_config.copy()
                config_with_optimizer['optimizer'] = optimizer_name
                config_with_optimizer['loss_function'] = loss_function

                # Generate a unique run name for this optimizer variant
                run_name = f"{model_name}_{optimizer_name}_{loss_function}"

                try:
                    print(f"\n{'='*60}")
                    print(f"Training {run_name.upper()}")
                    print(f"{'='*60}")

                    results = train_model_with_cv(
                        X, y, run_name, config_with_optimizer, save_dir, verbose=1, test_size=0.2
                    )
                    all_results[run_name] = results

                    # Clear session to free memory
                    tf.keras.backend.clear_session()

                except Exception as e:
                    print(f"\nError training {run_name}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue

    # Save summary
    print("\n" + "="*60)
    print("Cross-Validation Summary")
    print("="*60)

    summary_path = os.path.join(save_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Cross-Validation Training Summary - Optimizer & Loss Function Variants\n")
        f.write("="*50 + "\n\n")

        for run_name, results in all_results.items():
            agg = results['aggregate_results']
            f.write(f"\n{run_name.upper()}\n")
            f.write("-"*30 + "\n")
            f.write(f"Best Fold: {agg['best_fold']}\n\n")

            f.write("Validation Metrics (mean ± std):\n")
            for metric, values in agg['val_metrics'].items():
                if not metric.startswith('train_'):
                    f.write(f"  {metric}: {values['mean']:.6f} ± {values['std']:.6f}\n")

            # Add test metrics if available
            if 'test_metrics' in results:
                f.write("\nTest Metrics:\n")
                for metric, value in results['test_metrics'].items():
                    if not metric.startswith('train_'):
                        f.write(f"  {metric}: {value:.6f}\n")

    print(f"\nSummary saved to: {summary_path}")

    # Print summary to console
    print("\n" + "="*60)
    print("Final Results - Optimizer Comparison")
    print("="*60)

    # Group results by model for comparison
    results_by_model = {}
    for run_name, results in all_results.items():
        model_name = run_name.rsplit('_', 2)[0]  # Remove optimizer and loss suffix
        if model_name not in results_by_model:
            results_by_model[model_name] = []
        results_by_model[model_name].append((run_name, results))

    for model_name, runs in results_by_model.items():
        print(f"\n{model_name.upper()}:")
        print("-" * 40)

        for run_name, results in runs:
            agg = results['aggregate_results']
            # Extract optimizer and loss function from run_name
            parts = run_name.replace(model_name + '_', '').rsplit('_', 1)
            optimizer_name = parts[0]
            loss_function = parts[1] if len(parts) > 1 else 'unknown'

            print(f"\n  Optimizer: {optimizer_name.upper()}, Loss: {loss_function.upper()}")
            print(f"    Best Fold: {agg['best_fold']}")
            print(f"    Val RMSE: {agg['val_metrics']['val_rmse']['mean']:.6f} ± {agg['val_metrics']['val_rmse']['std']:.6f}")
            print(f"    Val MAE:  {agg['val_metrics']['val_mae']['mean']:.6f} ± {agg['val_metrics']['val_mae']['std']:.6f}")
            print(f"    Val R²:   {agg['val_metrics']['val_r2']['mean']:.6f} ± {agg['val_metrics']['val_r2']['std']:.6f}")

            # Print test results if available
            if 'test_metrics' in results:
                print(f"    Test RMSE: {results['test_metrics']['test_rmse']:.6f}")
                print(f"    Test MAE:  {results['test_metrics']['test_mae']:.6f}")
                print(f"    Test R²:   {results['test_metrics']['test_r2']:.6f}")

    print(f"\n\nAll results saved to: {save_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
