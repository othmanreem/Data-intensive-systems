#!/usr/bin/env python3
"""
Standardized Timing Benchmarking Framework for Classification Models

This framework provides fair and consistent timing benchmarks for comparing
classification models (A4, A5, A5b, A6) with metrics for:
- Inference time (mean, std, min, max, percentiles)
- Memory usage
- Prediction accuracy
- Model size
- Feature extraction time

Usage:
    python benchmark_timing.py [--samples N] [--repeats M] [--output FILE]

Author: Benchmark Framework v1.0
"""

import os
import sys
import pickle
import time
import tracemalloc
import warnings
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import statistics

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Import model paths
from all_classification import (
    a4_rf,
    a5_ensemnble,
    a5b_adaboost,
    a5b_bagging_tree,
    a6_svm
)

# Import custom classes for unpickling
from adaboost_classes import (
    AdaBoostEnsemble,
    WeightedDecisionTree
)

# ============================================================================
# Configuration
# ============================================================================

REPO_ROOT    = os.path.abspath(os.path.join(project_root, '..'))
DATA_DIR     = os.path.join(REPO_ROOT, 'Datasets_all')
OUTPUT_DIR   = os.path.join(project_root, 'benchmark_results')

# Weaklink categories (14 classes)
WEAKLINK_CATEGORIES = [
    'ExcessiveForwardLean', 'ForwardHead', 'LeftArmFallForward',
    'LeftAsymmetricalWeightShift', 'LeftHeelRises', 'LeftKneeMovesInward',
    'LeftKneeMovesOutward', 'LeftShoulderElevation', 'RightArmFallForward',
    'RightAsymmetricalWeightShift', 'RightHeelRises', 'RightKneeMovesInward',
    'RightKneeMovesOutward', 'RightShoulderElevation'
]

# Duplicate NASM columns
DUPLICATE_NASM_COLS = [
    'No_1_NASM_Deviation',
    'No_2_NASM_Deviation',
    'No_3_NASM_Deviation',
    'No_4_NASM_Deviation',
    'No_5_NASM_Deviation',
]

EXCLUDE_COLS = ['ID', 'WeakestLink', 'EstimatedScore']
EXPECTED_CLASSES = WEAKLINK_CATEGORIES.copy()

# Benchmark parameters
DEFAULT_NUM_SAMPLES = 100
DEFAULT_NUM_REPEATES = 10
DEFAULT_OUTPUT_FILE = None


# ============================================================================
# Data Classes for Results
# ============================================================================

@dataclass
class ModelMetrics:
    """Metrics for a single model benchmark."""
    model_name: str
    model_path: str

    # Timing metrics (seconds)
    inference_time_mean: float = 0.0
    inference_time_std: float = 0.0
    inference_time_min: float = 0.0
    inference_time_max: float = 0.0
    inference_time_p50: float = 0.0
    inference_time_p95: float = 0.0
    inference_time_p99: float = 0.0

    # Memory metrics (bytes)
    memory_usage_mean: float = 0.0
    memory_usage_std: float = 0.0
    memory_usage_peak: float = 0.0

    # Prediction metrics
    accuracy: float = 0.0
    predictions_correct: int = 0
    predictions_total: int = 0

    # Model characteristics
    model_size_bytes: int = 0
    num_features: int = 0
    num_parameters: int = 0
    model_type: str = ""

    # Feature extraction time (seconds)
    feature_extraction_time_mean: float = 0.0

    # Raw timing samples
    timing_samples: List[float] = field(default_factory=list)
    memory_samples: List[float] = field(default_factory=list)

    # Status
    status: str = "SUCCESS"
    error_message: str = ""


@dataclass
class BenchmarkResults:
    """Complete benchmark results for all models."""
    timestamp: str
    num_samples: int
    num_repeats: int
    models: Dict[str, ModelMetrics] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'timestamp': self.timestamp,
            'num_samples': self.num_samples,
            'num_repeats': self.num_repeats,
            'models': {
                name: {
                    **asdict(metrics),
                    'timing_samples': list(metrics.timing_samples),
                    'memory_samples': list(metrics.memory_samples)
                }
                for name, metrics in self.models.items()
            }
        }

    def to_json(self, filepath: Optional[str] = None) -> str:
        """Export to JSON string or file."""
        data = self.to_dict()
        json_str = json.dumps(data, indent=2, default=str)

        if filepath:
            os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
            with open(filepath, 'w') as f:
                f.write(json_str)

        return json_str


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_and_prepare_data() -> Dict[str, Any]:
    """Load and prepare data following the same pipeline as classification_baseline.py.

    Returns:
        Dictionary containing:
        - feature_columns: List of feature column names
        - scaler: Fitted StandardScaler
        - X_train, X_test: Feature matrices (unscaled)
        - y_train, y_test: Target arrays
        - merged_df: Merged dataframe
    """
    # Load datasets
    movement_features_df = pd.read_csv(os.path.join(DATA_DIR, 'aimoscores.csv'))
    weaklink_scores_df = pd.read_csv(os.path.join(DATA_DIR, 'scores_and_weaklink.csv'))

    print(f'  Movement features shape: {movement_features_df.shape}')
    print(f'  Weak link scores shape: {weaklink_scores_df.shape}')

    # Create WeakestLink target column
    weaklink_scores_df['WeakestLink'] = (
        weaklink_scores_df[WEAKLINK_CATEGORIES].idxmax(axis=1)
    )

    # Merge datasets
    target_df = weaklink_scores_df[['ID', 'WeakestLink']].copy()
    merged_df = movement_features_df.merge(target_df, on='ID', how='inner')
    print(f'  Merged dataset shape: {merged_df.shape}')

    # Extract feature columns - include ALL columns except EXCLUDE_COLS
    feature_columns = [c for c in merged_df.columns if c not in EXCLUDE_COLS]

    X = merged_df[feature_columns].values
    y = merged_df['WeakestLink'].values

    print(f'  Feature matrix shape: {X.shape}')
    print(f'  Number of features: {len(feature_columns)}')
    print(f'  Number of classes: {len(np.unique(y))}')

    # Create train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Fit scaler on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return {
        'feature_columns': feature_columns,
        'scaler': scaler,
        'X_train': X_train,
        'X_train_scaled': X_train_scaled,
        'y_train': y_train,
        'X_test': X_test,
        'X_test_scaled': X_test_scaled,
        'y_test': y_test,
        'merged_df': merged_df,
    }


def create_samples_from_test_data(
    data: Dict[str, Any],
    num_samples: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Create samples from test data for benchmarking.

    Args:
        data: Dictionary from load_and_prepare_data()
        num_samples: Number of samples to select

    Returns:
        Tuple of (sample_features, true_labels)
    """
    # Use test data for benchmarking
    X_test = data['X_test']
    y_test = data['y_test']

    # Select first num_samples from test set
    n_samples = min(num_samples, len(X_test))
    sample_features = X_test[:n_samples]
    true_labels = y_test[:n_samples]

    return sample_features, true_labels


# ============================================================================
# Model Loading Functions
# ============================================================================

def load_model(model_path: str, model_name: str) -> Tuple[Any, Optional[Any], Optional[List[str]], Any]:
    """Load a model from a pickle file.

    Args:
        model_path: Path to the pickle file
        model_name: Name of the model for logging

    Returns:
        Tuple of (model, scaler, feature_columns, artifact)
    """
    full_path = os.path.join(project_root, model_path)

    if not os.path.exists(full_path):
        print(f"  ⚠️  Model file not found: {full_path}")
        return None, None, None, None

    try:
        with open(full_path, 'rb') as f:
            artifact = pickle.load(f)

        # Extract model and scaler based on artifact structure
        if isinstance(artifact, dict):
            model = artifact.get('model')
            scaler = artifact.get('scaler')
            feature_columns = artifact.get('feature_columns')
        else:
            # A6 SVM is a Pipeline object
            model = artifact
            scaler = None
            feature_columns = None

            # Extract scaler from pipeline if it exists
            if hasattr(model, 'steps') and len(model.steps) >= 1:
                for step_name, step_obj in model.steps:
                    if hasattr(step_obj, 'transform'):
                        if hasattr(step_obj, 'n_features_in_') and not hasattr(step_obj, 'predict'):
                            scaler = step_obj
                            break

                # Extract feature columns from scaler
                if hasattr(model, 'steps') and len(model.steps) > 0:
                    first_step = model.steps[0][1]
                    if hasattr(first_step, 'get_feature_names_out'):
                        try:
                            names = first_step.get_feature_names_out()
                            import re
                            if not all(re.fullmatch(r'x\d+', n) for n in names):
                                feature_columns = names
                        except:
                            pass

        print(f"  ✓ Loaded {model_name}")
        return model, scaler, feature_columns, artifact
    except Exception as e:
        print(f"  ✗ Error loading {model_name}: {e}")
        return None, None, None, None


def get_model_info(model: Any) -> Dict[str, Any]:
    """Extract model information for benchmarking.

    Args:
        model: The trained model

    Returns:
        Dictionary with model characteristics
    """
    info = {
        'model_type': type(model).__name__,
        'num_parameters': 0,
        'num_features': 0
    }

    # Count parameters based on model type
    if hasattr(model, 'n_estimators'):
        info['num_parameters'] += getattr(model, 'n_estimators', 0)

    if hasattr(model, 'estimators_'):
        info['num_parameters'] += len(getattr(model, 'estimators_', []))

    if hasattr(model, 'n_features_in_'):
        info['num_features'] = model.n_features_in_

    if hasattr(model, 'classes_'):
        info['num_classes'] = len(model.classes_)

    # For ensemble models
    if hasattr(model, 'estimators_'):
        for est in getattr(model, 'estimators_', []):
            if hasattr(est, 'n_features_in_'):
                info['num_features'] = est.n_features_in_
                break

    return info


# ============================================================================
# Benchmarking Functions
# ============================================================================

def measure_inference_time(
    model: Any,
    scaler: Optional[Any],
    sample_features: np.ndarray,
    model_feature_columns: Optional[List[str]],
    feature_columns: List[str],
    num_repeats: int,
    single_sample_mode: bool = False
) -> Tuple[List[float], List[float], Optional[str]]:
    """Measure inference time for a model.

    Args:
        model: The trained model
        scaler: Scaler for feature preprocessing
        sample_features: Input features
        model_feature_columns: Expected feature columns for the model
        feature_columns: All available feature columns
        num_repeats: Number of repetitions for averaging
        single_sample_mode: If True, measure each sample individually (for single sample latency)

    Returns:
        Tuple of (timing_samples, memory_samples, error_message)
    """
    timing_samples = []
    memory_samples = []

    try:
        # Filter features if needed
        if model_feature_columns is not None:
            available_features = [f for f in model_feature_columns if f in feature_columns]
            if len(available_features) > 0:
                # Convert column names to indices for numpy array
                feature_indices = [feature_columns.index(f) for f in available_features]
                test_features = sample_features[:, feature_indices]
            else:
                test_features = sample_features
        else:
            # model_feature_columns is None - likely A6 SVM pipeline
            # Check if we need to drop duplicate NASM columns
            if hasattr(model, 'steps') and len(model.steps) > 0:
                first_step = model.steps[0][1]
                n_expected = getattr(first_step, 'n_features_in_', None)
                if n_expected is not None:
                    # Identify indices of duplicate NASM columns
                    dup_indices = [i for i, c in enumerate(feature_columns) if c in DUPLICATE_NASM_COLS]
                    # Get all indices except duplicate NASM columns
                    valid_indices = [i for i in range(len(feature_columns)) if i not in dup_indices]
                    if len(valid_indices) == n_expected:
                        # Select only the columns that match expected features
                        test_features = sample_features[:, valid_indices]
                    else:
                        # Fallback: slice to expected number of features
                        test_features = sample_features[:, :n_expected]
                else:
                    test_features = sample_features
            else:
                test_features = sample_features

        # Handle A6 SVM pipeline (scaler already in pipeline)
        if model_feature_columns is None and hasattr(model, 'steps'):
            scaler_to_use = None
        else:
            scaler_to_use = scaler

        # Determine how many predictions to make
        if single_sample_mode:
            # For single sample mode: repeat each sample individually
            num_predictions = num_repeats * len(test_features)
        else:
            # For batch mode: num_repeats on all samples
            num_predictions = num_repeats

        for i in range(num_predictions):
            # Start memory tracking
            tracemalloc.start()
            start_time = time.perf_counter()

            # Make prediction
            if single_sample_mode:
                # Single sample prediction: use one row at a time
                single_sample = test_features[i % len(test_features)].reshape(1, -1)
                if scaler_to_use is not None:
                    features = scaler_to_use.transform(single_sample)
                else:
                    features = single_sample
            else:
                # Batch prediction: use all samples
                if scaler_to_use is not None:
                    features = scaler_to_use.transform(test_features)
                else:
                    features = test_features

            prediction = model.predict(features)

            end_time = time.perf_counter()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Record measurements
            timing_samples.append(end_time - start_time)
            memory_samples.append(peak)

        return timing_samples, memory_samples, None

    except Exception as e:
        return [], [], str(e)


def calculate_percentiles(values: List[float]) -> Dict[str, float]:
    """Calculate percentiles for a list of values.

    Args:
        values: List of numeric values

    Returns:
        Dictionary with percentile values
    """
    if not values:
        return {
            'p50': 0.0,
            'p95': 0.0,
            'p99': 0.0
        }

    sorted_values = sorted(values)
    n = len(sorted_values)

    return {
        'p50': sorted_values[int(n * 0.50)],
        'p95': sorted_values[int(n * 0.95)],
        'p99': sorted_values[int(n * 0.99)]
    }


def benchmark_single_model(
    model_name: str,
    model_path: str,
    sample_features: np.ndarray,
    true_labels: np.ndarray,
    feature_columns: List[str],
    num_repeats: int,
    single_sample_mode: bool = False
) -> ModelMetrics:
    """Benchmark a single model.

    Args:
        model_name: Name of the model
        model_path: Path to the model file
        sample_features: Input features for benchmarking
        true_labels: Ground truth labels
        feature_columns: All available feature columns
        num_repeats: Number of repetitions
        single_sample_mode: If True, measure each sample individually (for single sample latency)

    Returns:
        ModelMetrics object with benchmark results
    """
    metrics = ModelMetrics(model_name=model_name, model_path=model_path)

    print(f"\n  Benchmarking {model_name}...")

    # Load model
    model, scaler, model_feature_columns, artifact = load_model(model_path, model_name)

    if model is None:
        metrics.status = "LOAD_ERROR"
        metrics.error_message = "Failed to load model"
        return metrics

    # Get model info
    model_info = get_model_info(model)
    metrics.model_type = model_info.get('model_type', type(model).__name__)
    metrics.num_features = model_info.get('num_features', 0)

    # Get model size
    try:
        model_size = os.path.getsize(os.path.join(project_root, model_path))
        metrics.model_size_bytes = model_size
    except:
        metrics.model_size_bytes = 0

    # Run inference benchmarks
    timing_samples, memory_samples, error = measure_inference_time(
        model, scaler, sample_features, model_feature_columns,
        feature_columns, num_repeats, single_sample_mode=single_sample_mode
    )

    if error:
        metrics.status = "INFERENCE_ERROR"
        metrics.error_message = error
        return metrics

    # Store raw samples
    metrics.timing_samples = timing_samples
    metrics.memory_samples = memory_samples

    # Calculate timing statistics
    if timing_samples:
        metrics.inference_time_mean = statistics.mean(timing_samples)
        metrics.inference_time_std = statistics.stdev(timing_samples) if len(timing_samples) > 1 else 0.0
        metrics.inference_time_min = min(timing_samples)
        metrics.inference_time_max = max(timing_samples)

        percentiles = calculate_percentiles(timing_samples)
        metrics.inference_time_p50 = percentiles['p50']
        metrics.inference_time_p95 = percentiles['p95']
        metrics.inference_time_p99 = percentiles['p99']

    # Calculate memory statistics
    if memory_samples:
        metrics.memory_usage_mean = statistics.mean(memory_samples)
        metrics.memory_usage_std = statistics.stdev(memory_samples) if len(memory_samples) > 1 else 0.0
        metrics.memory_usage_peak = max(memory_samples)

    # Test accuracy on the same samples
    try:
        # Filter features for prediction
        if model_feature_columns is not None:
            available_features = [f for f in model_feature_columns if f in feature_columns]
            if len(available_features) > 0:
                # Convert column names to indices for numpy array
                feature_indices = [feature_columns.index(f) for f in available_features]
                test_features = sample_features[:, feature_indices]
            else:
                test_features = sample_features
        else:
            # model_feature_columns is None - likely A6 SVM pipeline
            # Check if we need to drop duplicate NASM columns
            if hasattr(model, 'steps') and len(model.steps) > 0:
                first_step = model.steps[0][1]
                n_expected = getattr(first_step, 'n_features_in_', None)
                if n_expected is not None:
                    # Identify indices of duplicate NASM columns
                    dup_indices = [i for i, c in enumerate(feature_columns) if c in DUPLICATE_NASM_COLS]
                    # Get all indices except duplicate NASM columns
                    valid_indices = [i for i in range(len(feature_columns)) if i not in dup_indices]
                    if len(valid_indices) == n_expected:
                        # Select only the columns that match expected features
                        test_features = sample_features[:, valid_indices]
                    else:
                        # Fallback: slice to expected number of features
                        test_features = sample_features[:, :n_expected]
                else:
                    test_features = sample_features
            else:
                test_features = sample_features

        # Handle A6 SVM pipeline
        if model_feature_columns is None and hasattr(model, 'steps'):
            scaler_to_use = None
        else:
            scaler_to_use = scaler

        if scaler_to_use is not None:
            features = scaler_to_use.transform(test_features)
        else:
            features = test_features

        predictions = model.predict(features)

        # Calculate accuracy
        correct = np.sum(predictions == true_labels)
        metrics.predictions_correct = int(correct)
        metrics.predictions_total = len(true_labels)
        metrics.accuracy = correct / len(true_labels)

    except Exception as e:
        print(f"    ⚠️  Accuracy calculation failed: {e}")

    metrics.status = "SUCCESS"
    return metrics


def run_benchmark(
    num_samples: int = DEFAULT_NUM_SAMPLES,
    num_repeats: int = DEFAULT_NUM_REPEATES,
    output_file: Optional[str] = None,
    single_sample_mode: bool = False
) -> BenchmarkResults:
    """Run complete benchmark on all models.

    Args:
        num_samples: Number of samples to benchmark
        num_repeats: Number of repetitions per sample
        output_file: Optional output file path for results
        single_sample_mode: If True, measure each sample individually (for single sample latency)

    Returns:
        BenchmarkResults object with all results
    """
    print("=" * 70)
    print("STANDARDIZED TIMING BENCHMARKING FRAMEWORK")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Number of samples: {num_samples}")
    print(f"  Number of repeats per sample: {num_repeats}")
    print(f"  Total predictions per model: {num_samples * num_repeats}")
    print()

    # Load data
    print("Loading data...")
    data = load_and_prepare_data()
    print()

    # Create samples
    sample_features, true_labels = create_samples_from_test_data(data, num_samples)
    print(f"Created {num_samples} test samples for benchmarking")
    print()

    # Define models to benchmark
    models_to_benchmark = [
        ('A4 Random Forest', a4_rf),
        ('A5 Ensemble', a5_ensemnble),
        ('A5b Adaboost', a5b_adaboost),
        ('A5b Bagging Trees', a5b_bagging_tree),
        ('A6 SVM', a6_svm),
    ]

    # Initialize results
    results = BenchmarkResults(
        timestamp=datetime.now().isoformat(),
        num_samples=num_samples,
        num_repeats=num_repeats
    )

    # Benchmark each model
    print("=" * 70)
    print("Running Benchmarks")
    print("=" * 70)

    for model_name, model_path in models_to_benchmark:
        metrics = benchmark_single_model(
            model_name=model_name,
            model_path=model_path,
            sample_features=sample_features,
            true_labels=true_labels,
            feature_columns=data['feature_columns'],
            num_repeats=num_repeats,
            single_sample_mode=single_sample_mode
        )
        results.models[model_name] = metrics

        # Print summary for this model
        print(f"\n  {model_name} Results:")
        print(f"    Status: {metrics.status}")

        if metrics.status == "SUCCESS":
            print(f"    Inference Time:")
            print(f"      Mean: {metrics.inference_time_mean*1000:.3f} ms")
            print(f"      Std:  {metrics.inference_time_std*1000:.3f} ms")
            print(f"      P50:  {metrics.inference_time_p50*1000:.3f} ms")
            print(f"      P95:  {metrics.inference_time_p95*1000:.3f} ms")
            print(f"      P99:  {metrics.inference_time_p99*1000:.3f} ms")
            print(f"    Memory Usage:")
            print(f"      Mean: {metrics.memory_usage_mean/1024:.1f} KB")
            print(f"      Peak: {metrics.memory_usage_peak/1024:.1f} KB")
            print(f"    Accuracy: {metrics.accuracy*100:.1f}% ({metrics.predictions_correct}/{metrics.predictions_total})")
            print(f"    Model Size: {metrics.model_size_bytes/1024:.1f} KB")
            print(f"    Features: {metrics.num_features}")
        else:
            print(f"    Error: {metrics.error_message}")
        print()

    # Save results
    if output_file is None:
        output_file = os.path.join(OUTPUT_DIR, f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    json_output = results.to_json(output_file)
    print(f"Results saved to: {output_file}")

    return results


def run_single_sample_benchmark(
    num_samples: int = DEFAULT_NUM_SAMPLES,
    num_repeats: int = DEFAULT_NUM_REPEATES,
    output_file: Optional[str] = None
) -> BenchmarkResults:
    """Run benchmark with single sample prediction latency measurement.

    This function measures the latency for individual predictions rather than
    batch predictions, giving a more realistic view of single sample performance.

    Args:
        num_samples: Number of samples to benchmark
        num_repeats: Number of repetitions per sample
        output_file: Optional output file path for results

    Returns:
        BenchmarkResults object with all results
    """
    return run_benchmark(
        num_samples=num_samples,
        num_repeats=num_repeats,
        output_file=output_file,
        single_sample_mode=True
    )


# ============================================================================
# Comparison and Analysis Functions
# ============================================================================

def print_comparison_table(results: BenchmarkResults):
    """Print a formatted comparison table of all models."""
    print("\n" + "=" * 90)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 90)

    # Header
    print(f"{'Model':<20} {'Time (ms)':<15} {'Std':<10} {'P95':<10} {'Acc (%)':<10} {'Mem (KB)':<12} {'Size (KB)':<12}")
    print("-" * 90)

    # Sort by inference time for comparison
    sorted_models = sorted(
        results.models.items(),
        key=lambda x: x[1].inference_time_mean if x[1].status == "SUCCESS" else float('inf')
    )

    for model_name, metrics in sorted_models:
        if metrics.status == "SUCCESS":
            time_ms = metrics.inference_time_mean * 1000
            std_ms = metrics.inference_time_std * 1000
            p95_ms = metrics.inference_time_p95 * 1000
            acc = metrics.accuracy * 100
            mem_kb = metrics.memory_usage_mean / 1024
            size_kb = metrics.model_size_bytes / 1024

            print(f"{model_name:<20} {time_ms:<15.3f} {std_ms:<10.3f} {p95_ms:<10.3f} {acc:<10.1f} {mem_kb:<12.1f} {size_kb:<12.1f}")
        else:
            print(f"{model_name:<20} {'ERROR':<15} {'-':<10} {'-':<10} {'-':<10} {'-':<12} {'-':<12}")

    print("=" * 90)


def find_optimal_model(results: BenchmarkResults, priority: str = "speed"):
    """Find the optimal model based on specified criteria.

    Args:
        results: BenchmarkResults object
        priority: Optimization priority ("speed", "accuracy", "memory", "balanced")

    Returns:
        Tuple of (best_model_name, best_metrics)
    """
    valid_models = {
        name: metrics for name, metrics in results.models.items()
        if metrics.status == "SUCCESS"
    }

    if not valid_models:
        return None, None

    if priority == "speed":
        # Minimum inference time
        best = min(valid_models.items(), key=lambda x: x[1].inference_time_mean)
    elif priority == "accuracy":
        # Maximum accuracy
        best = max(valid_models.items(), key=lambda x: x[1].accuracy)
    elif priority == "memory":
        # Minimum memory usage
        best = min(valid_models.items(), key=lambda x: x[1].memory_usage_mean)
    elif priority == "balanced":
        # Balanced score: weighted combination
        def balanced_score(item):
            metrics = item[1]
            # Normalize and combine metrics
            time_score = metrics.inference_time_mean
            acc_score = 1 - metrics.accuracy
            mem_score = metrics.memory_usage_mean / 1000000  # Scale down

            # Weighted sum (weights can be adjusted)
            return 0.5 * time_score + 0.3 * acc_score + 0.2 * mem_score

        best = min(valid_models.items(), key=balanced_score)
    else:
        best = min(valid_models.items(), key=lambda x: x[1].inference_time_mean)

    return best


def print_recommendations(results: BenchmarkResults):
    """Print model recommendations based on different criteria."""
    print("\n" + "=" * 70)
    print("MODEL RECOMMENDATIONS")
    print("=" * 70)

    criteria = [
        ("Fastest Inference", "speed"),
        ("Highest Accuracy", "accuracy"),
        ("Lowest Memory Usage", "memory"),
        ("Best Balanced Performance", "balanced"),
    ]

    for description, priority in criteria:
        model_name, metrics = find_optimal_model(results, priority)
        if model_name:
            print(f"\n{description}:")
            print(f"  Model: {model_name}")
            if priority == "speed":
                print(f"  Inference Time: {metrics.inference_time_mean*1000:.3f} ms")
            elif priority == "accuracy":
                print(f"  Accuracy: {metrics.accuracy*100:.1f}%")
            elif priority == "memory":
                print(f"  Memory Usage: {metrics.memory_usage_mean/1024:.1f} KB")
            elif priority == "balanced":
                print(f"  Inference Time: {metrics.inference_time_mean*1000:.3f} ms")
                print(f"  Accuracy: {metrics.accuracy*100:.1f}%")
                print(f"  Memory Usage: {metrics.memory_usage_mean/1024:.1f} KB")
        else:
            print(f"\n{description}:")
            print("  No valid models found")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point for the benchmarking framework."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Standardized Timing Benchmarking Framework for Classification Models'
    )
    parser.add_argument(
        '--samples', '-n',
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help=f'Number of samples to benchmark (default: {DEFAULT_NUM_SAMPLES})'
    )
    parser.add_argument(
        '--repeats', '-r',
        type=int,
        default=DEFAULT_NUM_REPEATES,
        help=f'Number of repeats per sample (default: {DEFAULT_NUM_REPEATES})'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=DEFAULT_OUTPUT_FILE,
        help='Output file for results (default: benchmark_results/timestamp.json)'
    )
    parser.add_argument(
        '--compare', '-c',
        action='store_true',
        help='Print comparison table after benchmarking'
    )
    parser.add_argument(
        '--recommend', '-R',
        action='store_true',
        help='Print model recommendations after benchmarking'
    )
    parser.add_argument(
        '--single-sample', '-s',
        action='store_true',
        help='Measure single sample prediction latency (default: batch mode)'
    )

    args = parser.parse_args()

    # Run benchmark
    if args.single_sample:
        results = run_single_sample_benchmark(
            num_samples=args.samples,
            num_repeats=args.repeats,
            output_file=args.output
        )
    else:
        results = run_benchmark(
            num_samples=args.samples,
            num_repeats=args.repeats,
            output_file=args.output
        )

    # Print comparison table if requested
    if args.compare:
        print_comparison_table(results)

    # Print recommendations if requested
    if args.recommend:
        print_recommendations(results)

    # Return results for programmatic use
    return results


if __name__ == "__main__":
    results = main()
