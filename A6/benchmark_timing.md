# Standardized Timing Benchmarking Framework

A comprehensive benchmarking framework for fair and consistent comparison of classification models (A4, A5, A5b, A6).

## Features

This framework provides standardized metrics for model comparison:

- **Inference Time**: Mean, standard deviation, min, max, and percentiles (P50, P95, P99)
- **Memory Usage**: Mean, standard deviation, and peak memory consumption
- **Prediction Accuracy**: Correct predictions and accuracy percentage
- **Model Characteristics**: Model size, number of features, model type
- **Consistent Data Pipeline**: Uses the same data processing for all models

## Installation

No additional dependencies required. Uses existing project dependencies:
- `numpy`
- `pandas`
- `scikit-learn`
- `pickle` (standard library)

## Usage

### Basic Usage

```bash
python benchmark_timing.py
```

### Advanced Usage

```bash
# Specify number of samples and repeats
python benchmark_timing.py --samples 200 --repeats 20

# Save results to specific file
python benchmark_timing.py --output results/my_benchmark.json

# Print comparison table
python benchmark_timing.py --compare

# Print model recommendations
python benchmark_timing.py --recommend

# All options combined
python benchmark_timing.py -n 150 -r 15 -o results/benchmark.json -c -R
```

### Command Line Arguments

| Argument | Short | Description | Default |
|----------|-------|-------------|---------|
| `--samples` | `-n` | Number of test samples | 100 |
| `--repeats` | `-r` | Number of repetitions per sample | 10 |
| `--output` | `-o` | Output file path for JSON results | Auto-generated |
| `--compare` | `-c` | Print comparison table | False |
| `--recommend` | `-R` | Print model recommendations | False |

## Output

### Console Output

The framework prints real-time progress and results:

```
======================================================================
STANDARDIZED TIMING BENCHMARKING FRAMEWORK
======================================================================

Configuration:
  Number of samples: 100
  Number of repeats per sample: 10
  Total predictions per model: 1000

Loading data...
  Movement features shape: (1000, 150)
  Weak link scores shape: (1000, 20)
  Merged dataset shape: (1000, 165)
  Feature matrix shape: (1000, 160)
  Number of features: 160
  Number of classes: 14

======================================================================
Running Benchmarks
======================================================================

  Benchmarking A4 Random Forest...

  A4 Random Forest Results:
    Status: SUCCESS
    Inference Time:
      Mean: 1.234 ms
      Std:  0.123 ms
      P50:  1.200 ms
      P95:  1.500 ms
      P99:  1.800 ms
    Memory Usage:
      Mean: 256.5 KB
      Peak: 512.0 KB
    Accuracy: 78.5% (78/100)
    Model Size: 1250.0 KB
    Features: 160
```

### JSON Results

Results are saved to JSON format with all metrics:

```json
{
  "timestamp": "2024-01-15T10:30:45.123456",
  "num_samples": 100,
  "num_repeats": 10,
  "models": {
    "A4 Random Forest": {
      "model_name": "A4 Random Forest",
      "model_path": "../A4/models/weaklink_classifier_rf.pkl",
      "inference_time_mean": 0.001234,
      "inference_time_std": 0.000123,
      "inference_time_min": 0.001000,
      "inference_time_max": 0.001800,
      "inference_time_p50": 0.001200,
      "inference_time_p95": 0.001500,
      "inference_time_p99": 0.001800,
      "memory_usage_mean": 262656.0,
      "memory_usage_std": 10240.0,
      "memory_usage_peak": 524288.0,
      "accuracy": 0.785,
      "predictions_correct": 78,
      "predictions_total": 100,
      "model_size_bytes": 1280000,
      "num_features": 160,
      "num_parameters": 10,
      "model_type": "RandomForestClassifier",
      "timing_samples": [0.0012, 0.0013, ...],
      "memory_samples": [262144, 266240, ...],
      "status": "SUCCESS",
      "error_message": ""
    }
  }
}
```

## Model Comparison Table

With `--compare` flag, prints a formatted comparison:

```
==========================================================================
MODEL COMPARISON SUMMARY
==========================================================================
Model                Time (ms)       Std       P95       Acc (%)    Mem (KB)   Size (KB) 
--------------------------------------------------------------------------
A5b Adaboost         0.850           0.050     1.100     75.2       128.5      512.0
A5 Ensemble          1.100           0.080     1.350     79.8       256.3      768.0
A4 Random Forest     1.234           0.123     1.500     78.5       256.5      1250.0
A5b Bagging Trees    1.450           0.150     1.800     77.1       384.2      1024.0
A6 SVM               2.100           0.200     2.500     81.2       512.0      2048.0
==========================================================================
```

## Model Recommendations

With `--recommend` flag, provides optimal model suggestions:

```
======================================================================
MODEL RECOMMENDATIONS
======================================================================

Fastest Inference:
  Model: A5b Adaboost
  Inference Time: 0.850 ms

Highest Accuracy:
  Model: A6 SVM
  Accuracy: 81.2%

Lowest Memory Usage:
  Model: A5b Adaboost
  Memory Usage: 128.5 KB

Best Balanced Performance:
  Model: A5 Ensemble
  Inference Time: 1.100 ms
  Accuracy: 79.8%
  Memory Usage: 256.3 KB
```

## Benchmarking Metrics Explained

### Inference Time Metrics

| Metric | Description |
|--------|-------------|
| **Mean** | Average inference time across all repetitions |
| **Std** | Standard deviation (variability) |
| **Min/Max** | Fastest and slowest inference times |
| **P50** | Median (50th percentile) |
| **P95** | 95th percentile (95% of predictions are faster) |
| **P99** | 99th percentile (99% of predictions are faster) |

### Memory Metrics

| Metric | Description |
|--------|-------------|
| **Mean** | Average memory usage |
| **Std** | Standard deviation of memory usage |
| **Peak** | Maximum memory consumed |

### Accuracy Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Percentage of correct predictions |
| **Predictions Correct/Total** | Raw counts |

## Implementation Details

### Data Pipeline

All models use the same data loading and preprocessing pipeline:
1. Load movement features and weaklink scores
2. Create WeakestLink target column
3. Merge datasets
4. Extract features (excluding ID, WeakestLink, EstimatedScore)
5. Train/test split (80/20, stratified, random_state=42)
6. StandardScaler fitted on training data

### Feature Handling

- A4 Random Forest model was trained WITH duplicate NASM columns
- Other models (A5, A5b, A6) were trained WITHOUT duplicate NASM columns
- The framework automatically filters features based on each model's expectations

### Memory Tracking

Uses Python's `tracemalloc` module for accurate memory measurement:
- Tracks memory before and after each prediction
- Records both current and peak memory usage

### Timing Precision

Uses `time.perf_counter()` for high-resolution timing measurements.

## Extending the Framework

### Adding New Models

1. Add model path to `all_classification.py`:
```python
a7_new_model = "../A7/models/new_model.pkl"
```

2. Import in `benchmark_timing.py`:
```python
from all_classification import (
    a4_rf,
    a5_ensemnble,
    a5b_adaboost,
    a5b_bagging_tree,
    a6_svm,
    a7_new_model,  # Add here
)
```

3. Add to `models_to_benchmark` list in `run_benchmark()`:
```python
models_to_benchmark = [
    ('A4 Random Forest', a4_rf),
    ('A5 Ensemble', a5_ensemnble),
    ('A5b Adaboost', a5b_adaboost),
    ('A5b Bagging Trees', a5b_bagging_tree),
    ('A6 SVM', a6_svm),
    ('A7 New Model', a7_new_model),  # Add here
]
```

### Adding New Metrics

Extend the `ModelMetrics` dataclass:

```python
@dataclass
class ModelMetrics:
    # ... existing fields ...
    new_metric: float = 0.0
```

Then calculate it in `benchmark_single_model()`.

## Troubleshooting

### Model Loading Errors

If a model fails to load:
1. Verify the model file exists at the specified path
2. Check that all required dependencies are installed
3. Ensure custom classes (like `AdaBoostEnsemble`) are importable

### Inconsistent Results

If results vary significantly between runs:
1. Increase `--repeats` for better averaging
2. Ensure no other processes are running during benchmarking
3. Check system resource utilization

### Memory Issues

If running out of memory:
1. Reduce `--samples` count
2. Close other applications
3. Check available system memory

## License

Part of the Data Intensive Systems project.
```

Now let me also create a simple comparison script for quick analysis:

<tool_call>
<function=edit_file>
<parameter=display_description>
Create simple comparison script for quick analysis
</parameter>
<parameter=path>
/home/cyclonaut/Dokumente/Studium Växjö/data intensive project/repo_march/Data-intensive-systems/A6/compare_benchmarks.py
</parameter>
<parameter=mode>
create
</parameter>
</function>
</tool_call>