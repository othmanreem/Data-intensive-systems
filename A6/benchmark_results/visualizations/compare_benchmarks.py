#!/usr/bin/env python3
"""
Script to compare response times (inference times) from two benchmark JSON files.
Generates a visualization comparing the models from both benchmarks.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# File paths
benchmark_path = Path(__file__).parent / "../benchmark_20260310_090052.json"
single_benchmark_path = Path(__file__).parent / "../single_benchmark_20260310_090011.json"

# Load benchmark data
with open(benchmark_path, 'r') as f:
    benchmark_data = json.load(f)

with open(single_benchmark_path, 'r') as f:
    single_benchmark_data = json.load(f)

# Extract model data
def extract_model_data(data_dict):
    models = {}
    for model_name, model_info in data_dict.get('models', {}).items():
        models[model_name] = {
            'mean': model_info.get('inference_time_mean', 0),
            'std': model_info.get('inference_time_std', 0),
            'min': model_info.get('inference_time_min', 0),
            'max': model_info.get('inference_time_max', 0),
            'p50': model_info.get('inference_time_p50', 0),
            'p95': model_info.get('inference_time_p95', 0),
            'p99': model_info.get('inference_time_p99', 0),
            'accuracy': model_info.get('accuracy', 0),
            'timing_samples': model_info.get('timing_samples', [])
        }
    return models

benchmark_models = extract_model_data(benchmark_data)
single_benchmark_models = extract_model_data(single_benchmark_data)

# Get all model names (should be the same in both)
all_model_names = sorted(benchmark_models.keys())

# Create figure with subplots
fig = plt.figure(figsize=(16, 10))

# 1. Bar chart comparing mean inference times
ax1 = fig.add_subplot(2, 3, 1)
x = np.arange(len(all_model_names))
width = 0.35

benchmark_means = [benchmark_models[m]['mean'] * 1000 for m in all_model_names]  # Convert to ms
single_means = [single_benchmark_models[m]['mean'] * 1000 for m in all_model_names]  # Convert to ms

bars1 = ax1.bar(x - width/2, benchmark_means, width, label='Multi-benchmark (100 samples)', alpha=0.8)
bars2 = ax1.bar(x + width/2, single_means, width, label='Single-benchmark (10 samples)', alpha=0.8)

ax1.set_xlabel('Model')
ax1.set_ylabel('Mean Inference Time (ms)')
ax1.set_title('Comparison of Mean Inference Times')
ax1.set_xticks(x)
ax1.set_xticklabels(all_model_names, rotation=45, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax1.annotate(f'{height:.3f}',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3),
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=8)

for bar in bars2:
    height = bar.get_height()
    ax1.annotate(f'{height:.3f}',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3),
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=8)

# 2. Box plot comparing timing distributions
ax2 = fig.add_subplot(2, 3, 2)

# Prepare data for box plot
all_data = []
labels = []
colors = []

for i, model_name in enumerate(all_model_names):
    benchmark_samples = benchmark_models[model_name]['timing_samples'][:10]  # Use first 10 for comparison
    single_samples = single_benchmark_models[model_name]['timing_samples'][:10]  # Use first 10 for comparison

    # Convert to ms
    benchmark_ms = [s * 1000 for s in benchmark_samples]
    single_ms = [s * 1000 for s in single_samples]

    all_data.append(benchmark_ms)
    all_data.append(single_ms)
    labels.append(f'{model_name}\nMulti')
    labels.append(f'{model_name}\nSingle')
    colors.extend([f'C{i}', f'C{i}'])

bp = ax2.boxplot(all_data, labels=labels, patch_artist=True, vert=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax2.set_xlabel('Model (Benchmark Type)')
ax2.set_ylabel('Inference Time (ms)')
ax2.set_title('Distribution of Inference Times (Box Plot)')
ax2.tick_params(axis='x', rotation=45)
ax2.grid(axis='y', alpha=0.3)

# 3. Comparison scatter plot with accuracy
ax3 = fig.add_subplot(2, 3, 3)

benchmark_accs = [benchmark_models[m]['accuracy'] * 100 for m in all_model_names]
single_accs = [single_benchmark_models[m]['accuracy'] * 100 for m in all_model_names]
benchmark_times = [benchmark_models[m]['mean'] * 1000 for m in all_model_names]
single_times = [single_benchmark_models[m]['mean'] * 1000 for m in all_model_names]

# Create scatter plot
for i, model_name in enumerate(all_model_names):
    ax3.scatter([benchmark_times[i]], [benchmark_accs[i]], marker='o', s=100,
                label=f'{model_name} (Multi)', alpha=0.8, color=f'C{i}')
    ax3.scatter([single_times[i]], [single_accs[i]], marker='s', s=100,
                label=f'{model_name} (Single)', alpha=0.8, color=f'C{i}')

ax3.set_xlabel('Mean Inference Time (ms)')
ax3.set_ylabel('Accuracy (%)')
ax3.set_title('Accuracy vs Inference Time Comparison')
ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
ax3.grid(True, alpha=0.3)

# 4. Percentile comparison
ax4 = fig.add_subplot(2, 3, 4)

x = np.arange(len(all_model_names))
width = 0.25

benchmark_p50 = [benchmark_models[m]['p50'] * 1000 for m in all_model_names]
benchmark_p95 = [benchmark_models[m]['p95'] * 1000 for m in all_model_names]
benchmark_p99 = [benchmark_models[m]['p99'] * 1000 for m in all_model_names]

single_p50 = [single_benchmark_models[m]['p50'] * 1000 for m in all_model_names]
single_p95 = [single_benchmark_models[m]['p95'] * 1000 for m in all_model_names]
single_p99 = [single_benchmark_models[m]['p99'] * 1000 for m in all_model_names]

bars_p50 = ax4.bar(x - width, benchmark_p50, width, label='P50 (Multi)', alpha=0.8)
bars_p95 = ax4.bar(x, benchmark_p95, width, label='P95 (Multi)', alpha=0.8)
bars_p99 = ax4.bar(x + width, benchmark_p99, width, label='P99 (Multi)', alpha=0.8)

# Single benchmark percentiles (offset)
ax4.bar(x - width + 0.05, single_p50, width*0.8, label='P50 (Single)', alpha=0.6, hatch='//')
ax4.bar(x + 0.05, single_p95, width*0.8, label='P95 (Single)', alpha=0.6, hatch='//')
ax4.bar(x + width + 0.05, single_p99, width*0.8, label='P99 (Single)', alpha=0.6, hatch='//')

ax4.set_xlabel('Model')
ax4.set_ylabel('Inference Time (ms)')
ax4.set_title('Percentile Comparison (P50, P95, P99)')
ax4.set_xticks(x)
ax4.set_xticklabels(all_model_names, rotation=45, ha='right')
ax4.legend(fontsize='small')
ax4.grid(axis='y', alpha=0.3)

# 5. Standard deviation comparison
ax5 = fig.add_subplot(2, 3, 5)

benchmark_std = [benchmark_models[m]['std'] * 1000 for m in all_model_names]
single_std = [single_benchmark_models[m]['std'] * 1000 for m in all_model_names]

x = np.arange(len(all_model_names))
width = 0.35

bars_std1 = ax5.bar(x - width/2, benchmark_std, width, label='Multi-benchmark', alpha=0.8)
bars_std2 = ax5.bar(x + width/2, single_std, width, label='Single-benchmark', alpha=0.8)

ax5.set_xlabel('Model')
ax5.set_ylabel('Standard Deviation (ms)')
ax5.set_title('Standard Deviation of Inference Times')
ax5.set_xticks(x)
ax5.set_xticklabels(all_model_names, rotation=45, ha='right')
ax5.legend()
ax5.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars_std1:
    height = bar.get_height()
    ax5.annotate(f'{height:.4f}',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3),
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=7)

for bar in bars_std2:
    height = bar.get_height()
    ax5.annotate(f'{height:.4f}',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3),
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=7)

# 6. Summary statistics table
ax6 = fig.add_subplot(2, 3, 6)
ax6.axis('off')

# Create table data
table_data = []
for model_name in all_model_names:
    row = [
        model_name,
        f"{benchmark_models[model_name]['mean']*1000:.3f} ± {benchmark_models[model_name]['std']*1000:.3f}",
        f"{benchmark_models[model_name]['min']*1000:.3f}",
        f"{benchmark_models[model_name]['max']*1000:.3f}",
        f"{benchmark_models[model_name]['accuracy']*100:.1f}%",
        f"{single_benchmark_models[model_name]['mean']*1000:.3f} ± {single_benchmark_models[model_name]['std']*1000:.3f}",
        f"{single_benchmark_models[model_name]['min']*1000:.3f}",
        f"{single_benchmark_models[model_name]['max']*1000:.3f}",
        f"{single_benchmark_models[model_name]['accuracy']*100:.1f}%"
    ]
    table_data.append(row)

columns = ['Model', 'Mean ± Std (ms)', 'Min (ms)', 'Max (ms)', 'Acc (%)',
           'Mean ± Std (ms)', 'Min (ms)', 'Max (ms)', 'Acc (%)']
row_labels = ['Multi', 'Single'] * len(all_model_names)

# Create table
table = ax6.table(cellText=table_data, colLabels=columns, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.1, 1.8)

# Style the table
for i in range(len(all_model_names)):
    for j in range(len(columns)):
        cell = table[(i+1, j)]
        cell.set_height(0.4)
        if j < 5:
            cell.set_facecolor('#f0f0f0')  # Light gray for multi-benchmark columns
        else:
            cell.set_facecolor('#e0e0f0')  # Light blue for single-benchmark columns

ax6.set_title('Summary Statistics Comparison', fontsize=12, pad=20)

# Save each subplot as a separate PNG image
output_dir = Path(__file__).parent

# 1. Bar chart comparing mean inference times
fig1, ax1_single = plt.subplots(figsize=(10, 6))
x = np.arange(len(all_model_names))
width = 0.35
benchmark_means = [benchmark_models[m]['mean'] * 1000 for m in all_model_names]
single_means = [single_benchmark_models[m]['mean'] * 1000 for m in all_model_names]
bars1 = ax1_single.bar(x - width/2, benchmark_means, width, label='Multi-benchmark (100 samples)', alpha=0.8)
bars2 = ax1_single.bar(x + width/2, single_means, width, label='Single-benchmark (10 samples)', alpha=0.8)
ax1_single.set_xlabel('Model')
ax1_single.set_ylabel('Mean Inference Time (ms)')
ax1_single.set_title('Comparison of Mean Inference Times')
ax1_single.set_xticks(x)
ax1_single.set_xticklabels(all_model_names, rotation=45, ha='right')
ax1_single.legend()
ax1_single.grid(axis='y', alpha=0.3)
for bar in bars1:
    height = bar.get_height()
    ax1_single.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
for bar in bars2:
    height = bar.get_height()
    ax1_single.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig(output_dir / "mean_inference_times.png", dpi=300, bbox_inches='tight')
plt.close(fig1)
print(f"Saved: mean_inference_times.png")

# 2. Box plot comparing timing distributions
fig2, ax2_single = plt.subplots(figsize=(12, 6))
all_data = []
labels = []
colors = []
for i, model_name in enumerate(all_model_names):
    benchmark_samples = benchmark_models[model_name]['timing_samples'][:10]
    single_samples = single_benchmark_models[model_name]['timing_samples'][:10]
    benchmark_ms = [s * 1000 for s in benchmark_samples]
    single_ms = [s * 1000 for s in single_samples]
    all_data.append(benchmark_ms)
    all_data.append(single_ms)
    labels.append(f'{model_name}\nMulti')
    labels.append(f'{model_name}\nSingle')
    colors.extend([f'C{i}', f'C{i}'])
bp = ax2_single.boxplot(all_data, labels=labels, patch_artist=True, vert=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax2_single.set_xlabel('Model (Benchmark Type)')
ax2_single.set_ylabel('Inference Time (ms)')
ax2_single.set_title('Distribution of Inference Times (Box Plot)')
ax2_single.tick_params(axis='x', rotation=45)
ax2_single.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / "inference_time_distribution.png", dpi=300, bbox_inches='tight')
plt.close(fig2)
print(f"Saved: inference_time_distribution.png")

# 3. Comparison scatter plot with accuracy
fig3, ax3_single = plt.subplots(figsize=(10, 6))
benchmark_accs = [benchmark_models[m]['accuracy'] * 100 for m in all_model_names]
single_accs = [single_benchmark_models[m]['accuracy'] * 100 for m in all_model_names]
benchmark_times = [benchmark_models[m]['mean'] * 1000 for m in all_model_names]
single_times = [single_benchmark_models[m]['mean'] * 1000 for m in all_model_names]
for i, model_name in enumerate(all_model_names):
    ax3_single.scatter([benchmark_times[i]], [benchmark_accs[i]], marker='o', s=100, label=f'{model_name} (Multi)', alpha=0.8, color=f'C{i}')
    ax3_single.scatter([single_times[i]], [single_accs[i]], marker='s', s=100, label=f'{model_name} (Single)', alpha=0.8, color=f'C{i}')
ax3_single.set_xlabel('Mean Inference Time (ms)')
ax3_single.set_ylabel('Accuracy (%)')
ax3_single.set_title('Accuracy vs Inference Time Comparison')
ax3_single.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
ax3_single.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / "accuracy_vs_inference_time.png", dpi=300, bbox_inches='tight')
plt.close(fig3)
print(f"Saved: accuracy_vs_inference_time.png")

# 4. Percentile comparison
fig4, ax4_single = plt.subplots(figsize=(12, 6))
x = np.arange(len(all_model_names))
width = 0.25
benchmark_p50 = [benchmark_models[m]['p50'] * 1000 for m in all_model_names]
benchmark_p95 = [benchmark_models[m]['p95'] * 1000 for m in all_model_names]
benchmark_p99 = [benchmark_models[m]['p99'] * 1000 for m in all_model_names]
single_p50 = [single_benchmark_models[m]['p50'] * 1000 for m in all_model_names]
single_p95 = [single_benchmark_models[m]['p95'] * 1000 for m in all_model_names]
single_p99 = [single_benchmark_models[m]['p99'] * 1000 for m in all_model_names]
bars_p50 = ax4_single.bar(x - width, benchmark_p50, width, label='P50 (Multi)', alpha=0.8)
bars_p95 = ax4_single.bar(x, benchmark_p95, width, label='P95 (Multi)', alpha=0.8)
bars_p99 = ax4_single.bar(x + width, benchmark_p99, width, label='P99 (Multi)', alpha=0.8)
ax4_single.bar(x - width + 0.05, single_p50, width*0.8, label='P50 (Single)', alpha=0.6, hatch='//')
ax4_single.bar(x + 0.05, single_p95, width*0.8, label='P95 (Single)', alpha=0.6, hatch='//')
ax4_single.bar(x + width + 0.05, single_p99, width*0.8, label='P99 (Single)', alpha=0.6, hatch='//')
ax4_single.set_xlabel('Model')
ax4_single.set_ylabel('Inference Time (ms)')
ax4_single.set_title('Percentile Comparison (P50, P95, P99)')
ax4_single.set_xticks(x)
ax4_single.set_xticklabels(all_model_names, rotation=45, ha='right')
ax4_single.legend(fontsize='small')
ax4_single.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / "percentile_comparison.png", dpi=300, bbox_inches='tight')
plt.close(fig4)
print(f"Saved: percentile_comparison.png")

# 5. Standard deviation comparison
fig5, ax5_single = plt.subplots(figsize=(10, 6))
benchmark_std = [benchmark_models[m]['std'] * 1000 for m in all_model_names]
single_std = [single_benchmark_models[m]['std'] * 1000 for m in all_model_names]
x = np.arange(len(all_model_names))
width = 0.35
bars_std1 = ax5_single.bar(x - width/2, benchmark_std, width, label='Multi-benchmark', alpha=0.8)
bars_std2 = ax5_single.bar(x + width/2, single_std, width, label='Single-benchmark', alpha=0.8)
ax5_single.set_xlabel('Model')
ax5_single.set_ylabel('Standard Deviation (ms)')
ax5_single.set_title('Standard Deviation of Inference Times')
ax5_single.set_xticks(x)
ax5_single.set_xticklabels(all_model_names, rotation=45, ha='right')
ax5_single.legend()
ax5_single.grid(axis='y', alpha=0.3)
for bar in bars_std1:
    height = bar.get_height()
    ax5_single.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=7)
for bar in bars_std2:
    height = bar.get_height()
    ax5_single.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=7)
plt.tight_layout()
plt.savefig(output_dir / "standard_deviation_comparison.png", dpi=300, bbox_inches='tight')
plt.close(fig5)
print(f"Saved: standard_deviation_comparison.png")

# 6. Summary statistics table
fig6, ax6_single = plt.subplots(figsize=(14, 6))
ax6_single.axis('off')
table_data = []
for model_name in all_model_names:
    row = [
        model_name,
        f"{benchmark_models[model_name]['mean']*1000:.3f} ± {benchmark_models[model_name]['std']*1000:.3f}",
        f"{benchmark_models[model_name]['min']*1000:.3f}",
        f"{benchmark_models[model_name]['max']*1000:.3f}",
        f"{benchmark_models[model_name]['accuracy']*100:.1f}%",
        f"{single_benchmark_models[model_name]['mean']*1000:.3f} ± {single_benchmark_models[model_name]['std']*1000:.3f}",
        f"{single_benchmark_models[model_name]['min']*1000:.3f}",
        f"{single_benchmark_models[model_name]['max']*1000:.3f}",
        f"{single_benchmark_models[model_name]['accuracy']*100:.1f}%"
    ]
    table_data.append(row)
columns = ['Model', 'Mean ± Std (ms)', 'Min (ms)', 'Max (ms)', 'Acc (%)',
           'Mean ± Std (ms)', 'Min (ms)', 'Max (ms)', 'Acc (%)']
table = ax6_single.table(cellText=table_data, colLabels=columns, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.1, 1.8)
for i in range(len(all_model_names)):
    for j in range(len(columns)):
        cell = table[(i+1, j)]
        cell.set_height(0.4)
        if j < 5:
            cell.set_facecolor('#f0f0f0')
        else:
            cell.set_facecolor('#e0e0f0')
ax6_single.set_title('Summary Statistics Comparison', fontsize=12, pad=20)
plt.tight_layout()
plt.savefig(output_dir / "summary_statistics.png", dpi=300, bbox_inches='tight')
plt.close(fig6)
print(f"Saved: summary_statistics.png")

print(f"\nAll individual visualizations saved to: {output_dir}")

# Also save as interactive HTML
html_output = Path(__file__).parent / "response_time_comparison.html"
with open(html_output, 'w') as f:
    f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>Benchmark Response Time Comparison</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ text-align: center; }}
        .chart {{ max-width: 1200px; margin: 0 auto; }}
        .model-section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .model-title {{ font-weight: bold; font-size: 1.2em; margin-bottom: 10px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f4f4f4; }}
    </style>
</head>
<body>
    <h1>Benchmark Response Time Comparison</h1>
    <p><strong>Multi-benchmark:</strong> {benchmark_data['num_samples']} samples, {benchmark_data['num_repeats']} repeats</p>
    <p><strong>Single-benchmark:</strong> {single_benchmark_data['num_samples']} samples, {single_benchmark_data['num_repeats']} repeats</p>
    <p><img src="response_time_comparison.png" alt="Comparison Chart" class="chart"></p>
    <h2>Detailed Statistics</h2>
""")
    for model_name in all_model_names:
        f.write(f"""
    <div class="model-section">
        <div class="model-title">{model_name}</div>
        <table>
            <tr>
                <th>Metric</th>
                <th>Multi-benchmark</th>
                <th>Single-benchmark</th>
                <th>Change</th>
            </tr>
            <tr>
                <td>Mean (ms)</td>
                <td>{benchmark_models[model_name]['mean']*1000:.4f}</td>
                <td>{single_benchmark_models[model_name]['mean']*1000:.4f}</td>
                <td>{((single_benchmark_models[model_name]['mean'] - benchmark_models[model_name]['mean']) / benchmark_models[model_name]['mean'] * 100):.1f}%</td>
            </tr>
            <tr>
                <td>Std (ms)</td>
                <td>{benchmark_models[model_name]['std']*1000:.4f}</td>
                <td>{single_benchmark_models[model_name]['std']*1000:.4f}</td>
                <td>{((single_benchmark_models[model_name]['std'] - benchmark_models[model_name]['std']) / benchmark_models[model_name]['std'] * 100):.1f}%</td>
            </tr>
            <tr>
                <td>Min (ms)</td>
                <td>{benchmark_models[model_name]['min']*1000:.4f}</td>
                <td>{single_benchmark_models[model_name]['min']*1000:.4f}</td>
                <td>{((single_benchmark_models[model_name]['min'] - benchmark_models[model_name]['min']) / benchmark_models[model_name]['min'] * 100):.1f}%</td>
            </tr>
            <tr>
                <td>Max (ms)</td>
                <td>{benchmark_models[model_name]['max']*1000:.4f}</td>
                <td>{single_benchmark_models[model_name]['max']*1000:.4f}</td>
                <td>{((single_benchmark_models[model_name]['max'] - benchmark_models[model_name]['max']) / benchmark_models[model_name]['max'] * 100):.1f}%</td>
            </tr>
            <tr>
                <td>Accuracy</td>
                <td>{benchmark_models[model_name]['accuracy']*100:.1f}%</td>
                <td>{single_benchmark_models[model_name]['accuracy']*100:.1f}%</td>
                <td>{(single_benchmark_models[model_name]['accuracy'] - benchmark_models[model_name]['accuracy']) * 100:.1f}pp</td>
            </tr>
        </table>
    </div>
""")
    f.write("""
</body>
</html>""")
print(f"HTML report saved to: {html_output}")

# Print summary to console
print("\n=== Summary ===")
print(f"Multi-benchmark: {benchmark_data['num_samples']} samples, {benchmark_data['num_repeats']} repeats")
print(f"Single-benchmark: {single_benchmark_data['num_samples']} samples, {single_benchmark_data['num_repeats']} repeats")
print("\nModel Comparison:")
print("-" * 80)
for model_name in all_model_names:
    b_mean = benchmark_models[model_name]['mean'] * 1000
    s_mean = single_benchmark_models[model_name]['mean'] * 1000
    change = ((s_mean - b_mean) / b_mean * 100)
    print(f"{model_name:20s} | Multi: {b_mean:6.3f}ms | Single: {s_mean:6.3f}ms | Change: {change:+6.1f}%")
