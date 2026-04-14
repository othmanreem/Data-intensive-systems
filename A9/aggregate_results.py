"""
A9 Results Aggregation Script

Aggregates all cross-validation results from the cv_results folders
and produces summary tables for the A9_Report.ipynb notebook.

Usage:
    python aggregate_results.py
    
Output:
    - results_summary.csv: All experiment results in one table
    - results_summary.json: Same data in JSON format
    - Prints summary to console
"""

import os
import json
import re
import pandas as pd
from pathlib import Path

# Result directories
RESULT_DIRS = {
    'different_models': 'cv_results_different_models',
    'conv1d_variants': 'cv_results_conv1D_variants',
    'optimizer': 'cv_results_optimizer',
    'loss_optimizer': 'cv_results_loss_optimizer',
}


def parse_summary_txt(filepath):
    """Parse summary.txt file to extract metrics."""
    results = []
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Find all model sections with their metrics
    # Pattern matches lines like "  Optimizer: SGD, Loss: MSE" or just model names
    sections = re.split(r'\n(?=[A-Z_]+(?:_[A-Z]+)*\n-{10,})', content)
    
    for section in sections:
        if 'Val RMSE:' in section or 'Test RMSE:' in section:
            lines = section.strip().split('\n')
            model_name = lines[0].strip() if lines else 'unknown'
            
            # Skip separator lines
            if model_name.startswith('-'):
                continue
            
            row = {'model': model_name.lower()}
            
            # Extract metrics using regex
            val_rmse = re.search(r'Val RMSE:\s*([\d.]+)\s*±\s*([\d.]+)', section)
            val_mae = re.search(r'Val MAE:\s*([\d.]+)\s*±\s*([\d.]+)', section)
            val_r2 = re.search(r'Val R²:\s*([\d.]+)\s*±\s*([\d.]+)', section)
            test_rmse = re.search(r'Test RMSE:\s*([\d.]+)', section)
            test_mae = re.search(r'Test MAE:\s*([\d.]+)', section)
            test_r2 = re.search(r'Test R²:\s*([\d.]+)', section)
            best_fold = re.search(r'Best Fold:\s*(\d+)', section)
            
            if val_rmse:
                row['val_rmse_mean'] = float(val_rmse.group(1))
                row['val_rmse_std'] = float(val_rmse.group(2))
            if val_mae:
                row['val_mae_mean'] = float(val_mae.group(1))
                row['val_mae_std'] = float(val_mae.group(2))
            if val_r2:
                row['val_r2_mean'] = float(val_r2.group(1))
                row['val_r2_std'] = float(val_r2.group(2))
            if test_rmse:
                row['test_rmse'] = float(test_rmse.group(1))
            if test_mae:
                row['test_mae'] = float(test_mae.group(1))
            if test_r2:
                row['test_r2'] = float(test_r2.group(1))
            if best_fold:
                row['best_fold'] = int(best_fold.group(1))
            
            # Only add if we found some metrics
            if len(row) > 1:
                results.append(row)
    
    return results


def parse_summary_header(filepath):
    """Parse the header summary section of summary.txt."""
    results = []
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Pattern 1: Standard format (DENSE:, CONV1D:, CONV1D_V3: etc.)
    # Fixed to properly capture model names with underscores and numbers
    pattern1 = r'([A-Z][A-Z0-9_]*):\s*\n\s*Best Fold:\s*(\d+)\s*\n\s*Val RMSE:\s*([\d.]+)\s*±\s*([\d.]+)\s*\n\s*Val MAE:\s*([\d.]+)\s*±\s*([\d.]+)\s*\n\s*Val R²:\s*([\d.]+)\s*±\s*([\d.]+)(?:\s*\n\s*Test RMSE:\s*([\d.]+))?(?:\s*\n\s*Test MAE:\s*([\d.]+))?(?:\s*\n\s*Test R²:\s*([\d.]+))?'
    
    # Pattern 2: Optimizer format (Optimizer: SGD, etc.)
    pattern2 = r'Optimizer:\s*([A-Z]+)(?:,\s*Loss:\s*([A-Z]+))?\s*\n\s*Best Fold:\s*(\d+)\s*\n\s*Val RMSE:\s*([\d.]+)\s*±\s*([\d.]+)\s*\n\s*Val MAE:\s*([\d.]+)\s*±\s*([\d.]+)\s*\n\s*Val R²:\s*([\d.]+)\s*±\s*([\d.]+)(?:\s*\n\s*Test RMSE:\s*([\d.]+))?(?:\s*\n\s*Test MAE:\s*([\d.]+))?(?:\s*\n\s*Test R²:\s*([\d.]+))?'
    
    seen_models = set()
    
    # Try pattern 1
    matches = re.finditer(pattern1, content, re.MULTILINE)
    for match in matches:
        model_name = match.group(1).lower()
        
        # Skip detailed sections (those that start with model name and have "-----" after)
        # We only want the summary sections at the top
        pos = match.start()
        next_line_start = content.find('\n', match.end()) + 1
        if next_line_start < len(content):
            next_line = content[next_line_start:next_line_start+50]
            if '---' in next_line:
                continue
        
        # Skip duplicates
        if model_name in seen_models:
            continue
        seen_models.add(model_name)
            
        row = {
            'model': model_name,
            'best_fold': int(match.group(2)),
            'val_rmse_mean': float(match.group(3)),
            'val_rmse_std': float(match.group(4)),
            'val_mae_mean': float(match.group(5)),
            'val_mae_std': float(match.group(6)),
            'val_r2_mean': float(match.group(7)),
            'val_r2_std': float(match.group(8)),
        }
        if match.group(9):
            row['test_rmse'] = float(match.group(9))
        if match.group(10):
            row['test_mae'] = float(match.group(10))
        if match.group(11):
            row['test_r2'] = float(match.group(11))
        
        results.append(row)
    
    # Try pattern 2 for optimizer/loss variants
    matches = re.finditer(pattern2, content, re.MULTILINE)
    for match in matches:
        optimizer = match.group(1).lower()
        loss = match.group(2).lower() if match.group(2) else 'mse'
        model_name = f"conv1d_v3_{optimizer}_{loss}"
        
        # Skip duplicates
        if model_name in seen_models:
            continue
        seen_models.add(model_name)
        
        row = {
            'model': model_name,
            'best_fold': int(match.group(3)),
            'val_rmse_mean': float(match.group(4)),
            'val_rmse_std': float(match.group(5)),
            'val_mae_mean': float(match.group(6)),
            'val_mae_std': float(match.group(7)),
            'val_r2_mean': float(match.group(8)),
            'val_r2_std': float(match.group(9)),
        }
        if match.group(10):
            row['test_rmse'] = float(match.group(10))
        if match.group(11):
            row['test_mae'] = float(match.group(11))
        if match.group(12):
            row['test_r2'] = float(match.group(12))
        
        results.append(row)
    
    return results


def main():
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    all_results = []
    
    print("=" * 60)
    print("A9 Results Aggregation")
    print("=" * 60)
    
    # Load results from each experiment directory by parsing summary.txt
    for exp_name, dir_name in RESULT_DIRS.items():
        summary_path = Path(dir_name) / 'summary.txt'
        print(f"\nLoading from {summary_path}...")
        
        if not summary_path.exists():
            print(f"  Warning: {summary_path} not found")
            continue
        
        results = parse_summary_header(summary_path)
        print(f"  Found {len(results)} model results")
        
        for row in results:
            row['experiment'] = exp_name
            all_results.append(row)
    
    if not all_results:
        print("\nNo results found!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Reorder columns
    cols = ['experiment', 'model', 'best_fold', 'val_r2_mean', 'val_r2_std', 
            'val_rmse_mean', 'val_rmse_std', 'val_mae_mean', 'val_mae_std',
            'test_r2', 'test_rmse', 'test_mae']
    df = df[[c for c in cols if c in df.columns]]
    
    # Sort by test_r2 descending, then val_r2_mean
    df['sort_key'] = df['test_r2'].fillna(df['val_r2_mean'])
    df = df.sort_values('sort_key', ascending=False).drop('sort_key', axis=1)
    
    # Save to CSV
    df.to_csv('results_summary.csv', index=False)
    print(f"\nSaved to results_summary.csv ({len(df)} rows)")
    
    # Save to JSON
    df.to_json('results_summary.json', orient='records', indent=2)
    print(f"Saved to results_summary.json")
    
    # Print summary tables
    print("\n" + "=" * 60)
    print("SUMMARY: All Experiments Ranked by R²")
    print("=" * 60)
    
    print(df.to_string(index=False, float_format=lambda x: f'{x:.4f}' if pd.notna(x) else 'N/A'))
    
    # Best model
    print("\n" + "=" * 60)
    print("CHAMPION MODEL")
    print("=" * 60)
    best = df.iloc[0]
    print(f"Model: {best['model']}")
    print(f"Experiment: {best['experiment']}")
    if pd.notna(best.get('val_r2_mean')):
        print(f"Validation R²: {best['val_r2_mean']:.4f} ± {best['val_r2_std']:.4f}")
        print(f"Validation RMSE: {best['val_rmse_mean']:.4f} ± {best['val_rmse_std']:.4f}")
    if pd.notna(best.get('test_r2')):
        print(f"Test R²: {best['test_r2']:.4f}")
        print(f"Test RMSE: {best['test_rmse']:.4f}")
    
    # Dead ends (R² < 0.85)
    print("\n" + "=" * 60)
    print("DEAD ENDS (R² < 0.85)")
    print("=" * 60)
    r2_col = 'test_r2' if 'test_r2' in df.columns else 'val_r2_mean'
    dead_ends = df[df[r2_col] < 0.85]
    if len(dead_ends) > 0:
        print(dead_ends[['experiment', 'model', 'val_r2_mean', 'test_r2']].to_string(index=False))
    else:
        print("None found")
    
    # Group summaries
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARIES")
    print("=" * 60)
    
    for exp_name in RESULT_DIRS.keys():
        exp_df = df[df['experiment'] == exp_name]
        if len(exp_df) > 0:
            print(f"\n{exp_name.upper()}:")
            print(exp_df[['model', 'val_r2_mean', 'val_rmse_mean', 'test_r2', 'test_rmse']].to_string(index=False))


if __name__ == '__main__':
    main()
