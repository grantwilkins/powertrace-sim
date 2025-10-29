#!/usr/bin/env python3
"""
Aggregate training metrics from all configurations and generate LaTeX table.
"""

import argparse
import csv
import glob
import os
from collections import defaultdict
import numpy as np


def load_metrics_from_csv(csv_file):
    """Load metrics from a single CSV file."""
    metrics = {}
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metrics[row['metric']] = float(row['value'])
    return metrics


def parse_job_name(filename):
    """Parse model, hardware, and tp from filename."""
    # Format: model_hardware_tpX_metrics.csv
    basename = os.path.basename(filename)
    basename = basename.replace('_metrics.csv', '')
    parts = basename.split('_')

    # Handle model names with hyphens
    if 'deepseek-r1-distill' in basename:
        if '70b' in basename:
            model = 'deepseek-r1-distill-70b'
            hardware_tp = basename.replace('deepseek-r1-distill-70b_', '')
        else:
            model = 'deepseek-r1-distill-8b'
            hardware_tp = basename.replace('deepseek-r1-distill-8b_', '')
    elif 'llama-3-70b' in basename:
        model = 'llama-3-70b'
        hardware_tp = basename.replace('llama-3-70b_', '')
    elif 'llama-3-8b' in basename:
        model = 'llama-3-8b'
        hardware_tp = basename.replace('llama-3-8b_', '')
    else:
        return None, None, None

    # Parse hardware and tp
    hw_tp_parts = hardware_tp.split('_')
    hardware = hw_tp_parts[0]
    tp_str = hw_tp_parts[1].replace('tp', '')
    tp = int(tp_str)

    return model, hardware, tp


def aggregate_metrics(metrics_dir):
    """Aggregate metrics across all configurations."""
    # Structure: model -> hardware -> tp -> metrics
    all_metrics = defaultdict(lambda: defaultdict(dict))

    metric_files = glob.glob(os.path.join(metrics_dir, '*_metrics.csv'))

    for metric_file in metric_files:
        model, hardware, tp = parse_job_name(metric_file)
        if model is None:
            print(f"Warning: Could not parse filename: {metric_file}")
            continue

        metrics = load_metrics_from_csv(metric_file)
        all_metrics[model][hardware][tp] = metrics

    return all_metrics


def compute_model_statistics(all_metrics):
    """Compute mean and std across hardware and TP for each model."""
    model_stats = {}

    for model in all_metrics:
        # Collect all metric values across hardware and tp
        val_f1_values = []
        autocorr_r2_values = []
        transition_mae_values = []
        ece_values = []

        for hardware in all_metrics[model]:
            for tp in all_metrics[model][hardware]:
                metrics = all_metrics[model][hardware][tp]
                val_f1_values.append(metrics.get('val_f1', 0))
                autocorr_r2_values.append(metrics.get('autocorr_r2', 0))
                transition_mae_values.append(metrics.get('transition_mae', 0))
                ece_values.append(metrics.get('ece', 0))

        model_stats[model] = {
            'val_f1_mean': np.mean(val_f1_values),
            'val_f1_std': np.std(val_f1_values),
            'autocorr_r2_mean': np.mean(autocorr_r2_values),
            'autocorr_r2_std': np.std(autocorr_r2_values),
            'transition_mae_mean': np.mean(transition_mae_values),
            'transition_mae_std': np.std(transition_mae_values),
            'ece_mean': np.mean(ece_values),
            'ece_std': np.std(ece_values),
        }

    return model_stats


def format_latex_table(model_stats):
    """Format the statistics as a LaTeX table."""
    # Model display names
    model_display = {
        'llama-3-8b': r'\texttt{Llama-3.1 (8B)}',
        'llama-3-70b': r'\texttt{Llama-3.1 (70B)}',
        'deepseek-r1-distill-8b': r'\texttt{DeepSeek-Distill (8B)}',
        'deepseek-r1-distill-70b': r'\texttt{DeepSeek-Distill (70B)}',
    }

    lines = []
    lines.append(r"\begin{table*}[!htb]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Training performance of BiGRU classifiers per model, averaged across hardware (A100, H100) and tensor parallelism levels. ")
    lines.append(r"Metrics report mean~$\pm$~standard deviation across configurations. ")
    lines.append(r"Arrows indicate the direction of improvement.}")
    lines.append(r"\label{tab:train_eval_model}")
    lines.append(r"\vspace{0.4em}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & \textbf{Validation F1~$\uparrow$} & \textbf{Autocorr~$R^2~\uparrow$} & \textbf{Transition~MAE~(W)~$\downarrow$} & \textbf{Expected Calibration Error} $\downarrow$ \\")
    lines.append(r"\midrule")

    # Sort models for consistent ordering
    model_order = ['llama-3-8b', 'llama-3-70b', 'deepseek-r1-distill-8b', 'deepseek-r1-distill-70b']

    for model in model_order:
        if model not in model_stats:
            continue
        stats = model_stats[model]

        f1_str = f"{stats['val_f1_mean']:.2f}~$\\pm$~{stats['val_f1_std']:.2f}"
        autocorr_str = f"{stats['autocorr_r2_mean']:.2f}~$\\pm$~{stats['autocorr_r2_std']:.2f}"
        mae_str = f"{stats['transition_mae_mean']:.1f}~$\\pm$~{stats['transition_mae_std']:.1f}"
        ece_str = f"{stats['ece_mean']:.4f}~$\\pm$~{stats['ece_std']:.4f}"

        line = f"{model_display[model]} & {f1_str} & {autocorr_str} & {mae_str} & {ece_str} \\\\"
        lines.append(line)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description="Aggregate training metrics and generate LaTeX table")
    parser.add_argument(
        '--metrics_dir',
        type=str,
        default='./training_results/metrics',
        help='Directory containing metric CSV files'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./training_results/table_summary.tex',
        help='Output LaTeX file'
    )
    parser.add_argument(
        '--csv_output',
        type=str,
        default='./training_results/model_statistics.csv',
        help='Output CSV file with statistics'
    )
    args = parser.parse_args()

    print(f"Aggregating metrics from: {args.metrics_dir}")
    all_metrics = aggregate_metrics(args.metrics_dir)

    if not all_metrics:
        print("Error: No metrics files found!")
        return

    print(f"\nFound metrics for {len(all_metrics)} models")
    for model in all_metrics:
        num_configs = sum(len(all_metrics[model][hw]) for hw in all_metrics[model])
        print(f"  {model}: {num_configs} configurations")

    print("\nComputing statistics...")
    model_stats = compute_model_statistics(all_metrics)

    # Save to CSV
    with open(args.csv_output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'model',
            'val_f1_mean', 'val_f1_std',
            'autocorr_r2_mean', 'autocorr_r2_std',
            'transition_mae_mean', 'transition_mae_std',
            'ece_mean', 'ece_std'
        ])
        for model, stats in sorted(model_stats.items()):
            writer.writerow([
                model,
                stats['val_f1_mean'], stats['val_f1_std'],
                stats['autocorr_r2_mean'], stats['autocorr_r2_std'],
                stats['transition_mae_mean'], stats['transition_mae_std'],
                stats['ece_mean'], stats['ece_std']
            ])
    print(f"\nStatistics saved to: {args.csv_output}")

    # Generate LaTeX table
    latex_table = format_latex_table(model_stats)
    with open(args.output, 'w') as f:
        f.write(latex_table)
    print(f"LaTeX table saved to: {args.output}")

    print("\n" + "="*80)
    print("Model Statistics Summary:")
    print("="*80)
    for model in sorted(model_stats.keys()):
        stats = model_stats[model]
        print(f"\n{model}:")
        print(f"  Validation F1: {stats['val_f1_mean']:.2f} ± {stats['val_f1_std']:.2f}")
        print(f"  Autocorr R²: {stats['autocorr_r2_mean']:.2f} ± {stats['autocorr_r2_std']:.2f}")
        print(f"  Transition MAE: {stats['transition_mae_mean']:.1f} ± {stats['transition_mae_std']:.1f} W")
        print(f"  ECE: {stats['ece_mean']:.4f} ± {stats['ece_std']:.4f}")


if __name__ == '__main__':
    main()
