# Training Pipeline for Power Trace Classifier Models

This document describes the automated training pipeline for generating metrics for the paper table.

## Overview

The pipeline consists of three main scripts:

1. **run_all_training.py** - Runs all training configurations
2. **aggregate_metrics.py** - Aggregates results and generates LaTeX table
3. **model/train_entry.py** - Individual training job entry point (called by run_all_training.py)

## Training Configurations

The pipeline trains BiGRU classifiers for the following configurations:

| Model | Hardware | TP Values |
|-------|----------|-----------|
| llama-3-8b | A100, H100 | 1, 2 |
| llama-3-70b | A100, H100 | 4, 8 |
| deepseek-r1-distill-8b | A100, H100 | 1, 2 |
| deepseek-r1-distill-70b | A100, H100 | 4, 8 |

**Total: 16 training jobs** (4 models × 2 hardware × 2 TP values)

## Metrics Computed

For each configuration, the following metrics are computed:

1. **Validation F1 Score** - Weighted F1 score for state classification
2. **Autocorrelation R²** - Correlation between real and synthetic power trace autocorrelation functions
3. **Transition MAE (W)** - Mean absolute error of power differences at state transitions
4. **Expected Calibration Error (ECE)** - Measures calibration of predicted probabilities

## Usage

### Step 1: Run All Training Jobs

```bash
# Run all training jobs with default settings
python3 run_all_training.py

# Customize training parameters
python3 run_all_training.py \
    --lr 1e-4 \
    --num_epochs 500 \
    --data_dir ./model/training_data

# Train specific models only
python3 run_all_training.py \
    --models llama-3-8b llama-3-70b

# Train on specific hardware only
python3 run_all_training.py \
    --hardware h100

# Dry run to see commands without executing
python3 run_all_training.py --dry_run
```

### Step 2: Aggregate Results

After training completes, aggregate the metrics:

```bash
python3 aggregate_metrics.py
```

This will generate:
- `training_results/model_statistics.csv` - Statistics per model
- `training_results/table_summary.tex` - LaTeX table ready for paper

### Step 3: Review Results

Check the generated LaTeX table:

```bash
cat training_results/table_summary.tex
```

Individual configuration metrics are saved in:
- `training_results/metrics/{model}_{hardware}_tp{X}_metrics.csv`
- `training_results/metrics/{model}_{hardware}_tp{X}_losses.csv`

## Wandb Integration

Each training job automatically logs to Wandb with a job name format:
```
{model}_{hardware}_tp{tp}
```

Example: `llama-3-8b_h100_tp2`

Final metrics are logged under the `final_metrics/` prefix:
- `final_metrics/val_f1`
- `final_metrics/autocorr_r2`
- `final_metrics/transition_mae`
- `final_metrics/ece`

## Output Files

### Training Results Directory Structure

```
training_results/
├── jobs_summary.json              # Summary of all training jobs
├── model_statistics.csv           # Aggregated statistics per model
├── table_summary.tex              # LaTeX table for paper
└── metrics/
    ├── llama-3-8b_a100_tp1_metrics.csv
    ├── llama-3-8b_a100_tp1_losses.csv
    ├── llama-3-8b_a100_tp2_metrics.csv
    ├── llama-3-8b_a100_tp2_losses.csv
    └── ... (16 configurations × 2 files each)
```

### Metrics CSV Format

Each `*_metrics.csv` file contains:
```csv
metric,value
val_f1,0.4321
val_precision,0.5432
val_recall,0.3876
ece,0.0686
autocorr_r2,0.9900
transition_mae,48.5
final_train_loss,0.1234
final_val_loss,0.2345
val_accuracy,0.8765
```

### Losses CSV Format

Each `*_losses.csv` file contains:
```csv
epoch,train_loss
1,1.2345
2,0.9876
...
500,0.1234
```

## Table Output Format

The generated LaTeX table follows this format:

```latex
\begin{table*}[!htb]
\centering
\small
\caption{Training performance of BiGRU classifiers per model...}
\label{tab:train_eval_model}
\vspace{0.4em}
\begin{tabular}{lcccc}
\toprule
\textbf{Model} & \textbf{Validation F1~$\uparrow$} & ... \\
\midrule
\texttt{Llama-3.1 (8B)}  & 0.43~$\pm$~0.06 & ... \\
...
\bottomrule
\end{tabular}
\end{table*}
```

## Implementation Details

### Metrics Computation

1. **Validation F1**: Computed using sklearn's weighted F1 score on state predictions
2. **Autocorrelation R²**:
   - Compute ACF for real power trace (50 lags)
   - Compute ACF for synthetic power trace
   - Calculate Pearson correlation coefficient squared
3. **Transition MAE**:
   - Identify state transitions (where state label changes)
   - Compute power differences at transitions
   - Take mean absolute value
4. **ECE**:
   - Bin predicted probabilities into 10 bins
   - For each bin, compare average confidence to average accuracy
   - Weighted sum of absolute differences

### Training Parameters

- **Learning rate**: 1e-4 (default)
- **Epochs**: 500 (default)
- **Hidden size**: 64
- **Validation split**: 20%
- **Batch size**: 1 (due to variable sequence lengths)
- **Optimizer**: Adam
- **Loss**: CrossEntropyLoss
- **Gradient clipping**: max_norm=1.0

## Troubleshooting

### Missing Data Files

If you see warnings about missing data files:
```
WARNING: Data file not found: ./model/training_data/vllm-benchmark_xxx.npz
```

Ensure all required data files exist in `model/training_data/`:
- `vllm-benchmark_llama-3-8b_{a100,h100}.npz`
- `vllm-benchmark_llama-3-70b_{a100,h100}.npz`
- `vllm-benchmark_deepseek-r1-8b_{a100,h100}.npz`
- `vllm-benchmark_deepseek-r1-70b_{a100,h100}.npz`

### No Metrics Files Found

If `aggregate_metrics.py` reports no metrics files:
1. Check that training completed successfully
2. Verify metrics files exist in `training_results/metrics/`
3. Check file naming matches expected pattern

### Wandb Connection Issues

To disable Wandb logging, modify `model/classifiers/train.py`:
```python
use_wandb: bool = False  # Change default to False
```

## Manual Training

To run a single training job manually:

```bash
python3 -m model.train_entry \
    --data_file ./model/training_data/vllm-benchmark_llama-3-70b_a100.npz \
    --model llama-3-70b \
    --tp 4 \
    --hardware_accelerator h100 \
    --lr 1e-4 \
    --num_epochs 500
```

## Next Steps

After generating the table:

1. Copy the LaTeX table into your paper
2. Verify the metrics match expected ranges
3. Compare results across models and configurations
4. Plot training curves from `*_losses.csv` files if needed
