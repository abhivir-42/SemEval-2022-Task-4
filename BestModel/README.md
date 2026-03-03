# PCL Detection: Training & Inference

## Quick Start

The main training notebook is `train.ipynb`. It's self-contained and handles:
- Data loading (with community-aware formatting)
- Model training (DeBERTa-v3-base + focal loss, 5 epochs)
- Threshold optimization
- Prediction generation (`dev.txt`, `test.txt`)
- Error analysis figures

## GPU Setup

### Google Colab (Recommended)
1. Upload `train.ipynb` to Colab
2. Upload `../data/` to Colab (or mount Google Drive)
3. Runtime → Change runtime type → GPU (T4)
4. Run all cells
5. Download `dev.txt`, `test.txt`, and figures

### Local Mac with MPS
```bash
cd BestModel
python train.ipynb  # Jupyter will run with mps device
```

### Other GPU (Lambda Labs, RunPod, etc.)
- Ensure CUDA is available
- Install dependencies: `pip install transformers torch datasets scikit-learn matplotlib seaborn`
- Run notebook

## Output Files

After training, you'll have:
- `output/best_model/` — saved DeBERTa model + tokenizer
- `output/config.json` — hyperparameters + optimal threshold
- `../dev.txt` — dev set predictions (2,094 lines)
- `../test.txt` — test set predictions (3,832 lines)
- `../report/figures/threshold_optimization.png` — threshold curve
- `../report/figures/confusion_matrix.png` — confusion matrix
- `../report/figures/keyword_performance.png` — per-community F1
- `../report/figures/pr_curve.png` — precision-recall curve

## Configuration

Key hyperparameters (in training cell):
- `NUM_EPOCHS = 5`
- `BATCH_SIZE = 16`
- `LEARNING_RATE = 2e-5`
- `FOCAL_ALPHA = 0.75` (positive class weight)
- `FOCAL_GAMMA = 2.0` (focusing parameter)
- `MAX_LENGTH = 256` (tokens)

## Data

The notebook expects data at:
- `../data/dontpatronizeme_pcl.tsv` (full dataset)
- `../data/practice-splits/train_semeval_parids-labels.csv`
- `../data/practice-splits/dev_semeval_parids-labels.csv`
- `../data/test/task4_test.tsv` (test set)

All paths are relative to `BestModel/` directory.

## Novel Approach

Three components beyond the RoBERTa-base baseline:

1. **Community-Aware Input**: Prepend `[Community: keyword]` to each paragraph
   - Reason: PCL rates vary 6× across communities (2.8% to 16.5%)

2. **Focal Loss**: Dynamic down-weighting of easy examples
   - Reason: 9.5:1 class imbalance; preserves all data vs. baseline's downsampling

3. **Threshold Optimization**: Search for threshold that maximizes F1
   - Reason: Optimal threshold ~0.3-0.4, not default 0.5

## Expected Results

- **Baseline (RoBERTa-base)**: F1 = 0.48 (dev), 0.49 (test)
- **Our model**: Target F1 > 0.54 (top 30% on leaderboard)
