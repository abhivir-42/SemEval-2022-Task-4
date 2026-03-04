"""
PCL Detection: Enhanced DeBERTa-v3 with Focal Loss + Multi-Seed Ensemble

Improvements over v1:
  - DeBERTa-v3-large backbone (from base)
  - Multi-seed ensemble (average probabilities across N runs)
  - Layerwise learning rate decay (lower layers learn slower)
  - Label smoothing in focal loss (better calibration)
  - Gradient checkpointing for memory efficiency

Usage:
    # Full ensemble (3 seeds, ~90 min on T4):
    python src/train.py --base_dir .

    # Quick single-seed run (~30 min on T4):
    python src/train.py --base_dir . --seeds 42

    # Use base model if GPU memory is limited:
    python src/train.py --base_dir . --model_name microsoft/deberta-v3-base
"""

import argparse
import gc
import json
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train PCL detection model")
    parser.add_argument("--base_dir", type=str, required=True, help="Project root directory")
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-large")
    parser.add_argument("--seeds", type=str, default="42,123,456",
                        help="Comma-separated seeds for ensemble")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Auto-set based on model if not specified")
    parser.add_argument("--gradient_accumulation", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--focal_alpha", type=float, default=0.75)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--layerwise_lr_decay", type=float, default=0.95)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--early_stopping_patience", type=int, default=2)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Data Loading
# ============================================================
def load_data(base_dir):
    data_dir = os.path.join(base_dir, "data")
    tsv_path = os.path.join(data_dir, "dontpatronizeme_pcl.tsv")
    train_split_path = os.path.join(data_dir, "practice-splits", "train_semeval_parids-labels.csv")
    dev_split_path = os.path.join(data_dir, "practice-splits", "dev_semeval_parids-labels.csv")
    test_path = os.path.join(data_dir, "test", "task4_test.tsv")

    full_df = pd.read_csv(
        tsv_path, sep="\t", skiprows=4, header=None,
        names=["par_id", "art_id", "keyword", "country", "text", "label"],
    )
    full_df["par_id"] = full_df["par_id"].astype(str)
    full_df["binary_label"] = (full_df["label"] >= 2).astype(int)

    train_ids = pd.read_csv(train_split_path)
    dev_ids = pd.read_csv(dev_split_path)
    train_ids["par_id"] = train_ids["par_id"].astype(str)
    dev_ids["par_id"] = dev_ids["par_id"].astype(str)

    train_df = full_df[full_df["par_id"].isin(train_ids["par_id"])].copy()
    dev_df = dev_ids[["par_id"]].merge(full_df, on="par_id", how="left").copy()

    test_df = pd.read_csv(
        test_path, sep="\t", header=None,
        names=["par_id", "art_id", "keyword", "country", "text"],
    )
    test_df["par_id"] = test_df["par_id"].astype(str)

    # Community-aware input formatting
    def fmt(row):
        return f"[Community: {row['keyword']}] {row['text']}"

    train_df["input_text"] = train_df.apply(fmt, axis=1)
    dev_df["input_text"] = dev_df.apply(fmt, axis=1)
    test_df["input_text"] = test_df.apply(fmt, axis=1)

    print(f"Train: {len(train_df)} ({train_df['binary_label'].sum()} pos, "
          f"{train_df['binary_label'].mean():.1%})")
    print(f"Dev:   {len(dev_df)} ({dev_df['binary_label'].sum()} pos)")
    print(f"Test:  {len(test_df)} (labels hidden)")

    return train_df, dev_df, test_df


# ============================================================
# Dataset
# ============================================================
class PCLDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=256):
        self.encodings = tokenizer(
            texts, truncation=True, padding="max_length",
            max_length=max_length, return_tensors="pt",
        )
        self.labels = labels

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ============================================================
# Focal Loss with Label Smoothing
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(
            logits, targets, reduction="none",
            label_smoothing=self.label_smoothing,
        )
        p_t = torch.exp(-ce_loss)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_weight = (1 - p_t) ** self.gamma
        return (alpha_t * focal_weight * ce_loss).mean()


# ============================================================
# Trainer with Focal Loss + Layerwise LR Decay
# ============================================================
class ImprovedTrainer(Trainer):
    def __init__(self, focal_alpha=0.75, focal_gamma=2.0, label_smoothing=0.0,
                 layerwise_lr_decay=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss = FocalLoss(
            alpha=focal_alpha, gamma=focal_gamma, label_smoothing=label_smoothing
        )
        self.layerwise_lr_decay = layerwise_lr_decay

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = self.focal_loss(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        if self.layerwise_lr_decay is not None and self.layerwise_lr_decay < 1.0:
            return self._create_layerwise_optimizer()
        return super().create_optimizer()

    def _create_layerwise_optimizer(self):
        model = self.model
        decay = self.layerwise_lr_decay
        base_lr = self.args.learning_rate
        no_decay = ["bias", "LayerNorm.weight", "layernorm.weight"]
        num_layers = model.config.num_hidden_layers

        optimizer_grouped_parameters = []

        # Track which parameters have been assigned
        assigned_params = set()

        # Embedding parameters (lowest LR)
        embed_lr = base_lr * (decay ** num_layers)
        embed_decay, embed_no_decay = [], []
        for n, p in model.named_parameters():
            if "embeddings" in n:
                assigned_params.add(n)
                if any(nd in n for nd in no_decay):
                    embed_no_decay.append(p)
                else:
                    embed_decay.append(p)
        if embed_decay:
            optimizer_grouped_parameters.append({
                "params": embed_decay, "weight_decay": self.args.weight_decay, "lr": embed_lr,
            })
        if embed_no_decay:
            optimizer_grouped_parameters.append({
                "params": embed_no_decay, "weight_decay": 0.0, "lr": embed_lr,
            })

        # Encoder layers (increasing LR from bottom to top)
        for i in range(num_layers):
            layer_lr = base_lr * (decay ** (num_layers - i - 1))
            layer_decay, layer_no_decay = [], []
            for n, p in model.named_parameters():
                if f"encoder.layer.{i}." in n:
                    assigned_params.add(n)
                    if any(nd in n for nd in no_decay):
                        layer_no_decay.append(p)
                    else:
                        layer_decay.append(p)
            if layer_decay:
                optimizer_grouped_parameters.append({
                    "params": layer_decay, "weight_decay": self.args.weight_decay, "lr": layer_lr,
                })
            if layer_no_decay:
                optimizer_grouped_parameters.append({
                    "params": layer_no_decay, "weight_decay": 0.0, "lr": layer_lr,
                })

        # Remaining parameters: classifier head, pooler, rel_embeddings (highest LR)
        head_decay, head_no_decay = [], []
        for n, p in model.named_parameters():
            if n not in assigned_params:
                if any(nd in n for nd in no_decay):
                    head_no_decay.append(p)
                else:
                    head_decay.append(p)
        if head_decay:
            optimizer_grouped_parameters.append({
                "params": head_decay, "weight_decay": self.args.weight_decay, "lr": base_lr,
            })
        if head_no_decay:
            optimizer_grouped_parameters.append({
                "params": head_no_decay, "weight_decay": 0.0, "lr": base_lr,
            })

        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

        print(f"  Layerwise LR decay={decay}: embed={embed_lr:.2e}, "
              f"layer_0={base_lr * decay ** (num_layers - 1):.2e}, "
              f"layer_{num_layers-1}={base_lr:.2e}, head={base_lr:.2e}")

        return self.optimizer


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "f1": f1_score(labels, preds, pos_label=1),
        "precision": precision_score(labels, preds, pos_label=1, zero_division=0),
        "recall": recall_score(labels, preds, pos_label=1, zero_division=0),
    }


# ============================================================
# Threshold Optimization
# ============================================================
def find_optimal_threshold(probs, labels):
    thresholds = np.arange(0.10, 0.90, 0.01)
    f1_scores = [
        f1_score(labels, (probs >= t).astype(int), pos_label=1, zero_division=0)
        for t in thresholds
    ]
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx], thresholds, f1_scores


# ============================================================
# Single-Seed Training
# ============================================================
def train_single_seed(args, seed, train_df, tokenizer, dev_dataset, test_dataset,
                      batch_size, grad_accum, lr):
    """Train one model with a given seed, return dev/test probabilities."""
    set_seed(seed)
    print(f"\n{'='*60}")
    print(f"SEED {seed}")
    print(f"{'='*60}")

    # Internal train/val split (different per seed for diversity)
    train_split, val_split = train_test_split(
        train_df, test_size=0.15, random_state=seed, stratify=train_df["binary_label"],
    )
    train_split = train_split.reset_index(drop=True)
    val_split = val_split.reset_index(drop=True)
    print(f"  Internal train: {len(train_split)} | val: {len(val_split)}")

    train_dataset = PCLDataset(
        train_split["input_text"].tolist(),
        train_split["binary_label"].tolist(),
        tokenizer, args.max_length,
    )
    val_dataset = PCLDataset(
        val_split["input_text"].tolist(),
        val_split["binary_label"].tolist(),
        tokenizer, args.max_length,
    )

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    # Gradient checkpointing for large models
    if "large" in args.model_name.lower():
        model.gradient_checkpointing_enable()
        print("  Gradient checkpointing: ON")

    seed_output_dir = os.path.join(args.base_dir, "BestModel", "output", f"seed_{seed}")
    os.makedirs(seed_output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=seed_output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=50,
        seed=seed,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = ImprovedTrainer(
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        layerwise_lr_decay=args.layerwise_lr_decay,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    print(f"  Training: batch={batch_size}, grad_accum={grad_accum}, lr={lr}")
    result = trainer.train()
    print(f"  Loss: {result.training_loss:.4f} | Time: {result.metrics['train_runtime']:.0f}s")

    val_metrics = trainer.evaluate(val_dataset)
    print(f"  Val F1: {val_metrics['eval_f1']:.4f}")

    # Get probabilities
    dev_outputs = trainer.predict(dev_dataset)
    dev_probs = torch.softmax(torch.tensor(dev_outputs.predictions), dim=-1)[:, 1].numpy()

    test_outputs = trainer.predict(test_dataset)
    test_probs = torch.softmax(torch.tensor(test_outputs.predictions), dim=-1)[:, 1].numpy()

    # Clean up GPU memory
    del model, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return dev_probs, test_probs


# ============================================================
# Plotting
# ============================================================
def save_plots(dev_df, dev_preds, dev_probs, dev_labels, best_threshold, thresholds,
               f1_scores, fig_dir):
    os.makedirs(fig_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.15)

    # 1. Threshold optimization curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, f1_scores, "b-", linewidth=2)
    ax.axvline(x=best_threshold, color="r", linestyle="--", label=f"Optimal: {best_threshold:.2f}")
    ax.axvline(x=0.5, color="gray", linestyle=":", label="Default: 0.50")
    ax.axhline(y=0.48, color="orange", linestyle=":", alpha=0.7, label="Baseline: 0.48")
    ax.set_xlabel("Classification Threshold")
    ax.set_ylabel("F1 Score (Positive Class)")
    ax.set_title("Threshold Optimization for PCL Detection")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "threshold_optimization.png"), dpi=150)
    plt.close()

    # 2. Confusion matrix
    cm = confusion_matrix(dev_labels, dev_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred: No PCL", "Pred: PCL"],
                yticklabels=["True: No PCL", "True: PCL"], ax=ax)
    ax.set_title("Confusion Matrix (Official Dev Set)")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "confusion_matrix.png"), dpi=150)
    plt.close()

    # 3. Per-keyword performance
    analysis = dev_df.copy()
    analysis["pred"] = dev_preds
    analysis["prob"] = dev_probs

    kw_rows = []
    for kw in sorted(analysis["keyword"].unique()):
        mask = analysis["keyword"] == kw
        kl = analysis.loc[mask, "binary_label"].values
        kp = analysis.loc[mask, "pred"].values
        n_pos = kl.sum()
        kw_rows.append({
            "keyword": kw, "n_total": mask.sum(), "n_positive": int(n_pos),
            "f1": f1_score(kl, kp, pos_label=1, zero_division=0) if n_pos > 0 else 0,
            "precision": precision_score(kl, kp, pos_label=1, zero_division=0) if n_pos > 0 else 0,
            "recall": recall_score(kl, kp, pos_label=1, zero_division=0) if n_pos > 0 else 0,
        })
    kw_df = pd.DataFrame(kw_rows)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(kw_df))
    w = 0.25
    ax.bar([i - w for i in x], kw_df["f1"], w, label="F1", color="steelblue")
    ax.bar(x, kw_df["precision"], w, label="Precision", color="coral")
    ax.bar([i + w for i in x], kw_df["recall"], w, label="Recall", color="seagreen")
    ax.set_xticks(list(x))
    ax.set_xticklabels(kw_df["keyword"], rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Performance by Community Keyword")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "keyword_performance.png"), dpi=150)
    plt.close()

    # 4. Precision-Recall curve
    prec_curve, rec_curve, _ = precision_recall_curve(dev_labels, dev_probs)
    ap = average_precision_score(dev_labels, dev_probs)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rec_curve, prec_curve, "b-", linewidth=2, label=f"DeBERTa-v3 (AP={ap:.3f})")
    ax.scatter(
        [recall_score(dev_labels, dev_preds, pos_label=1)],
        [precision_score(dev_labels, dev_preds, pos_label=1)],
        color="red", s=100, zorder=5, label=f"Operating point (t={best_threshold:.2f})",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "pr_curve.png"), dpi=150)
    plt.close()

    print(f"Saved 4 figures to {fig_dir}")
    return kw_df


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    output_dir = os.path.join(args.base_dir, "BestModel", "output")
    fig_dir = os.path.join(args.base_dir, "report", "figures")
    os.makedirs(output_dir, exist_ok=True)

    # Auto-detect device
    device_name = "CPU"
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        device_name = "Apple MPS"
    print(f"Device: {device_name}")
    print(f"Model: {args.model_name}")
    print(f"Seeds: {seeds} ({'ensemble' if len(seeds) > 1 else 'single'})")

    # Auto-set hyperparameters based on model size
    is_large = "large" in args.model_name.lower()
    batch_size = args.batch_size or (4 if is_large else 16)
    grad_accum = args.gradient_accumulation or (8 if is_large else 2)
    lr = args.learning_rate or (1e-5 if is_large else 2e-5)
    print(f"Config: batch={batch_size}, grad_accum={grad_accum}, lr={lr}, "
          f"label_smoothing={args.label_smoothing}, llrd={args.layerwise_lr_decay}")

    # Load data
    train_df, dev_df, test_df = load_data(args.base_dir)
    dev_labels = dev_df["binary_label"].values

    # Tokenize once (shared across seeds)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dev_dataset = PCLDataset(
        dev_df["input_text"].tolist(), dev_df["binary_label"].tolist(),
        tokenizer, args.max_length,
    )
    test_dataset = PCLDataset(
        test_df["input_text"].tolist(), labels=None,
        tokenizer=tokenizer, max_length=args.max_length,
    )

    # Train with each seed
    all_dev_probs = []
    all_test_probs = []
    seed_f1s = []

    for seed in seeds:
        dev_probs, test_probs = train_single_seed(
            args, seed, train_df, tokenizer, dev_dataset, test_dataset,
            batch_size, grad_accum, lr,
        )
        # Quick single-seed eval
        _, single_f1, _, _ = find_optimal_threshold(dev_probs, dev_labels)
        print(f"  Seed {seed} dev F1: {single_f1:.4f}")

        all_dev_probs.append(dev_probs)
        all_test_probs.append(test_probs)
        seed_f1s.append(single_f1)

        # Save individual probabilities for reproducibility
        np.save(os.path.join(output_dir, f"dev_probs_seed_{seed}.npy"), dev_probs)
        np.save(os.path.join(output_dir, f"test_probs_seed_{seed}.npy"), test_probs)

    # Ensemble: average probabilities
    if len(seeds) > 1:
        print(f"\n{'='*60}")
        print(f"ENSEMBLE ({len(seeds)} seeds)")
        print(f"{'='*60}")
        print(f"Individual F1s: {[f'{f:.4f}' for f in seed_f1s]}")

    ensemble_dev_probs = np.mean(all_dev_probs, axis=0)
    ensemble_test_probs = np.mean(all_test_probs, axis=0)

    # Threshold optimization on (averaged) probabilities
    best_threshold, best_f1, thresholds, f1_scores = find_optimal_threshold(
        ensemble_dev_probs, dev_labels
    )
    default_f1 = f1_score(dev_labels, (ensemble_dev_probs >= 0.5).astype(int), pos_label=1)

    print(f"\nThreshold optimization:")
    print(f"  Optimal threshold: {best_threshold:.2f} -> F1={best_f1:.4f}")
    print(f"  Default (0.50)              -> F1={default_f1:.4f}")
    print(f"  Previous best               -> F1=0.5895")
    print(f"  Baseline                    -> F1=0.4800")

    # Generate final predictions
    dev_preds = (ensemble_dev_probs >= best_threshold).astype(int)
    print(f"\nDev set results:")
    print(classification_report(dev_labels, dev_preds, target_names=["No PCL", "PCL"]))

    # Save dev.txt
    dev_path = os.path.join(args.base_dir, "dev.txt")
    with open(dev_path, "w") as f:
        for p in dev_preds:
            f.write(f"{p}\n")
    print(f"Saved {dev_path} ({len(dev_preds)} predictions, {sum(dev_preds)} positive)")

    # Save test.txt
    test_preds = (ensemble_test_probs >= best_threshold).astype(int)
    test_path = os.path.join(args.base_dir, "test.txt")
    with open(test_path, "w") as f:
        for p in test_preds:
            f.write(f"{p}\n")
    print(f"Saved {test_path} ({len(test_preds)} predictions, {sum(test_preds)} positive)")

    assert len(dev_preds) == 2094, f"Expected 2094, got {len(dev_preds)}"
    assert len(test_preds) == 3832, f"Expected 3832, got {len(test_preds)}"

    # Save config
    config = {
        "model_name": args.model_name,
        "max_length": args.max_length,
        "community_aware": True,
        "ensemble_seeds": seeds,
        "optimal_threshold": float(best_threshold),
        "dev_f1": float(best_f1),
        "default_f1": float(default_f1),
        "individual_seed_f1s": {str(s): float(f) for s, f in zip(seeds, seed_f1s)},
        "focal_alpha": args.focal_alpha,
        "focal_gamma": args.focal_gamma,
        "label_smoothing": args.label_smoothing,
        "layerwise_lr_decay": args.layerwise_lr_decay,
        "batch_size": batch_size,
        "gradient_accumulation": grad_accum,
        "learning_rate": lr,
        "best_seed": int(seeds[np.argmax(seed_f1s)]),
    }
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nConfig saved to {config_path}")

    # Generate figures
    kw_df = save_plots(
        dev_df, dev_preds, ensemble_dev_probs, dev_labels,
        best_threshold, thresholds, f1_scores, fig_dir,
    )
    print("\nPer-keyword performance:")
    print(kw_df.to_string(index=False, float_format="{:.3f}".format))

    # Error analysis
    analysis = dev_df.copy()
    analysis["pred"] = dev_preds
    analysis["prob"] = ensemble_dev_probs
    fp = analysis[(analysis["binary_label"] == 0) & (analysis["pred"] == 1)]
    fn = analysis[(analysis["binary_label"] == 1) & (analysis["pred"] == 0)]
    print(f"\nError breakdown: {len(fp)} false positives, {len(fn)} false negatives")

    print(f"\nTop 5 false positives (highest confidence):")
    for _, r in fp.nlargest(5, "prob").iterrows():
        print(f"  [{r['keyword']}] p={r['prob']:.3f} orig_label={r['label']} | {r['text'][:120]}...")

    print(f"\nTop 5 false negatives (most confident wrong):")
    for _, r in fn.nsmallest(5, "prob").iterrows():
        print(f"  [{r['keyword']}] p={r['prob']:.3f} orig_label={r['label']} | {r['text'][:120]}...")

    print("\n=== DONE ===")
    print(f"{'Ensemble' if len(seeds) > 1 else 'Single'} Dev F1: {best_f1:.4f} "
          f"(threshold={best_threshold:.2f})")
    print(f"Files: {dev_path}, {test_path}")


if __name__ == "__main__":
    main()
