"""
PCL Detection: Context-Enriched DeBERTa-v3 with Focal Loss
Standalone training script for SLURM batch execution.

Usage:
    python src/train.py --base_dir /vol/bitbucket/as9422/SemEval-2022-Task-4
"""

import argparse
import json
import os
import random

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for cluster
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


# ============================================================
# Configuration
# ============================================================
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 256
COMMUNITY_AWARE = True

NUM_EPOCHS = 5
BATCH_SIZE = 16
GRADIENT_ACCUMULATION = 2
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
EARLY_STOPPING_PATIENCE = 2

FOCAL_ALPHA = 0.75
FOCAL_GAMMA = 2.0
SEED = 42


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

    # Full dataset
    full_df = pd.read_csv(
        tsv_path, sep="\t", skiprows=4, header=None,
        names=["par_id", "art_id", "keyword", "country", "text", "label"],
    )
    full_df["par_id"] = full_df["par_id"].astype(str)
    full_df["binary_label"] = (full_df["label"] >= 2).astype(int)

    # Splits
    train_ids = pd.read_csv(train_split_path)
    dev_ids = pd.read_csv(dev_split_path)
    train_ids["par_id"] = train_ids["par_id"].astype(str)
    dev_ids["par_id"] = dev_ids["par_id"].astype(str)

    train_df = full_df[full_df["par_id"].isin(train_ids["par_id"])].copy()
    dev_df = dev_ids[["par_id"]].merge(full_df, on="par_id", how="left").copy()

    # Test set
    test_df = pd.read_csv(
        test_path, sep="\t", header=None,
        names=["par_id", "art_id", "keyword", "country", "text"],
    )
    test_df["par_id"] = test_df["par_id"].astype(str)

    # Community-aware formatting
    def fmt(row):
        if COMMUNITY_AWARE:
            return f"[Community: {row['keyword']}] {row['text']}"
        return row["text"]

    train_df["input_text"] = train_df.apply(fmt, axis=1)
    dev_df["input_text"] = dev_df.apply(fmt, axis=1)
    test_df["input_text"] = test_df.apply(fmt, axis=1)

    print(f"Train: {len(train_df)} ({train_df['binary_label'].sum()} pos, {train_df['binary_label'].mean():.1%})")
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
# Focal Loss
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction="none")
        p_t = torch.exp(-ce_loss)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_weight = (1 - p_t) ** self.gamma
        return (alpha_t * focal_weight * ce_loss).mean()


class FocalLossTrainer(Trainer):
    def __init__(self, focal_alpha=0.75, focal_gamma=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = self.focal_loss(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


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
    f1_scores = [f1_score(labels, (probs >= t).astype(int), pos_label=1, zero_division=0) for t in thresholds]
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx], thresholds, f1_scores


# ============================================================
# Plotting
# ============================================================
def save_plots(dev_df, dev_preds, dev_probs, dev_labels, best_threshold, thresholds, f1_scores, fig_dir):
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
def main(base_dir):
    set_seed(SEED)
    output_dir = os.path.join(base_dir, "BestModel", "output")
    fig_dir = os.path.join(base_dir, "report", "figures")
    os.makedirs(output_dir, exist_ok=True)

    device_name = "CPU"
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        device_name = "Apple MPS"
    print(f"Device: {device_name}")

    # Load data
    train_df, dev_df, test_df = load_data(base_dir)

    # Internal train/val split
    train_split, val_split = train_test_split(
        train_df, test_size=0.15, random_state=SEED, stratify=train_df["binary_label"],
    )
    train_split = train_split.reset_index(drop=True)
    val_split = val_split.reset_index(drop=True)
    print(f"Internal train: {len(train_split)} | val: {len(val_split)}")

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = PCLDataset(train_split["input_text"].tolist(), train_split["binary_label"].tolist(), tokenizer, MAX_LENGTH)
    val_dataset = PCLDataset(val_split["input_text"].tolist(), val_split["binary_label"].tolist(), tokenizer, MAX_LENGTH)
    dev_dataset = PCLDataset(dev_df["input_text"].tolist(), dev_df["binary_label"].tolist(), tokenizer, MAX_LENGTH)
    test_dataset = PCLDataset(test_df["input_text"].tolist(), labels=None, tokenizer=tokenizer, max_length=MAX_LENGTH)

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=50,
        seed=SEED,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = FocalLossTrainer(
        focal_alpha=FOCAL_ALPHA,
        focal_gamma=FOCAL_GAMMA,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
    )

    # Train
    print(f"\nTraining {MODEL_NAME} | focal_alpha={FOCAL_ALPHA} gamma={FOCAL_GAMMA}")
    result = trainer.train()
    print(f"Training loss: {result.training_loss:.4f} | Time: {result.metrics['train_runtime']:.0f}s")

    # Validation
    val_metrics = trainer.evaluate(val_dataset)
    print(f"Val F1: {val_metrics['eval_f1']:.4f} | P: {val_metrics['eval_precision']:.4f} | R: {val_metrics['eval_recall']:.4f}")

    # Threshold optimization on dev set
    dev_labels = dev_df["binary_label"].values
    dev_outputs = trainer.predict(dev_dataset)
    dev_probs = torch.softmax(torch.tensor(dev_outputs.predictions), dim=-1)[:, 1].numpy()

    best_threshold, best_f1, thresholds, f1_scores = find_optimal_threshold(dev_probs, dev_labels)
    default_f1 = f1_score(dev_labels, (dev_probs >= 0.5).astype(int), pos_label=1)

    print(f"\nThreshold optimization:")
    print(f"  Optimal threshold: {best_threshold:.2f} -> F1={best_f1:.4f}")
    print(f"  Default (0.50)              -> F1={default_f1:.4f}")
    print(f"  Baseline                    -> F1=0.4800")

    # Generate predictions
    dev_preds = (dev_probs >= best_threshold).astype(int)

    print(f"\nDev set results:")
    print(classification_report(dev_labels, dev_preds, target_names=["No PCL", "PCL"]))

    # Save dev.txt
    dev_path = os.path.join(base_dir, "dev.txt")
    with open(dev_path, "w") as f:
        for p in dev_preds:
            f.write(f"{p}\n")
    print(f"Saved {dev_path} ({len(dev_preds)} predictions, {sum(dev_preds)} positive)")

    # Save test.txt
    test_outputs = trainer.predict(test_dataset)
    test_probs = torch.softmax(torch.tensor(test_outputs.predictions), dim=-1)[:, 1].numpy()
    test_preds = (test_probs >= best_threshold).astype(int)

    test_path = os.path.join(base_dir, "test.txt")
    with open(test_path, "w") as f:
        for p in test_preds:
            f.write(f"{p}\n")
    print(f"Saved {test_path} ({len(test_preds)} predictions, {sum(test_preds)} positive)")

    assert len(dev_preds) == 2094, f"Expected 2094, got {len(dev_preds)}"
    assert len(test_preds) == 3832, f"Expected 3832, got {len(test_preds)}"

    # Save model + config
    model_path = os.path.join(output_dir, "best_model")
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)

    config = {
        "model_name": MODEL_NAME, "max_length": MAX_LENGTH,
        "community_aware": COMMUNITY_AWARE,
        "optimal_threshold": float(best_threshold),
        "dev_f1": float(best_f1), "default_f1": float(default_f1),
        "focal_alpha": FOCAL_ALPHA, "focal_gamma": FOCAL_GAMMA,
        "seed": SEED,
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nModel saved to {model_path}")

    # Generate figures
    kw_df = save_plots(dev_df, dev_preds, dev_probs, dev_labels, best_threshold, thresholds, f1_scores, fig_dir)
    print("\nPer-keyword performance:")
    print(kw_df.to_string(index=False, float_format="{:.3f}".format))

    # Error analysis summary
    analysis = dev_df.copy()
    analysis["pred"] = dev_preds
    analysis["prob"] = dev_probs
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
    print(f"Dev F1: {best_f1:.4f} (threshold={best_threshold:.2f})")
    print(f"Files: {dev_path}, {test_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True, help="Project root directory")
    args = parser.parse_args()
    main(args.base_dir)
