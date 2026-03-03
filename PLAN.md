# Comprehensive Plan: SemEval 2022 Task 4 - PCL Detection

## Table of Contents
1. [Task Overview](#1-task-overview)
2. [Current State Analysis](#2-current-state-analysis)
3. [Strategic Approach](#3-strategic-approach)
4. [Execution Plan with Sub-Agents](#4-execution-plan-with-sub-agents)
5. [Technical Deep-Dive](#5-technical-deep-dive)
6. [Repository Structure](#6-repository-structure)
7. [Risk Mitigation](#7-risk-mitigation)
8. [Mark Allocation Strategy](#8-mark-allocation-strategy)

---

## 1. Task Overview

**Goal**: Binary classification of paragraphs as containing Patronizing and Condescending Language (PCL=1) or not (PCL=0), targeting vulnerable communities in news media.

**Key Constraints**:
- Deadline: Wednesday, March 4th, 7pm
- Metric: F1 score of the **positive class** (PCL=1)
- Baseline to beat: RoBERTa-base → F1=0.48 (dev), F1=0.49 (test)
- Must produce `dev.txt` and `test.txt` prediction files
- Need a "novel" approach (not just swapping pre-trained models)
- Report in LaTeX, code in `src/`, best model in `BestModel/`

**Deliverables**:
| Deliverable | Location | Exercise |
|---|---|---|
| Paper review | Report Chapter 1 | Ex 1 (6 marks) |
| EDA with 2 techniques | Report Chapter 2 + `src/` | Ex 2 (6 marks) |
| Approach description | Report Chapter 3 | Ex 3 (4 marks) |
| Training code + model | `src/` + `BestModel/` | Ex 4 (1 mark) |
| dev.txt + test.txt | Root directory | Ex 5.1 (6 marks) |
| Error analysis | Report Chapter 4 | Ex 5.2 (5 marks) |
| Report quality | `report/` | Ex 6 (2 marks) |

**Total: 30 marks**

---

## 2. Current State Analysis

### 2.1 Data Landscape

| Dataset | Records | Notes |
|---|---|---|
| Full dataset (`dontpatronizeme_pcl.tsv`) | 10,467 paragraphs | Tab-separated, 5-point annotation scale |
| Train split | 8,375 paragraph IDs | From `train_semeval_parids-labels.csv` |
| Dev split (official) | 2,094 paragraph IDs | From `dev_semeval_parids-labels.csv` — labels available |
| Test set (official) | 3,832 paragraphs | From `task4_test.tsv` — **labels hidden** |

**Binary labels**: The original 5-point scale (0-4) is collapsed: labels 0-1 → negative (No PCL), labels 2-4 → positive (PCL).

**Critical: Class imbalance**. From the paper, only ~995/10,637 paragraphs are positive (~9.4%). In the train split, this means roughly ~780 positive and ~7,595 negative examples. This 1:10 ratio is the single most important challenge.

### 2.2 Baseline Analysis

The provided baseline (`baseline/Reconstruct_and_RoBERTa_baseline_train_dev_dataset.ipynb`) does:
1. Uses `simpletransformers` library with `roberta-base`
2. Downsamples negatives to 2:1 ratio (keeps ~1,560 negatives + ~780 positives ≈ 2,382 total)
3. Trains for **only 1 epoch** — severely undertrained
4. No learning rate scheduling, no class weighting
5. Default classification threshold of 0.5
6. Achieves F1=0.48 on dev

**Why the baseline is weak** (and where we can improve):
- 1 epoch is grossly insufficient for fine-tuning
- Downsampling throws away ~85% of negative examples (valuable context for what PCL is NOT)
- No threshold optimization
- No proper evaluation during training
- `roberta-base` is decent but not the strongest available model

### 2.3 What Makes PCL Detection Hard

From the paper analysis:
1. **Subtlety**: PCL is unconscious, well-intentioned — no obvious lexical markers like hate speech
2. **World knowledge**: Detecting "shallow solutions" or "presuppositions" requires understanding context
3. **Subjectivity**: Even human annotators only achieve 41% Kappa agreement (moderate)
4. **Imbalance**: ~9.4% positive rate means naive classifiers default to predicting "No PCL"
5. **Community variation**: PCL patterns differ across vulnerable communities (homeless, refugees, etc.)
6. **Confusable patterns**: Emotional/flowery language about vulnerable groups isn't always patronizing

---

## 3. Strategic Approach

### 3.1 The "Novel" Contribution

The professor clarified: **"Novel = You need to go beyond using a pre-trained model."** Simply using DeBERTa instead of RoBERTa is NOT sufficient.

**Our proposed approach: "Context-Enriched DeBERTa with Focal Loss for PCL Detection"**

Three novel components beyond the baseline:

#### Component 1: Community-Aware Input Representation
- **What**: Prepend `"[Community: {keyword}] "` to each text before tokenization
- **Why**: The paper (Table 2) shows PCL patterns vary dramatically across communities. "Homeless" has 600 PCL spans vs "immigrant" with only 102. The types of PCL also differ — "shallow solution" is heavily associated with "homeless" and "in-need", while "presupposition" clusters with "hopeless" and "poor-families"
- **Mechanism**: By explicitly telling the model which community is being discussed, we enable it to learn community-specific PCL patterns. This is analogous to how NLI models benefit from explicit hypothesis-premise formatting
- **Evidence**: The test set includes keyword/country metadata, so this is applicable at inference time

#### Component 2: Focal Loss for Extreme Class Imbalance
- **What**: Replace standard cross-entropy with focal loss: FL(p_t) = -α_t(1-p_t)^γ log(p_t)
- **Why**: The baseline's downsampling approach discards ~85% of negative examples. Focal loss instead keeps ALL data but down-weights easy (well-classified) examples, focusing training on the hard boundary cases that matter most for PCL detection
- **Parameters**: α=0.75 (for positive class weighting), γ=2.0 (focusing parameter)
- **Advantage over alternatives**: Unlike simple class weights which uniformly up-weight all positives, focal loss specifically targets hard-to-classify examples — exactly the subtle PCL instances that drive the task's difficulty

#### Component 3: Threshold Optimization via F1 Maximization
- **What**: Instead of default threshold=0.5, systematically search for the threshold that maximizes F1 on a held-out validation set
- **Why**: With severe class imbalance, the optimal decision boundary is NOT at 0.5. The model's sigmoid outputs are calibrated to the class prior, so a lower threshold (typically 0.3-0.4) better captures the rare positive class
- **Method**: After training, compute F1 at thresholds from 0.1 to 0.9 in steps of 0.01 on an internal validation set; select the threshold with highest F1

### 3.2 Model Choice: DeBERTa-v3-base

**Why DeBERTa-v3 over RoBERTa**:
- DeBERTa's disentangled attention separates content and position information, better capturing the positional patterns of PCL (e.g., framing language that appears at the start of paragraphs)
- Enhanced mask decoder provides stronger pre-training signal
- DeBERTa-v3 uses ELECTRA-style pre-training (replaced token detection), which is more sample-efficient — critical for our small positive class
- Consistently outperforms RoBERTa on SuperGLUE and similar NLU benchmarks
- The `-base` size (184M params) is trainable on a single GPU within reasonable time
- DeBERTa was the backbone of many top-performing SemEval 2022 Task 4 systems

### 3.3 Expected Performance (Informed by SemEval 2022 Results)

**Official competition results** (79 teams participated in Subtask 1):
| Percentile | Approximate F1 (Test) |
|---|---|
| Top 10% (rank 1-8) | ~0.59 - 0.66 |
| Top 30% (rank 1-24) | ~0.54 - 0.66 |
| Median (~rank 40) | ~0.47 - 0.50 |
| Baseline | 0.49 |

**Top systems used**: DeBERTa (large/xlarge), focal loss, threshold optimization, ensembles, data augmentation (back-translation), multi-task learning with Task 2 categories. This directly validates our approach.

- Baseline: F1=0.48 (dev), 0.49 (test)
- Our target: F1=0.55-0.62 (dev) — aiming for top 30%, stretch for top 10%
- Breaking this down:
  - Proper training (5 epochs + early stopping): +0.05-0.08
  - DeBERTa-v3 over RoBERTa: +0.02-0.04
  - Focal loss over downsampling: +0.02-0.04
  - Threshold optimization: +0.02-0.05
  - Community-aware input: +0.01-0.03
  - Optional K-fold ensemble: +0.01-0.03 additional

---

## 4. Execution Plan with Sub-Agents

### Overview: Phased Execution

```
Phase 1: Foundation [~3 hours]
├── Sub-agent A: Paper Review (Exercise 1)          ← PARALLEL
├── Sub-agent B: EDA Notebook (Exercise 2)           ← PARALLEL
└── Sub-agent C: Research SOTA approaches            ← PARALLEL

Phase 2: Implementation [~4 hours]
├── Main Agent: Data preprocessing pipeline
├── Main Agent: Model training script (DeBERTa + focal loss)
├── Main Agent: Threshold optimization
└── Main Agent: Prediction generation

Phase 3: Evaluation [~2 hours]
├── Sub-agent D: Error analysis (Exercise 5.2)       ← PARALLEL
└── Sub-agent E: Generate evaluation figures          ← PARALLEL

Phase 4: Report & Packaging [~2 hours]
├── Sub-agent F: Write LaTeX report sections          ← PARALLEL
├── Sub-agent G: Organize BestModel folder            ← PARALLEL
└── Main Agent: Final review + commit
```

### Phase 1: Foundation (Paper Review + EDA)

**These three sub-agents run in PARALLEL since they have no dependencies.**

#### Sub-Agent A: Paper Review (Exercise 1)
- **Type**: `general-purpose`
- **Task**: Read `paper.md` and `Appendix-Reviewing-NLP-Research-Papers.md`. Produce structured answers for:
  - Q1: Primary contributions (2 marks)
  - Q2: Technical strengths (2 marks)
  - Q3: Key weaknesses (2 marks)
- **Output**: Write answers directly into the report LaTeX or a markdown file for later inclusion
- **Key guidance for the agent**:
  - Contributions: (1) The Don't Patronize Me! dataset with 10,637 annotated paragraphs, (2) A two-level taxonomy of 7 PCL categories grouped into 3 higher-level categories, (3) Baseline experiments showing BERT-based models can detect PCL
  - Strengths: Expert annotation (not crowdsourced), rigorous two-step annotation process, multi-country/multi-keyword coverage, the 5-point annotation scale captures uncertainty
  - Weaknesses: Only 3 annotators (limited perspective), moderate IAA (41%), no ablation studies on annotation decisions, experiments limited to basic models without advanced techniques, no cross-domain evaluation

#### Sub-Agent B: EDA Notebook (Exercise 2)
- **Type**: `general-purpose`
- **Task**: Create `src/eda.ipynb` performing two EDA techniques:

  **Technique 1: Class Distribution & Community Analysis**
  - Bar chart showing class distribution (PCL vs No PCL) — visualize the ~10:1 imbalance
  - Heatmap showing PCL rate broken down by keyword community (10 keywords)
  - Cross-tabulation table
  - **Impact statement**: The severe imbalance justifies focal loss over simple downsampling. Community-specific PCL rates motivate our community-aware input representation

  **Technique 2: Text Length Distribution Analysis**
  - Overlapping histograms of text length (in tokens/words) for PCL vs non-PCL
  - Box plots comparing length distributions
  - Compute statistics: mean, median, 95th percentile for each class
  - **Impact statement**: Informs `max_seq_length` choice (should capture 95th percentile). If PCL texts are systematically longer, this is a discriminative feature the model can learn

- **Output**: Jupyter notebook with clear markdown cells for analysis, saved figures for the report

#### Sub-Agent C: SOTA Research (informational)
- **Type**: `general-purpose` (already launched — see background agent)
- **Task**: Research SemEval 2022 Task 4 winning approaches
- **Output**: Summary of top-performing techniques to validate/refine our approach

### Phase 2: Implementation

**This is the core implementation phase. Steps are sequential (each depends on the previous).**

#### Step 2.1: Data Preprocessing Pipeline
- **Agent**: Main agent
- **Create**: `src/data_utils.py`
- **Tasks**:
  1. Load `dontpatronizeme_pcl.tsv` (handle the 4-line disclaimer header)
  2. Join with train/dev split files using `par_id`
  3. Apply binary labeling: labels 0-1 → 0, labels 2-4 → 1
  4. For training: create internal train/validation split (85/15) from train data, stratified by label
  5. For test: load `task4_test.tsv` and parse correctly
  6. Implement community-aware text formatting: `f"[Community: {keyword}] {text}"`
  7. Create HuggingFace `Dataset` objects for efficient tokenization

#### Step 2.2: Model Training Script
- **Agent**: Main agent
- **Create**: `src/train.py` (or `BestModel/train.ipynb` for Colab)
- **Architecture**:
  ```
  Input text → DeBERTa-v3-base tokenizer → DeBERTa-v3-base encoder →
  Classification head (768 → 256 → 1) → Sigmoid → Focal Loss
  ```
- **Training Configuration**:
  | Hyperparameter | Value | Rationale |
  |---|---|---|
  | Model | `microsoft/deberta-v3-base` | Best NLU performance in base-size |
  | Max seq length | 256 | Covers 95%+ of paragraphs (verify in EDA) |
  | Batch size | 16 | Fits in Colab T4 GPU (16GB) |
  | Gradient accumulation | 2 | Effective batch size = 32 |
  | Learning rate | 2e-5 | Standard for fine-tuning transformers |
  | LR scheduler | Linear warmup (10%) + linear decay | Prevents catastrophic forgetting |
  | Epochs | 5 | With early stopping (patience=2) |
  | Optimizer | AdamW | weight_decay=0.01 |
  | Focal loss α | 0.75 | Up-weight positive class |
  | Focal loss γ | 2.0 | Standard focusing parameter |
  | Seed | 42 | Reproducibility |

- **Training loop**:
  1. Use ALL training data (no downsampling) — focal loss handles imbalance
  2. Evaluate on internal validation split every epoch
  3. Save best model checkpoint (by validation F1)
  4. Early stopping if F1 doesn't improve for 2 epochs

#### Step 2.3: Threshold Optimization
- **Agent**: Main agent
- **Add to**: `src/evaluate.py`
- **Method**:
  1. Load best model checkpoint
  2. Get prediction probabilities on official dev set
  3. Sweep threshold from 0.1 to 0.9 (step 0.01)
  4. Compute F1 at each threshold
  5. Select optimal threshold
  6. Plot threshold vs F1 curve (useful for report)

#### Step 2.4: Generate Predictions
- **Agent**: Main agent
- **Create**: `src/predict.py`
- **Tasks**:
  1. Load best model + optimal threshold
  2. Generate predictions for official dev set (2,094 paragraphs)
     - **Critical**: Output in order of par_ids from `dev_semeval_parids-labels.csv`
  3. Generate predictions for official test set (3,832 paragraphs)
     - **Critical**: Output in order of lines in `task4_test.tsv`
  4. Save as `dev.txt` and `test.txt` — one prediction (0 or 1) per line

### Phase 3: Evaluation & Error Analysis

**These sub-agents can run in PARALLEL after Phase 2.**

#### Sub-Agent D: Error Analysis (Exercise 5.2 — 2.5 marks)
- **Type**: `general-purpose`
- **Task**: Analyze model errors on the official dev set (where we have labels)
- **Analyses**:
  1. **Confusion matrix**: Show TP, FP, FN, TN counts
  2. **False positive analysis**: Sample 10 FP examples, analyze WHY the model predicted PCL when it's not:
     - Are they borderline cases (original label 1)?
     - Do they contain emotional/flowery language that mimics PCL?
     - Which community keywords appear most in FPs?
  3. **False negative analysis**: Sample 10 FN examples, analyze WHY the model missed PCL:
     - What category of PCL was missed? (using Task 2 labels from dev split)
     - Are they subtle, world-knowledge-dependent cases?
  4. **Compare with baseline**: Run baseline predictions and show Venn diagram of what each model gets right/wrong
- **Output**: Markdown/tables/figures for the report

#### Sub-Agent E: Local Evaluation Metrics (Exercise 5.2 — 2.5 marks)
- **Type**: `general-purpose`
- **Task**: Additional local evaluation beyond error analysis
- **Analyses**:
  1. **Performance by community keyword**: Bar chart showing F1/precision/recall broken down by the 10 keywords. Are some communities harder to classify?
  2. **Ablation study**: Compare performance of:
     - Full model (community-aware + focal loss + threshold opt)
     - Without community-aware input
     - Without focal loss (use standard CE with class weights)
     - Without threshold optimization (use 0.5)
  3. **Precision-Recall curve**: Plot PR curve, show AUC-PR
  4. **Confidence calibration**: How well do predicted probabilities reflect true probabilities?
- **Output**: Figures and analysis text for the report

### Phase 4: Report & Packaging

**These sub-agents can run in PARALLEL.**

#### Sub-Agent F: Write LaTeX Report
- **Type**: `general-purpose`
- **Task**: Write the report sections in `report/main.tex`
- **Structure**:
  - Chapter 1 (Exercise 1): Paper review — Q1, Q2, Q3
  - Chapter 2 (Exercise 2): EDA — 2 techniques with figures, analysis, impact
  - Chapter 3 (Exercise 3): Proposed approach — description, rationale, expected outcome
  - Chapter 4 (Exercise 5.2): Evaluation — error analysis + local evaluation
  - Abstract: Brief summary of approach and results
  - Front page: GitHub link, team name for leaderboard
- **Notes**:
  - Include citations for DeBERTa, focal loss paper, original PCL paper
  - Include figures from EDA and evaluation
  - Keep it concise — GTAs don't have time for long reports

#### Sub-Agent G: Organize Repository
- **Type**: `general-purpose`
- **Task**: Set up `BestModel/` folder and finalize repo structure
- **Contents of BestModel/**:
  - `train.ipynb` or `train.py` — the training code
  - Model weights (or link to them if too large)
  - `README.md` — instructions to reproduce
- **Also**:
  - Ensure `dev.txt` and `test.txt` are at the root (or easy to find)
  - Verify `.gitignore` is correct
  - Clean up any temporary files

---

## 5. Technical Deep-Dive

### 5.1 Focal Loss Implementation

Standard cross-entropy: CE(p, y) = -y·log(p) - (1-y)·log(1-p)

Focal loss modification:
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        # p_t = probability of correct class
        p_t = probs * targets + (1 - probs) * (1 - targets)
        # alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        # focal modulation
        focal_weight = (1 - p_t) ** self.gamma
        loss = -alpha_t * focal_weight * torch.log(p_t + 1e-8)
        return loss.mean()
```

**Why focal loss works here**: In our dataset, ~90% of examples are easy negatives that the model quickly learns to classify correctly. With standard CE, these easy examples dominate the gradient, drowning out the learning signal from the hard positives. Focal loss reduces the contribution of easy examples by the factor (1-p_t)^γ, redirecting learning capacity to the difficult borderline cases.

### 5.2 Community-Aware Input Format

```
Original:  "He said their efforts should not stop only at creating..."
Formatted: "[Community: poor-families] He said their efforts should not stop only at creating..."
```

The model tokenizes this naturally as special context tokens. DeBERTa's attention mechanism can then learn to associate community-specific patterns with PCL predictions. For example:
- For "homeless", shallow solutions like fundraising drives are common PCL
- For "women", presuppositions about capabilities are common PCL
- For "refugees", saviour narratives are common PCL

### 5.3 Data Pipeline

```
dontpatronizeme_pcl.tsv
        │
        ├── Join with train_semeval_parids-labels.csv
        │   ├── Binary label: {0,1} from 5-point scale (>=2 is positive)
        │   ├── Task 2 labels: 7-dim vector (for analysis only)
        │   └── 85/15 stratified split → internal train + internal val
        │
        ├── Join with dev_semeval_parids-labels.csv
        │   └── Official dev set (2,094 examples) — our "test set"
        │
        └── task4_test.tsv
            └── Official test set (3,832 examples) — blind predictions
```

### 5.4 Prediction File Format

**dev.txt**: 2,094 lines, one per paragraph, in order of par_ids from `dev_semeval_parids-labels.csv`
```
0
1
0
0
1
...
```

**test.txt**: 3,832 lines, one per paragraph, in order of lines in `task4_test.tsv`
```
0
1
0
...
```

### 5.5 Where to Train

**Option A: Google Colab (Recommended)**
- Free T4 GPU (16GB VRAM) — sufficient for DeBERTa-v3-base
- Training time estimate: ~15-25 minutes for 5 epochs
- Use `BestModel/train.ipynb` as the Colab notebook

**Option B: Local Mac (Apple Silicon MPS)**
- If M1/M2/M3 Mac, can use `device='mps'` with PyTorch
- Slower than T4 but workable for base-size models
- Training time estimate: ~30-45 minutes

**Option C: Local Mac (CPU only)**
- Very slow, not recommended
- Could work for small experiments only

---

## 6. Repository Structure

```
SemEval-2022-Task-4/
├── data/                          # (gitignored)
│   ├── dontpatronizeme_pcl.tsv
│   ├── practice-splits/
│   │   ├── train_semeval_parids-labels.csv
│   │   └── dev_semeval_parids-labels.csv
│   └── test/
│       └── task4_test.tsv
├── baseline/
│   └── Reconstruct_and_RoBERTa_baseline_train_dev_dataset.ipynb
├── src/
│   ├── eda.ipynb                  # Exercise 2: EDA notebook
│   ├── data_utils.py              # Data loading & preprocessing
│   ├── train.py                   # Training script
│   ├── model.py                   # Model definition (focal loss, etc.)
│   ├── evaluate.py                # Evaluation + threshold optimization
│   ├── predict.py                 # Generate dev.txt and test.txt
│   └── error_analysis.ipynb       # Exercise 5.2: Error analysis notebook
├── BestModel/
│   ├── train.ipynb                # Self-contained training notebook (for Colab)
│   ├── model/                     # Saved model weights (or link)
│   └── README.md                  # Reproduction instructions
├── report/
│   ├── main.tex                   # Main report
│   ├── includes.tex               # LaTeX config
│   ├── notation.tex               # Notation macros
│   ├── titlepage.tex              # Title page
│   └── figures/                   # Report figures
│       ├── imperial.pdf
│       ├── class_distribution.png
│       ├── community_heatmap.png
│       ├── text_length_dist.png
│       ├── confusion_matrix.png
│       ├── pr_curve.png
│       ├── threshold_optimization.png
│       ├── keyword_performance.png
│       └── ablation_results.png
├── dev.txt                        # Dev set predictions
├── test.txt                       # Test set predictions
├── paper.md                       # PCL paper
├── task.md                        # Coursework spec
├── faq.md                         # Ed forum Q&A
├── PLAN.md                        # This file
├── .gitignore
└── README.md                      # Repository overview (to be created)
```

---

## 7. Risk Mitigation

### Risk 1: Model doesn't beat baseline
- **Mitigation**: The baseline is very weak (1 epoch, heavy downsampling). Even proper training of RoBERTa-base should exceed 0.48 F1. DeBERTa + focal loss + threshold optimization provides multiple avenues for improvement
- **Fallback**: If DeBERTa fails, fall back to RoBERTa-base with just proper training + threshold optimization

### Risk 2: GPU/compute issues
- **Mitigation**: Have both Colab and local MPS options ready. Training script should auto-detect device
- **Fallback**: Reduce max_seq_length to 128 or use a smaller model (DeBERTa-v3-small)

### Risk 3: Overfitting on small positive class
- **Mitigation**: Early stopping on validation F1, focal loss prevents over-focusing on easy examples, no downsampling preserves all data
- **Fallback**: Add dropout (0.1-0.2) to classification head, reduce learning rate

### Risk 4: Running out of time
- **Priority order**:
  1. Get a working model that beats baseline (ensures marks for Ex 4, 5.1)
  2. Paper review (Ex 1 — 6 marks, relatively quick)
  3. EDA (Ex 2 — 6 marks)
  4. Error analysis (Ex 5.2 — 5 marks)
  5. Report polish (Ex 6 — 2 marks)
  6. Approach description (Ex 3 — 4 marks, can be written last)

### Risk 5: Test set format mismatch
- **Mitigation**: Carefully verify:
  - dev.txt has exactly 2,094 lines matching par_id order in dev_semeval_parids-labels.csv
  - test.txt has exactly 3,832 lines matching line order in task4_test.tsv
  - Each line contains only "0" or "1"

---

## 8. Mark Allocation Strategy

| Exercise | Marks | Difficulty | Time Est. | Priority |
|---|---|---|---|---|
| Ex 1: Paper Review | 6 | Low | 1.5h | HIGH |
| Ex 2: EDA | 6 | Medium | 2h | HIGH |
| Ex 3: Approach Description | 4 | Low | 1h | MEDIUM (write last) |
| Ex 4: Model Training | 1 | High | 4h | HIGH |
| Ex 5.1: Predictions | 6 | High | 0.5h | HIGHEST |
| Ex 5.2: Error Analysis | 5 | Medium | 2h | HIGH |
| Ex 6: Report Quality | 2 | Low | 1h | MEDIUM |
| **Total** | **30** | | **~12h** | |

**Strategy**: Maximize marks per hour. Ex 5.1 (6 marks) depends on Ex 4 (1 mark), so model training is critical path. Ex 1 and Ex 2 can be done in parallel with model training. Ex 3 should be written last since it describes the final submitted approach.

### Marks Breakdown for Ex 5.1:
- 1 mark: Submit dev.txt + test.txt in correct format ← nearly free
- 1 mark: dev.txt F1 > 0.48 ← should be achievable
- 1 mark: test.txt F1 > 0.49 ← should be achievable
- 1 mark: Top 60% on leaderboard ← likely with our approach
- 1 mark: Top 30% on leaderboard ← possible with our approach
- 1 mark: Top 10% on leaderboard ← stretch goal

---

## Appendix: Alternative Approaches Considered

### A1: Ensemble of Multiple Models
- **Idea**: Average predictions from DeBERTa-v3, RoBERTa-large, BERT-large
- **Pros**: More robust, reduces variance
- **Cons**: 3x training time, harder to explain as a single "approach"
- **Verdict**: Consider if time permits, but single DeBERTa should be sufficient

### A2: Multi-Task Learning with PCL Categories
- **Idea**: Jointly predict binary PCL + 7 category labels (Task 2)
- **Pros**: Category prediction as auxiliary task improves representation
- **Cons**: More complex architecture, Task 2 labels only for positive examples
- **Verdict**: Good idea but adds complexity; save for if primary approach underperforms

### A3: Data Augmentation via Back-Translation
- **Idea**: Translate PCL examples to French/German and back to generate paraphrases
- **Pros**: Increases positive class size, adds diversity
- **Cons**: Risk of losing subtle PCL nuances in translation, extra compute
- **Verdict**: Interesting but risky for PCL specifically (subtlety might be lost)

### A4: Contrastive Learning Pre-training
- **Idea**: Pre-train with contrastive loss on PCL vs non-PCL before fine-tuning
- **Pros**: Better representation of PCL boundaries
- **Cons**: Complex, needs careful negative sampling, extra training time
- **Verdict**: Too complex for the time available

### A5: K-Fold Cross-Validation Ensemble
- **Idea**: 5-fold CV, average predictions from all 5 models
- **Pros**: Uses all data for both training and validation, robust
- **Cons**: 5x training time
- **Verdict**: Very practical — implement if time allows. Would be a natural extension.
