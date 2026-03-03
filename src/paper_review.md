# Paper Review: "Don't Patronize Me! An Annotated Dataset with Patronizing and Condescending Language towards Vulnerable Communities"

**Authors:** Carla Perez-Almendros, Luis Espinosa-Anke, Steven Schockaert
**Venue:** COLING 2020
**Reviewer:** Abhivir

---

## Q1. Primary Contributions (2 marks)

This paper makes three primary contributions. First, it introduces the **Don't Patronize Me! dataset**, a novel resource of 10,637 paragraphs extracted from news stories across 20 English-speaking countries, annotated for the presence of patronizing and condescending language (PCL) directed at vulnerable communities such as refugees, homeless people, and immigrants (Section 4). Second, the authors propose a **two-level taxonomy of seven PCL categories** organized under three higher-level groups -- The Saviour (unbalanced power relations, shallow solution), The Expert (presupposition, authority voice), and The Poet (compassion, metaphor, the poorer the merrier) -- grounded in sociolinguistic literature on how media discourse can routinize discrimination (Sections 3 and 4.1). Third, the paper provides **baseline experiments** across two tasks -- binary PCL detection (Task 1) and multi-label PCL categorization (Task 2) -- using SVM, BiLSTM, and transformer-based models (BERT, RoBERTa, DistilBERT), establishing that the task is feasible yet challenging, with RoBERTa achieving the best overall F1 of 70.68 on Task 1 (Section 5, Tables 3-4). Together, these contributions open a new research direction in NLP for detecting subtle, well-intentioned but harmful language.

---

## Q2. Technical Strengths (2 marks)

**Novelty and significance of the task framing.** The paper identifies a genuinely underexplored niche in harmful language detection: language that is patronizing rather than overtly hostile. As the authors argue in Section 1, PCL differs from hate speech and offensive language because it is "generally used unconsciously and with good intentions," making it both socially important and technically challenging. The related work in Section 2 confirms that, at the time of writing, only Wang and Potts (2019) had addressed condescension computationally, and that work focused on direct interpersonal communication rather than media discourse about vulnerable communities.

**Carefully designed annotation scheme.** The two-step annotation process (Section 4.2) is methodologically sound. The use of a 3-point scale (0/1/2) per annotator to handle borderline cases, combined into a 5-point aggregated scale (Section 4.2.1), is a pragmatic and transparent approach to managing the inherent subjectivity of the task. The involvement of a third annotator to resolve total disagreements (0 vs. 2 cases) adds rigour. Inter-annotator agreement is reported honestly: 41% Kappa across all labels, rising to 61% when borderline cases are excluded.

**Thorough error analysis.** Tables 5 and 6 provide concrete examples of model failures, which is more informative than aggregate metrics alone. The analysis in Section 5 correctly identifies that categories requiring world knowledge (e.g., Metaphor with F1 of 43.44 for RoBERTa, and Shallow Solution) are harder to detect, offering actionable direction for future work. The observation that BERT-large underperforms BERT-base (F1 53.89 vs. 67.44, Table 3), likely due to overfitting on limited training data, is a useful practical insight.

**Reproducibility.** The authors report hyperparameters for all models (Section 5), fix random seeds, use 10-fold cross-validation, and release the dataset publicly, which supports reproducibility.

---

## Q3. Key Weaknesses (2 marks)

**Severe class imbalance is insufficiently addressed.** Only 995 of 10,637 paragraphs (approximately 9.4%) are positive examples for Task 1, yet the authors report only precision, recall, and F1 for the positive class (Table 3) without discussing how they handled this imbalance during training. There is no mention of oversampling, class weighting, or stratified splitting strategies. The reported metrics would be more convincing with macro/micro-averaged F1 or with confidence intervals across the 10 folds, as the current single-point estimates make it difficult to assess statistical significance of differences between models.

**No ablation or feature analysis.** The paper benchmarks several models but does not investigate what linguistic features drive predictions. For instance, the authors note in Section 5 that words like "us," "they," and "help" are strong indicators for Unbalanced Power Relations, but this claim is not supported by any feature importance analysis, attention visualization, or lexical overlap study. An ablation study -- such as removing keyword features or testing on out-of-domain vulnerable groups not seen during training -- would strengthen the claims about task difficulty.

**Span-level annotations are collected but not evaluated.** Section 4.2.2 describes a careful span-level annotation process using BRAT, yet the experiments in Section 5 treat Task 2 as a paragraph-level multi-label classification problem, discarding span boundary information entirely. The authors acknowledge this ("span boundaries are not used as part of the training data") but do not attempt any span-level evaluation, which is a missed opportunity given that span identification would be more useful in practice for explaining model predictions or flagging specific problematic phrases.

**Limited discussion of annotator bias and dataset scope.** The three annotators share backgrounds in "communication, media and data science" (Section 4), but the paper does not discuss how their cultural or demographic backgrounds might influence PCL judgments, which are inherently subjective. Furthermore, the keyword-based paragraph selection strategy (Section 4) introduces a systematic bias: the dataset only contains paragraphs mentioning one of ten predefined keywords, meaning PCL expressed through indirect references to vulnerable communities would be missed entirely. The authors mention plans for an "extended data statement" but do not include it in the paper itself.
