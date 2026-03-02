## Appendix: Exploratory Data Analysis

In NLP, **Exploratory Data Analysis (EDA)** focuses on the linguistic properties,
patterns, and potential biases hidden in the text.
Here is a breakdown of the typical NLP EDA workflow:

**1. Basic Statistical Profiling**
Before looking at the words, you look at the structure. This helps you
determine your model's constraints (like maximum sequence length).
    ● **Token Count:** What is the average, minimum, and maximum sentence
       length?
    ● **Vocabulary Size:** How many unique words exist? This dictates the size of
       your embedding layer.
    ● **Class Distribution:** Is the dataset balanced? (e.g., In a hate speech task, if
       98% of the data is "Non-Toxic," your model might achieve 98% accuracy
       just by guessing "Non-Toxic" every time).
**2. Lexical Analysis (The "Word" Level)**
This involves digging into the actual language used in the dataset.
    ● **N-gram Analysis:** What are the most common pairs (bigrams) or triplets
       (trigrams) of words? This reveals common phrases or domain-specific
       jargon.
    ● **Stop Word Density:** How much of the text is "filler" (the, is, at)? High
       density might mean you need more aggressive cleaning.
    ● **Word Clouds & Frequency:** A quick visual check to see if the most
       frequent words actually align with the task.
**3. Semantic & Syntactic Exploration**
Modern NLP requires understanding the "meaning" behind the statistics.


```
● Part-of-Speech (POS) Tagging: Are there more verbs than nouns? (e.g., in
instruction-following tasks, verbs are dominant).
● Named Entity Recognition (NER): Does the dataset focus on specific
people, locations, or organizations?
● Embedding Visualization: Using techniques like t-SNE or UMAP to
project high-dimensional word vectors into 2D space. This allows you to
see if similar concepts are naturally clustering together before you even
train a model.
```
**4. Identifying "Noise" and Artifacts**
The most important part of EDA is finding the "trash" in your data:
    ● **Duplicates:** Repeated entries can lead to data leakage (the model seeing
       the same sentence in both training and testing).
    ● **Special Characters/HTML:** Finding hidden tags like &amp; or \n that
       could confuse a tokenizer.
    ● **Outliers:** Extremely long or short sequences that might be errors in data
       collection.
**Why is EDA critical for your coursework?**
If you skip EDA and go straight to training, you are flying blind. EDA tells you:
    1. **If you need to augment your data** (if the classes are imbalanced).
    2. **What your max_length should be** (to avoid cutting off important info).
    3. **If your task is "too easy"** (e.g., if the model can guess the answer just by

## looking for a specific keyword).