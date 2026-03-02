## Appendix: Reviewing NLP Research Papers

When you are asked to review an NLP research paper, you are essentially
playing the role of a **technical judge**. You aren't just summarising the paper;
you are evaluating its quality and honesty.
Most formal reviews ask for these five specific components:

**1. The Summary (The "What")**
The review usually starts with a brief, objective summary.
    ● **What to answer:** What problem are they solving? What is their main
       contribution (a new dataset, a better algorithm, or a clever experiment)?
    ● **The goal:** Prove to the editors/instructors that you fully understood the
       technical core of the work.
**2. Strengths and Weaknesses (The "How Good")**
This is the heart of the review. You need to look at:
    ● **Originality:** Is this a new idea, or did they just "re-package" an old
       method?
    ● **Soundness:** Are the math and the experimental setup correct? For
       example, did they use a fair baseline to compare against?
    ● **Significance:** Does this paper actually matter to the NLP community?
**3. Evaluation and Results**
You must scrutinize the "Experiments" section.
    ● **Metrics:** Did they use the right tools to measure success? For example,
       using **Accuracy** on a dataset where 99% of the labels are the same is
       misleading.
    ● **Ablation Studies:** Did they prove that their specific change is what
       caused the improvement?
    ● **Error Analysis:** Did the authors look at where their model _failed_? A paper
       that only shows wins is usually hiding something.


**4. Clarity and Reproducibility**
    ● **Writing:** Is the paper easy to follow, or is it buried in unnecessary jargon?
    ● **The "Recipe":** Could a stranger recreate this exact model using only the
       details in the paper? If they didn’t provide hyperparameter settings (like
       learning rates or batch sizes), the answer is likely "No."
**5. Recommendation (The "Verdict")**
Finally, you are asked to give a score or a recommendation.
    ● **Accept:** The paper is solid, new, and well-explained.
    ● **Weak Accept:** Good idea, but needs better experiments or clearer
       writing.
    ● **Reject:** The math is wrong, the results aren't significant, or the authors
       ignored existing work that already solved the problem.
**Comparison of a Good vs. Bad Review**
    **Feature A "Bad" Review A "Good" Review**
    **Tone** "This paper is bad
       and I didn't like
       it."
          "While the idea is interesting, the lack of
          an ablation study makes it hard to verify
          the results."
    **Specificity** "The results are
       weak."
          "The model only outperforms the
          baseline by 0.2 BLEU points, which may
          not be statistically significant."
    **Advice** "Fix the
       grammar."
          "I suggest the authors clarify the
          transition between Section 3 and 4 to
          improve flow."