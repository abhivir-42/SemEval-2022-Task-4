These are the questions some people have asked on edstem and Chiraag Lala (the professor) and some teaching assistants have answered:


1: 
Hi,

Given I am fine-tuning an existing model for the task, my model size is ~2GB. Exercise 4 says to push the model to GitLab too, so would you like me to push this directly despite file size?

Also, I'm guessing we do not need to push anything except for model code and the model itself (i.e not the dataset, so running the code alone from the repo would encounter errors)

Thanks.

Comment

1 Answer
Chiraag Lala
3 weeks ago

2

Upload your model and code to your GitLab/Github repository if possible. If the model is too large, include a link to it. We will manually inspect your code but not run it (we don't have the time for that!). But, reproducibility is important, so make sure your README explains everything needed to replicate your work and run it. The reviewers will check whether the README is clear and it has instructions to reproduce your work. They will not actually reproduce the results themselves. So, from that perspective, include links to everything (including datasets) in the README.

I hope this helps :)



2: 
How rigorous do citations need to be for the report, if any? I saw a post earlier that mentioned citing a paper since they were improving upon a specific approach, which I'd agree with. I was just wondering about something like citing the paper that introduces RoBERTa.

Comment

1 Answer
Chiraag Lala
2 weeks ago

1

If you're training a model from scratch, no need to cite it (most likely you're not doing that). If you're using a pretrained model and then applying your approach, modification, or fine-tuning, you should cite it. Make sure it's clear which work is yours and which isn't.

I hope this helps :)



3:
If a component was included in our best model but ablation studies show it actually hurts performance, should we remove it from the final model or is it acceptable to keep it, discuss the finding in the evaluation, and reflect on why we hypothesised it would help?

Comment

1 Answer
Chiraag Lala
Last week


Up to you. If you have time, you can go back and remove it from the final model because you want to present your 'best' model. But if you don't have time, you can keep it and discuss it in local evaluation. Both are acceptable.

I hope this helps :)  



4:
Hi, wanted to clarify on the meaning of "failed" experiment. Say for example my ablation study shows that novel features added to my proposed approach versus simply using a base model and finetuning with hyperparameter tricks (e.g. early stopping) yields comparable model performance:

Can I submit the base model with finetuning with hyperparam tricks (as long as it outperforms the paper baseline of 0.48 and 0.49)

Explaining in local evaluation why all our novel features might not have worked will yield the marks under "Local Evaluation"

Or should we go back and try more approaches :/


Comment

1 Answer
Chiraag Lala
7 days ago


Ah! I see your point.

If this were a long-term project, I would have advised you to evaluate and analyse all the approaches you try. However, as this is a short coursework and GTAs don't have time to read long reports, I suggest you evaluate and analyse only the BestModel you propose.

I hope this helps :)



5:
i've noticed this step in EDA on lexical analysis , does it even make sense to discuss this considering that we know that Roberta is baseline and we will probably use transformer to beat it too? Since its SOTA now. Especially "stop word density" doesn't make sense to me...


Same goes for "part of speech tagging" , while i understand its usage in traditional NLP it seems to me that 


Comment

1 Answer
Chiraag Lala
7 days ago


Yes, that is okay. The examples in the Appendix are just to help you understand different kinds of EDA methods. It is certainly not an exhaustive list. Nor am I asking you to do all of them. If you don't see the point of a specific EDA, don't do it. It is fine :)



6:
Hi Chiraag,

I had a question about the order of the predictions in the dev.txt file. Should the order be:

   1. Order of the official dev set (i.e., dev_semeval_parids-labels.csv). So, in the same order as the par_ids in the file, so 4046, 1279, 8330, etc.

   2. Order of the full dataset (i.e., dontpatronizeme_pcl.tsv). So, essentially, filter the dev par_ids from the full dataset and predictions match the order of the full dataset.

I am assuming that (1) is the intended order but I wanted to make sure. 

For test.txt it would just simply be the line order in the test dataset file.

Thanks in advance!

Comment

1 Answer
Chiraag Lala
7 days ago

1

(1) is the intended order.

I hope this helps :)


7:
If we decide to do the coursework in a ipynb file, what should we do for the BestModel folder? Should we just put the file in the folder to begin with, or do we need to extract the code for the model only from our notebook into a new notebook, even if that means that the latter won't work independently? I obviously also understand that model weights will be saved in the folder, but I am more concerned about the "code" we need to have in that folder.

Comment

1 Answer
Chiraag Lala
4 days ago

1

Put the ipynb file in the BestModel folder. We will treat the ipynb file as the code.

I hope this helps :)

Comment

Anonymous


8:
Is it cool if we do one main notebook.ipynb at the root for EDA + analysis + report-facing work, and something like a BestModel/train.py script for the actual model, plus scripts for experiments on ablation studies?


1
Reply

Chiraag Lala
4d
Sure. That's alright :)


9:
Is using any model that is not RoBERTa-base baseline model considered novel? 

For example, could we use another pre-trained model eg. DistilBERT and call this the novelty? Then, we will explain why we expect DistilBERT to work better than RoBERTa-base. 

Or is it novel only if we do something beyond using the pre-trained DistilBERT, such as augmenting the training data / modifying the architecture of the pre-trained model etc?

Thank you!

Comment

1 Answer
Chiraag Lala
3 days ago


Novel = You need to go beyond using a pre-trained model.

Clearly state what your contribution is and what someone else's contribution is by citing the pretrained model.

I hoep this helps :)


