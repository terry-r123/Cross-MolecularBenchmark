---
license: mit
widget:
 - text: "MQIFVKTLTGKTITLEVEPS<mask>TIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
---
# **ESM-1b**

ESM-1b ([paper](https://www.pnas.org/content/118/15/e2016239118#:~:text=https%3A//doi.org/10.1073/pnas.2016239118), [repository](https://github.com/facebookresearch/esm)) is a transformer protein language model, trained on protein sequence data without label supervision. The model is pretrained on Uniref50 with an unsupervised masked language modeling (MLM) objective, meaning the model is trained to predict amino acids from the surrounding sequence context. This pretraining objective allows ESM-1b to learn generally useful features which can be transferred to downstream prediction tasks. ESM-1b has been evaluated on a variety of tasks related to protein structure and function, including remote homology detection, secondary structure prediction, contact prediction, and prediction of the effects of mutations on function, producing state-of-the-art results.

**Important note**: ESM-2 is now available in a range of checkpoint sizes. For most tasks, ESM-2 performance will be superior to ESM-1 and ESM-1b, and so we recommend using it instead unless your goal is explicitly to compare against ESM-1b. The ESM-2 checkpoint closest in size to ESM-1b is [esm2_t33_650M_UR50D](https://huggingface.co/facebook/esm2_t33_650M_UR50D).


## **Model description**

The ESM-1b model is based on the [RoBERTa](https://arxiv.org/abs/1907.11692) architecture and training procedure, using the Uniref50 2018_03 database of protein sequences. Note that the pretraining is on the raw protein sequences only. The training is purely unsupervised -- during training no labels are given related to structure or function.

Training is with the masked language modeling objective. The masking follows the procedure of [Devlin et al. 2019](https://arxiv.org/abs/1810.04805), randomly masking 15% of the amino acids in the input, and includes the pass-through and random token noise. One architecture difference from the RoBERTa model is that ESM-1b uses [pre-activation layer normalization](https://arxiv.org/abs/1603.05027).

The learned representations can be used as features for downstream tasks. For example if you have a dataset of measurements of protein activity you can fit a regression model on the features output by ESM-1b to predict the activity of new sequences. The model can also be fine-tuned.

ESM-1b can infer information about the structure and function of proteins without further supervision, i.e. it is capable of zero-shot transfer to structure and function prediction. [Rao et al. 2020](https://openreview.net/pdf?id=fylclEqgvgd) found that the attention heads of ESM-1b directly represent contacts in the 3d structure of the protein. [Meier et al. 2021](https://openreview.net/pdf?id=uXc42E9ZPFs) found that ESM-1b can be used to score the effect of sequence variations on protein function.


## **Intended uses & limitations**

The model can be used for feature extraction, fine-tuned on downstream tasks, or used directly to make inferences about the structure and function of protein sequences, like any other masked language model. For full examples, please see [our notebook on fine-tuning protein models](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/protein_language_modeling.ipynb)


## **Training data**

The ESM-1b model was pretrained on [Uniref50](https://www.uniprot.org/downloads) 2018-03, a dataset consisting of approximately 30 million protein sequences.


## **Training procedure**


### **Preprocessing**

The protein sequences are uppercased and tokenized using a single space and a vocabulary size of 21. The inputs of the model are then of the form:


```
<cls> Protein Sequence A
```


During training, sequences longer than 1023 tokens (without CLS) are randomly cropped to a length of 1023.

The details of the masking procedure for each sequence follow Devlin et al. 2019:



* 15% of the amino acids are masked.
* In 80% of the cases, the masked amino acids are replaced by `<mask>`.
* In 10% of the cases, the masked amino acids are replaced by a random amino acid (different) from the one they replace.
* In the 10% remaining cases, the masked amino acids are left as is.


### **Pretraining**

The model was trained on 128 NVIDIA v100 GPUs for 500K updates, using sequence length 1024 (131,072 tokens per batch). The optimizer used is Adam (betas=[0.9, 0.999]) with a learning rate of 1e-4, a weight decay of 0, learning rate warmup for 16k steps and inverse square root decay of the learning rate after.