---
license: cc-by-nc-sa-4.0
widget:
- text: ACCTGA<mask>TTCTGAGTC
tags:
- DNA
- biology
- genomics
datasets:
- InstaDeepAI/multi_species_genome
- InstaDeepAI/nucleotide_transformer_downstream_tasks
---
# nucleotide-transformer-v2-100m-multi-species

The Nucleotide Transformers are a collection of foundational language models that were pre-trained on DNA sequences from whole-genomes. Compared to other approaches, our models do not only integrate information from single reference genomes, but leverage DNA sequences from over 3,200 diverse human genomes, as well as 850 genomes from a wide range of species, including model and non-model organisms. Through robust and extensive evaluation, we show that these large models provide extremely accurate molecular phenotype prediction compared to existing methods

Part of this collection is the **nucleotide-transformer-v2-100m-multi-species**, a 100m parameters transformer pre-trained  on a collection of 850 genomes from a wide range of species, including model and non-model organisms.

**Developed by:** InstaDeep, NVIDIA and TUM

### Model Sources

<!-- Provide the basic links for the model. -->

- **Repository:** [Nucleotide Transformer](https://github.com/instadeepai/nucleotide-transformer)
- **Paper:** [The Nucleotide Transformer: Building and Evaluating Robust Foundation Models for Human Genomics](https://www.biorxiv.org/content/10.1101/2023.01.11.523679v1) 

### How to use

<!-- Need to adapt this section to our model. Need to figure out how to load the models from huggingface and do inference on them -->
Until its next release, the `transformers` library needs to be installed from source with the following command in order to use the models:
```bash
pip install --upgrade git+https://github.com/huggingface/transformers.git
```

A small snippet of code is given here in order to retrieve both logits and embeddings from a dummy DNA sequence.
```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

# Import the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-100m-multi-species", trust_remote_code=True)
model = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-100m-multi-species", trust_remote_code=True)

# Choose the length to which the input sequences are padded. By default, the 
# model max length is chosen, but feel free to decrease it as the time taken to 
# obtain the embeddings increases significantly with it.
max_length = tokenizer.model_max_length

# Create a dummy dna sequence and tokenize it
sequences = ["ATTCCGATTCCGATTCCG", "ATTTCTCTCTCTCTCTGAGATCGATCGATCGAT"]
tokens_ids = tokenizer.batch_encode_plus(sequences, return_tensors="pt", padding="max_length", max_length = max_length)["input_ids"]

# Compute the embeddings
attention_mask = tokens_ids != tokenizer.pad_token_id
torch_outs = model(
    tokens_ids,
    attention_mask=attention_mask,
    encoder_attention_mask=attention_mask,
    output_hidden_states=True
)

# Compute sequences embeddings
embeddings = torch_outs['hidden_states'][-1].detach().numpy()
print(f"Embeddings shape: {embeddings.shape}")
print(f"Embeddings per token: {embeddings}")

# Add embed dimension axis
attention_mask = torch.unsqueeze(attention_mask, dim=-1)

# Compute mean embeddings per sequence
mean_sequence_embeddings = torch.sum(attention_mask*embeddings, axis=-2)/torch.sum(attention_mask, axis=1)
print(f"Mean sequence embeddings: {mean_sequence_embeddings}")
```


## Training data

The **nucleotide-transformer-v2-100m-multi-species** model was pretrained on a total of 850 genomes downloaded from [NCBI](https://www.ncbi.nlm.nih.gov/). Plants and viruses are not included in these genomes, as their regulatory elements differ from those of interest in the paper's tasks. Some heavily studied model organisms were picked to be included in the collection of genomes, which represents a total of 174B nucleotides, i.e roughly 29B tokens. The data has been released as a HuggingFace dataset [here](https://huggingface.co/datasets/InstaDeepAI/multi_species_genomes).

## Training procedure

### Preprocessing

The DNA sequences are tokenized using the Nucleotide Transformer Tokenizer, which tokenizes sequences as 6-mers tokenizer when possible, otherwise tokenizing each nucleotide separately as described in the [Tokenization](https://github.com/instadeepai/nucleotide-transformer#tokenization-abc) section of the associated repository. This tokenizer has a vocabulary size of 4105. The inputs of the model are then of the form:

```
<CLS> <ACGTGT> <ACGTGC> <ACGGAC> <GACTAG> <TCAGCA>
```

The tokenized sequence have a maximum length of 1,000.

The masking procedure used is the standard one for Bert-style training:
- 15% of the tokens are masked.
- In 80% of the cases, the masked tokens are replaced by `[MASK]`.
- In 10% of the cases, the masked tokens are replaced by a random token (different) from the one they replace.
- In the 10% remaining cases, the masked tokens are left as is.

### Pretraining

The model was trained with 8 A100 80GB on 300B tokens, with an effective batch size of 1M tokens. The sequence length used was 1000 tokens. The Adam optimizer [38] was used with a learning rate schedule, and standard values for exponential decay rates and epsilon constants, β1 = 0.9, β2 = 0.999 and ε=1e-8. During a first warmup period, the learning rate was increased linearly between 5e-5 and 1e-4 over 16k steps before decreasing following a square root decay until the end of training.

### Architecture

The model belongs to the second generation of nucleotide transformers, with the changes in architecture consisting the use of rotary positional embeddings instead of learned ones, as well as the introduction of Gated Linear Units.

### BibTeX entry and citation info

```bibtex
@article{dalla2023nucleotide,
  title={The Nucleotide Transformer: Building and Evaluating Robust Foundation Models for Human Genomics},
  author={Dalla-Torre, Hugo and Gonzalez, Liam and Mendoza Revilla, Javier and Lopez Carranza, Nicolas and Henryk Grywaczewski, Adam and Oteri, Francesco and Dallago, Christian and Trop, Evan and Sirelkhatim, Hassan and Richard, Guillaume and others},
  journal={bioRxiv},
  pages={2023--01},
  year={2023},
  publisher={Cold Spring Harbor Laboratory}
}
```