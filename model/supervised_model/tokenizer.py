from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split

def create_rna_tokenizer(**kwargs):
    """
    Creates a PreTrainedTokenizerFast for RNA sequences, tokenizing at the nucleotide level.
    """
    # Define the RNA vocabulary
    vocab = {
        "A": 0,   # Adenine
        "C": 1,   # Cytosine
        "G": 2,   # Guanine
        "T": 3,   # T
        "N": 4,   # Any nucleotide (unknown)
        "[UNK]": 5,
        "[PAD]": 6,
    }

    # Create a tokenizer using the Tokenizers library
    base_tokenizer = Tokenizer(WordLevel(vocab, unk_token='[UNK]'))
    base_tokenizer.pre_tokenizer = Split('', 'isolated')  # Split every character (nucleotide)

    # Create a PreTrainedTokenizerFast tokenizer with the RNA vocab
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=base_tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token=None,
        sep_token=None,
        bos_token=None,
        eos_token=None,
        **kwargs
    )

    return tokenizer
