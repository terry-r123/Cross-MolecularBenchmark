from transformers import EsmTokenizer
import torch

class LucaOneTokenizer(EsmTokenizer):
    def __init__(self, *args, sequence_type='nucleotide', **kwargs):
        # Call the parent class's __init__ method
        super().__init__(*args, **kwargs)
        
        # Set the sequence_type attribute
        self.sequence_type = sequence_type

    def __call__(self, text, **kwargs):
        # Tokenize as usual
        outputs = super().__call__(text, **kwargs)

        breakpoint()
        
        # Ensure input_ids are tensors
        input_ids = outputs['input_ids']

        # Convert input_ids to tensors if they aren't already
        if not isinstance(input_ids, torch.Tensor):
            if isinstance(input_ids[0], list):  # Batched input
                input_ids = [torch.tensor(seq, dtype=torch.long) for seq in input_ids]
            else:  # Single input
                input_ids = torch.tensor(input_ids, dtype=torch.long)
            outputs['input_ids'] = input_ids  # Update with tensor versions
        
        # Create token_type_ids based on self.sequence_type
        if self.sequence_type == 'protein':
            token_value = 1
        elif self.sequence_type == 'nucleotide':
            token_value = 0
        else:
            raise ValueError("sequence_type must be 'protein' or 'nucleotide'")
        
        # Generate token_type_ids
        if isinstance(input_ids, list):  # Batched input
            outputs['token_type_ids'] = [torch.full_like(seq, token_value, dtype=torch.long) for seq in input_ids]
        else:  # Single input
            outputs['token_type_ids'] = torch.full_like(input_ids, token_value, dtype=torch.long)
        
        return outputs
