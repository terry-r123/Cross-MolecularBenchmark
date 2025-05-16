from transformers import EsmTokenizer
import torch

# Define the updated mapping dictionary
nucleotide_to_number_mapping = {
    'A': '1', 'a': '1',
    'U': '2', 'u': '2',
    'T': '2', 't': '2',
    'C': '3', 'c': '3',
    'G': '4', 'g': '4',
}
class LucaOneTokenizer(EsmTokenizer):
    def __init__(self, *args, sequence_type='nucleotide', **kwargs):
        # Call the parent class's __init__ method
        super().__init__(*args, **kwargs)
        
        # Set the sequence_type attribute
        self.sequence_type = sequence_type

    def __call__(self, text, **kwargs):
        if self.sequence_type == 'nucleotide':
            # Check the type of input_ids to match token_type_ids accordingly
            if isinstance(text, (list, tuple)):
                # # input_ids is a list
                # if isinstance(input_ids[0], list):
                #     # Batched input
                #     token_type_ids = [[token_value] * len(seq) for seq in input_ids]
                # else:
                #     # Single input
                #     token_type_ids = [token_value] * len(input_ids)
                text = [self.gene_seq_replace(seq) for seq in text]
            elif isinstance(text, str):
                text = self.gene_seq_replace(text)
            else:
                raise TypeError("text must be `str`, `List[str]`, `List[List[str]]`")
        # Tokenize as usual
        outputs = super().__call__(text, **kwargs)
        
        # Ensure input_ids are tensors
        input_ids = outputs['input_ids']

        # # Convert input_ids to tensors if they aren't already
        # if not isinstance(input_ids, torch.Tensor):
        #     if isinstance(input_ids[0], list):  # Batched input
        #         input_ids = [torch.tensor(seq, dtype=torch.long) for seq in input_ids]
        #     else:  # Single input
        #         input_ids = torch.tensor(input_ids, dtype=torch.long)
        #     outputs['input_ids'] = input_ids  # Update with tensor versions
        
        # Create token_type_ids based on self.sequence_type
        if self.sequence_type == 'protein':
            token_value = 1
        elif self.sequence_type == 'nucleotide':
            token_value = 0
        else:
            raise ValueError("sequence_type must be 'protein' or 'nucleotide'")
        
        # Check the type of input_ids to match token_type_ids accordingly
        if isinstance(input_ids, (list, tuple)):
            # input_ids is a list
            if isinstance(input_ids[0], list):
                # Batched input
                token_type_ids = [[token_value] * len(seq) for seq in input_ids]
            else:
                # Single input
                token_type_ids = [token_value] * len(input_ids)
        else:
            # input_ids is a tensor
            token_type_ids = torch.full_like(input_ids, token_value, dtype=torch.long)

        outputs['token_type_ids'] = token_type_ids

        # # Generate token_type_ids
        # if isinstance(input_ids[0], list):  # Batched input
        #     outputs['token_type_ids'] = [[token_value] * len(seq) for seq in input_ids]
        # else:  # Single input
        #     outputs['token_type_ids'] = torch.full_like(input_ids, token_value, dtype=torch.long)
        
        return outputs

    def gene_seq_replace(self, seq):
        '''
        Nucleic acid gene replace: A->1, U/T->2, C->3, G->4, N->5
        :param seq:
        :return:
        '''
        new_seq = ''
        for ch in seq:
            if ch in nucleotide_to_number_mapping:
                new_seq += nucleotide_to_number_mapping[ch]
            else:
                new_seq += '5'  # Unknown character
        return new_seq
