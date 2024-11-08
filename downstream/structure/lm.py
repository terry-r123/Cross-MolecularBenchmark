import warnings
warnings.filterwarnings("ignore")
from transformers import EsmTokenizer, EsmModel, BertForMaskedLM, BertModel, AutoConfig, BertTokenizer

from transformers import Trainer, TrainingArguments, BertTokenizer, AutoModel
import transformers
from peft import LoraConfig, get_peft_model
import sys
import os
current_path = os.path.dirname(os.path.abspath(__file__))
parent_dir = (os.path.dirname(os.path.dirname(current_path)))
print(parent_dir)
sys.path.append(parent_dir)

from model.rnalm.rnalm_config import RnaLmConfig
from model.rnalm.modeling_rnalm import RnaLmModel 
from model.rnalm.rnalm_tokenizer import RnaLmTokenizer
from model.rnafm.modeling_rnafm import RnaFmModel
from model.rnabert.modeling_rnabert import RnaBertModel
from model.rnamsm.modeling_rnamsm import RnaMsmModel
from model.splicebert.modeling_splicebert import SpliceBertModel
from model.utrbert.modeling_utrbert import UtrBertModel
from model.utrlm.modeling_utrlm import UtrLmModel
from model.ntv2.modeling_esm import EsmModel as NTv2Model
from model.esm.modeling_esm import EsmModel
from model.dnabert2.bert_layers import BertModel as DNABERT2Model
from tokenizer.tokenization_opensource import OpenRnaLMTokenizer
from model.lucagplm.v2_0.modeling_gplm import LucaOneModel
def get_extractor(args):
    '''
    '''

    # the pretrained extractor names
    name_dict = {'8m': 'esm8m_2parts_5m',
                 '35m': 'esm35m_25parts_31m',
                 '150m': 'esm150m_25parts_31m',
                 '650m': 'esm650m_50parts_100m',
                 '650m-1B': 'esm650m-1B_8clstr_8192',
                 '8m-1B' : 'esm8m_1B',
                 '35m-1B' : 'esm35m_1B',
                 '150m-1B': 'esm150m_1B'
                 }
    if args.model_type == 'rnalm':
        if args.token_type != 'single':
            tokenizer = EsmTokenizer.from_pretrained(
                args.model_name_or_path,
                cache_dir=args.cache_dir,
                model_max_length=args.model_max_length,
                padding_side="right",
                use_fast=True,
                trust_remote_code=True,
                token_type=args.token_type
                )
            #RnaLmTokenizer = EsmTokenizer
        else:
            #RnaLmTokenizer = RnaLmTokenizer
            tokenizer = RnaLmTokenizer.from_pretrained(
                args.model_name_or_path,
                cache_dir=args.cache_dir,
                model_max_length=args.model_max_length,
                padding_side="right",
                use_fast=True,
                trust_remote_code=True,
                token_type=args.token_type
                )
        # print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        if args.train_from_scratch:
            print(f'Train from scratch {args.model_type} model')
            config = RnaLmConfig.from_pretrained(args.model_name_or_path,
            attn_implementation=args.attn_implementation,)
            extractor = RnaLmModel(config)
        else:        
            # import  
            # with open() 
            extractor = RnaLmModel.from_pretrained(
                args.model_name_or_path,
            )
    if args.model_type == 'BEACON-B':
        if args.token_type != 'single':
            tokenizer = EsmTokenizer.from_pretrained(
                args.model_name_or_path,
                cache_dir=args.cache_dir,
                model_max_length=args.model_max_length,
                padding_side="right",
                use_fast=True,
                trust_remote_code=True,
                token_type=args.token_type
                )
            #RnaLmTokenizer = EsmTokenizer
        else:
            #RnaLmTokenizer = RnaLmTokenizer
            tokenizer = RnaLmTokenizer.from_pretrained(
                args.model_name_or_path,
                cache_dir=args.cache_dir,
                model_max_length=args.model_max_length,
                padding_side="right",
                use_fast=True,
                trust_remote_code=True,
                token_type=args.token_type
                )
        # print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        if args.train_from_scratch:
            print(f'Train from scratch {args.model_type} model')
            config = RnaLmConfig.from_pretrained(args.model_name_or_path,
            attn_implementation=args.attn_implementation,)
            extractor = RnaLmModel(config)
        else:
            print(f'args.model_name_or_path: {args.model_name_or_path}')    
            extractor = RnaLmModel.from_pretrained(
                args.model_name_or_path,
                attn_implementation=args.attn_implementation,
            )
    # elif args.model_type == 'rna-fm' or args.model_type == 'esm-rna':
    #     #assert args.model_scale in name_dict.keys(), print(f'args.model_scale should be in {name_dict.keys()}')

    #     #extractor = EsmModel.from_pretrained(f'{args.pretrained_lm_dir}/{name_dict[args.model_scale]}/')
    #     #tokenizer = EsmTokenizer.from_pretrained("/mnt/data/ai4bio/renyuchen/DNABERT/examples/rna_finetune/ssp/vocab_esm_mars.txt")
    #     tokenizer = transformers.AutoTokenizer.from_pretrained(
    #         args.model_name_or_path,
    #         cache_dir=args.cache_dir,
    #         model_max_length=args.model_max_length,
    #         padding_side="right",
    #         use_fast=True,
    #         trust_remote_code=True,
    #     )
    #     if args.train_from_scratch:
    #         print('Loading esm model')
    #         print('Train from scratch')
    #         config = AutoConfig.from_pretrained(args.model_name_or_path)
    #         extractor = EsmModel(config)
    #     else:           
    #         extractor = EsmModel.from_pretrained(args.model_name_or_path)
        
        
        #tokenizer = EsmTokenizer.from_pretrained(f"{args.pretrained_lm_dir}/vocab_esm_mars.txt")
    elif args.model_type == 'esm-protein':
        extractor = EsmModel.from_pretrained(args.model_name_or_path)

        tokenizer = EsmTokenizer.from_pretrained(f"{args.model_name_or_path}/vocab.txt")
    elif args.model_type == 'dnabert':
        
        #print(extractor)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            model_max_length=args.model_max_length,
            padding_side="right",
            use_fast=True,
            # trust_remote_code=True,
            )
        if args.train_from_scratch:
            print('Loading dnabert model')
            print('Train from scratch')
            config = MMoeBertConfig.from_pretrained(args.model_name_or_path)
            extractor = BertModel(config)
        else:
            extractor = BertModel.from_pretrained(
            args.model_name_or_path,
            )
    elif args.model_type == 'dnabert2':
        
        #print(extractor)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            model_max_length=args.model_max_length,
            padding_side="right",
            use_fast=True,
            # trust_remote_code=True,
            )
        if args.train_from_scratch:
            print('Loading dnabert model')
            print('Train from scratch')
            config = MMoeBertConfig.from_pretrained(args.model_name_or_path)
            extractor = DNABERT2(config)
        else:
            extractor = DNABERT2Model.from_pretrained(
            args.model_name_or_path,
            )    
    elif args.model_type in ['esm2', 'esm-1b', 'rna-fm','rnabert','rnamsm','splicebert-human510','splicebert-ms510','splicebert-ms1024','utrbert-3mer','utrbert-4mer','utrbert-5mer','utrbert-6mer','utr-lm-mrl','utr-lm-te-el']:
        tokenizer = OpenRnaLMTokenizer.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            model_max_length=args.model_max_length,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True,
        )
        if args.model_type == 'rna-fm':      
            print(args.model_type)
            print(f'Loading {args.model_type} model')
            extractor = RnaFmModel.from_pretrained(
                args.model_name_or_path,
                cache_dir=args.cache_dir,
                trust_remote_code=True,
 
            )     
        elif args.model_type == 'rnabert':      
            print(args.model_type)
            print(f'Loading {args.model_type} model')
            extractor = RnaBertModel.from_pretrained(
                args.model_name_or_path,
                cache_dir=args.cache_dir,
                trust_remote_code=True,

            )     
        elif args.model_type == 'rnamsm':
            print(args.model_type)
            print(f'Loading {args.model_type} model')
            extractor = RnaMsmModel.from_pretrained(
                args.model_name_or_path,
                cache_dir=args.cache_dir,
                trust_remote_code=True,
            )        
        elif 'splicebert' in args.model_type:
            print(args.model_type)
            print(f'Loading {args.model_type} model')
            extractor = SpliceBertModel.from_pretrained(
                args.model_name_or_path,
                cache_dir=args.cache_dir,
                trust_remote_code=True,

            )       
        elif 'utrbert' in args.model_type:
            print(args.model_type)
            print(f'Loading {args.model_type} model')
            extractor = UtrBertModel.from_pretrained(
                args.model_name_or_path,
                cache_dir=args.cache_dir,
                trust_remote_code=True,
            )  
        elif 'utr-lm' in args.model_type:
            print(args.model_type)
            print(f'Loading {args.model_type} model')
            extractor = UtrLmModel.from_pretrained(
                args.model_name_or_path,
                cache_dir=args.cache_dir,
                trust_remote_code=True,
            )  
        elif 'esm2' in args.model_type:
            print(args.model_type)
            print(f'Loading protein model {args.model_type} model')
            extractor = EsmModel.from_pretrained(
                args.model_name_or_path,
                cache_dir=args.cache_dir,
                trust_remote_code=True,
                ignore_mismatched_sizes=True, # change config for RNA downtasks
                pad_token_id=tokenizer.pad_token_id,
                vocab_size=tokenizer.vocab_size,
            )

            # Access the embedding layer
            embedding_layer = extractor.embeddings
            # Initialize the embedding layer with random values
            # torch.nn.init.normal_(embedding_layer.weight, mean=0.0, std=1.0)
            for embedding_layer in [extractor.embeddings.word_embeddings, extractor.embeddings.position_embeddings]:
                print(embedding_layer)
                import torch
                if isinstance(embedding_layer, torch.nn.Embedding):
                    print(f"init embedding_layer: {embedding_layer}")
                    embedding_layer.weight.data.normal_(mean=0.0, std=extractor.config.initializer_range)
                    if embedding_layer.padding_idx is not None:
                        embedding_layer.weight.data[embedding_layer.padding_idx].zero_()
        elif 'esm-1b' in args.model_type:
            print(args.model_type)
            print(f'Loading protein model {args.model_type} model')
            extractor = AutoModel.from_pretrained(
                args.model_name_or_path,
                cache_dir=args.cache_dir,
                trust_remote_code=True,
                ignore_mismatched_sizes=True, # change config for RNA downtasks
                pad_token_id=tokenizer.pad_token_id,
                vocab_size=tokenizer.vocab_size,
            )

            # Access the embedding layer
            embedding_layer = extractor.embeddings
            # Initialize the embedding layer with random values
            # torch.nn.init.normal_(embedding_layer.weight, mean=0.0, std=1.0)
            for embedding_layer in [extractor.embeddings.word_embeddings, extractor.embeddings.position_embeddings]:
                print(embedding_layer)
                import torch
                if isinstance(embedding_layer, torch.nn.Embedding):
                    print(f"init embedding_layer: {embedding_layer}")
                    embedding_layer.weight.data.normal_(mean=0.0, std=extractor.config.initializer_range)
                    if embedding_layer.padding_idx is not None:
                        embedding_layer.weight.data[embedding_layer.padding_idx].zero_()

    elif args.model_type == 'ntv2':
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            model_max_length=args.model_max_length,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True,
            )
        print(f'Loading {args.model_type} model')
        extractor = NTv2Model.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
        )
    elif args.model_type == 'LucaOne':
        # tokenizer = AlphabetV2_0.from_predefined("gene_prot")
        # model_config: PretrainedConfig = LucaOneConfig.from_json_file(os.path.join(model_dirpath, "config.json"))
        tokenizer = EsmTokenizer.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            model_max_length=args.model_max_length,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True,
            token_type=args.token_type
        )
        extractor = LucaOneModel.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
        )
        if not args.is_freeze:
            lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=list(args.lora_target_modules.split(",")),
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="SEQ_CLS",
                inference_mode=False,
            )
            extractor = get_peft_model(extractor, lora_config)
            extractor.print_trainable_parameters()

    return extractor, tokenizer


def unitest(args):

    extractor, tokenizer = get_extractor(args)

    # replace 'U' with 'T'
    seqs = ['ATGCATGCATGCATGCATGC']

    max_len = 128

    data_dict = tokenizer.batch_encode_plus(seqs,
                                            padding='max_length',
                                            max_length=max_len,
                                            truncation=True,
                                            return_tensors='pt')

    input_ids, attention_mask = data_dict['input_ids'], data_dict['attention_mask']

    output = extractor(input_ids=input_ids, attention_mask=attention_mask)

    #print(output.keys())

    #print(output['last_hidden_state'].shape)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrained_lm_dir", type=str, default='/public/home/taoshen/data/rna/mars_fm_data/mars_esm_preckpts')

    parser.add_argument("--model_scale", type=str, default='8m')

    args = parser.parse_args()

    unitest(args)