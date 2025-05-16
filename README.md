# COMET: A Comprehensive Cross-Molecular Benchmark for  Language Model Evaluation and Tasks in Biological Sequence Understanding
# multi-omic
This is the official codebase of the paper

![BEACON 总体图](https://github.com/terry-r123/Multi-omicsBechmark/blob/main/fig_full_vertical_0513.png)

# Prerequisites
## Installation

important libs: torch==1.13.1+cu117, transformers==4.38.1

```bash
git clone https://github.com/terry-r123/Multi-omicsBechmark.git
```

# Tasks and Datasets
The full list of current task names are :  
Datasets of Cross-Molecular tasks can be found in []  
Model checkpoints of opensource RNA language models and COMET can be found in []  
## Data Structure
[]  
The full list of current task names are :  
***DNA TASKS***  
·Enhancer Promter Interaction  
·Enhancer Activity  
·Gene Expression  
***RNA TASKS***  
·APA Isoform  
·Programmable RNA Switches  
·Secondary Sturcture  
·siRNA Efficiency  
***Cross-Molecular TASKS***  
·DNA-Protein Folding  
·CRISPER OFF Target  
·RNA-Protein Interaction  

### Results of the unpaired cross-molecular experiments
![image](https://github.com/user-attachments/assets/e6b0fe78-9eb6-42d9-b486-b1995aec6c18)
### Results of the native multi-molecular experiments
![image](https://github.com/user-attachments/assets/e79dc859-a429-4006-bd33-f4c8bee6993a)
### Results of the native multi-molecular experiments
![image](https://github.com/user-attachments/assets/81cb7b79-5022-4c64-bbd6-21a2c88409bf)

And the list of available embedders/models used for training on the tasks are:  
·DNABERT2  
·NTv2
·RNA-FM  
·BEACON-B  
·ESM-1b  
·ESM-2  
·LucaOne  
## Model settings
[]  

# Usage
## Finetuning
To evalute on all tasks, you can run the bash scripts in the scripts folder, for example:
```bash

```
## Computing embeddings
Embeddings from a dummy RNA sequence can be used as follows:
```bash
```

# License
This codebase is released under the Apache License 2.0 as in the LICENSE file.

# Citation

If you find this repo useful for your research, please consider citing the paper
```
@misc{ren2024beacon,
      title={BEACON: Benchmark for Comprehensive RNA Tasks and Language Models}, 
      author={Yuchen Ren and Zhiyuan Chen and Lifeng Qiao and Hongtai Jing and Yuchen Cai and Sheng Xu and Peng Ye and Xinzhu Ma and Siqi Sun and Hongliang Yan and Dong Yuan and Wanli Ouyang and Xihui Liu},
      year={2024},
      eprint={2406.10391},
      archivePrefix={arXiv},
      primaryClass={id='q-bio.QM' full_name='Quantitative Methods' is_active=True alt_name=None in_archive='q-bio' is_general=False description='All experimental, numerical, statistical and mathematical contributions of value to biology'}
}
```






