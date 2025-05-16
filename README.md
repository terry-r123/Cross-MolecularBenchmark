# COMET: A Comprehensive Cross-Molecular Benchmark for Language Model Evaluation and Tasks in Biological Sequence Understanding

This is the official codebase for the paper:
**COMET: A Comprehensive Cross-Molecular Benchmark for Language Model Evaluation and Tasks in Biological Sequence Understanding**

![COMET Overview](https://github.com/terry-r123/Multi-omicsBechmark/blob/main/fig_full_vertical_0513.png)

---

## 🔧 Prerequisites & Installation

**Key libraries:**

* torch==1.13.1+cu117
* transformers==4.38.1

```
git clone https://github.com/terry-r123/Multi-omicsBechmark.git
```

---

## 🧪 Tasks and Datasets

**Supported Task Categories:**

### 🧬 DNA Tasks

* Enhancer-Promoter Interaction
* Enhancer Activity
* Gene Expression

### 🧫 RNA Tasks

* APA Isoform
* Programmable RNA Switches
* RNA Secondary Structure
* siRNA Efficiency

### 🧬 Protein Tasks

* Thermostability
* EC
* Contact

### 🔗 Cross-Molecular Tasks

* DNA-Protein Folding
* CRISPR Off-Target Prediction
* RNA-Protein Interaction

**📁 Datasets:**
*To be released soon*

**📦 Pretrained Checkpoints:**
*To be released soon*

---

## 📂 Data Structure
The project’s data directory is organized as follows:

```
├── downstream/
│   ├── dna_tasks                      
│   ├── rna_tasks                
│   └── prot_tasks
│   └── ......                  
├── model/
│   ├── dnabert2                     
│   ├── ntv2      
│   ├── rnafm                 
│   └── rnalm
│   └── esm1b
│   └── esm2
│   └── ......       
├── scripts/
│   ├── single_molecule                    
│   ├── multi_molecule                  
│   └── cross_molecule
│   └── opensources               
└── README.md                                
```
---

## 🧠 Models

**Available models/embedders used in COMET:**
```
Common Biology Foundation Model: DNABERT2, NTv2, RnaFM, BEACON, ESM1b, ESM-2
Naive Model: CNN, Resnet, LSTM
Unify Biology Foundation Model: LucaOne   
```

### ⚙️ Model Settings

| Models | name | token | pos | length| 
| --- | --- | --- | ---| --- |
|DNABERT2| dnabert2 | single  | alibi| 1024| 
|NTv2 | ntv2 | single  | rope| 1024| 
|RNA-FM | rna-fm | single  | ape| 1024| 
|BEACON-B| rnalm | single | alibi | 1026 |
|ESM1b | esm1b | single  | ape| 1024| 
|ESM2 | esm-2 | single  | ape| 1024| 
---

# Results
## Results of the unpaired cross-molecular experiments
![image](https://github.com/user-attachments/assets/e6b0fe78-9eb6-42d9-b486-b1995aec6c18)
## Results of the native multi-molecular experiments
![image](https://github.com/user-attachments/assets/e79dc859-a429-4006-bd33-f4c8bee6993a)
## Results of the native multi-molecular experiments
![image](https://github.com/user-attachments/assets/81cb7b79-5022-4c64-bbd6-21a2c88409bf)



## 🚀 Usage

### 🔁 Finetuning

To evaluate on specific task, run the bash scripts under the `scripts/` folder. Take cross-molecule EC Task for example:

```
bash scripts/cross-molecule/ec.sh
```

## 📜 License

This codebase is released under the **Apache License 2.0**. See the LICENSE file for more details.

---

## 📖 Citation

If you find this repository useful for your research, please consider citing:

```
@misc{ren2024beacon,
      title={BEACON: Benchmark for Comprehensive RNA Tasks and Language Models}, 
      author={Yuchen Ren and Zhiyuan Chen and Lifeng Qiao and Hongtai Jing and Yuchen Cai and Sheng Xu and Peng Ye and Xinzhu Ma and Siqi Sun and Hongliang Yan and Dong Yuan and Wanli Ouyang and Xihui Liu},
      year={2024},
      eprint={2406.10391},
      archivePrefix={arXiv},
      primaryClass={q-bio.QM}
}
```

---

> 💡 For questions or suggestions, feel free to open an issue or pull request.

---
