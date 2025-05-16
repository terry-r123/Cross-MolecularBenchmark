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

*Details coming soon (e.g., how data folders are organized, input/output formats, etc.)*

---

## 🧠 Models

**Available models/embedders used in COMET:**
*Details coming soon*

### ⚙️ Model Settings

*More configuration details to be added here.*

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

To evaluate on all tasks, run the bash scripts under the `scripts/` folder. For example:

```
bash scripts/run_all_tasks.sh
```

### 🧬 Embedding Extraction

To extract embeddings from a dummy RNA sequence:

```
python embed.py --input sample_rna.fasta --model comet_rna
```

*Update this section with the exact script and argument details.*

---

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
