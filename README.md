# TranSyn

## 1 Introduction

TranSyn is a deep transfer learning framework for predicting the functional impact of synonymous variants.

Although synonymous variants do not change encoded amino acids, they can affect multiple molecular mechanisms, including splicing, translation efficiency, and protein conformation, and may therefore contribute to disease. Accurate prediction of functional synonymous variants remains challenging, largely due to the limited availability of labeled pathogenic samples.

In contrast, noncoding and missense variants have relatively abundant annotations. However, existing methods rarely exploit transferable knowledge from these variants to improve synonymous variant prediction.

To address this limitation, we propose **TranSyn**, a transfer learning framework that integrates:

- deep representations derived from foundation models pretrained on noncoding and missense variants (source domains)
- biologically informed handcrafted features

Experimental results demonstrate that TranSyn achieves **state-of-the-art performance** in deleterious synonymous variant prediction. Comparative analyses confirm the effectiveness of transferring knowledge from both noncoding and missense variants, with consistent improvements across different training set sizes and particularly strong gains under **small-sample conditions**.

Ablation studies further show that deep representations provide complementary information beyond handcrafted features. Finally, we perform genome-wide inference on actionable genes and investigate potential mechanisms underlying synonymous variant pathogenicity.

![Figure1](https://github.com/user-attachments/assets/fc8d4f94-98d5-48e9-9a3f-b4365b9544f0)

---

## 2 Installation

### 2.1 Dataset and Features

1. Target feature files can be downloaded from figshare: 10.6084/m9.figshare.31526149.

**Note:** Source-domain features are very large (~100GB in total), please contact me if you want to download them.

2. Environment Setup

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate transyn
```

Run the model:

```bash
python main.py
```

## 3 Results

Key outputs are stored in the following directories:

| Content                     | Path                                                                        |
| --------------------------- | --------------------------------------------------------------------------- |
| Pretrained weights          | `./model_weight/pretrained/`                                                |
| Final trained model         | `./model_weight/model.pt`                                                   |
| Actionable gene predictions | `./result/actionable_inference_results/ACMG.actionable.genes.GRCh38.vcf.gz` |

## 4 Citation

```citaion
The manuscript describing TranSyn is currently in preparation.
Citation information will be provided after publication.
```

## 5 Contact

For questions or issues, please feel free to open an `Issue` in this repository.
