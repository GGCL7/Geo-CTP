# Welcome to GeoCTP: Structure-aware prediction of multifunctional cancer therapy peptides via graph transformer and contrastive learning
Cancer therapy peptides (CTPs), as multifunctional peptides, possess the ability to target cancer cells or related biomarkers, exhibiting significant therapeutic potential. However, traditional experimental screening methods are time-consuming and labor-intensive. Computational models driven by artificial intelligence offer an effective solution to this challenge. In this study, we developed a geometric deep learning model that integrates both sequence and structural information for CTP prediction. We employed ESMfold to predict the 3D structure of the peptides, and a Graph Transformer was used to optimize the structural features. Additionally, ESM-2 language model  was utilized to extract semantic features from peptide sequences. The model further incorporates contrastive learning to enhance its ability to distinguish complex features. Our model demonstrated impressive performance in classification tasks and provided insights into potential functional sites by analyzing amino acids in high-attention regions. This study not only shows significant promise for CTP functional prediction but also introduces new perspectives for the application of bioinformatics in precision medicine.



![The workflow of this study](https://github.com/GGCL7/Geo-CTP/blob/main/workflow.png)


# Dataset for this study
We provided our dataset and you can find them [Dataset](https://github.com/GGCL7/Geo-CTP/tree/main/Dataset)
# Source code
We provide the source code and you can find them [Code](https://github.com/GGCL7/Geo-CTP/tree/main/Code)

# Predict 3D protein structure using ESMfold
To predict the 3D structure of a protein using ESMfold, you can call the following API:
```bash
https://api.esmatlas.com/foldSequence/v1/pdb/
```
# ESM-2 Language model embeddings
```bash
https://huggingface.co/docs/transformers/en/model_doc/esm
```
# Predicting CTPs and their functionalities

## Stage 1: Predict whether a peptide sequence is a CTP

To predict whether a given peptide sequence is a Cancer Therapy Peptide (CTP), use the following command:

```bash
python predict_stage1.py -i "KWKSFLKTFKSAKKTVAHTAAKAISS" -p example.pdb
```
Output example
```bash
Predicted Class: CTP
```
## Stage 2: Predict the functionalities of the CTP
If the peptide is predicted as a CTP, use the following command to predict its six functionalities:

```bash
python predict_stage2.py -i "KWKSFLKTFKSAKKTVAHTAAKAISS" -p example.pdb
```
Output example

```bash
Prediction Results:
Tumor Active Peptide: 0.9758
Cancer Targeted Peptides: 0.0031
Membrane Targeted: 0.9683
Cell Penetrating Peptides: 0.0210
Membrane Lysis: 0.0000
Induce Apoptosis: 0.0001
```
## 📄 Citations
If you use ESM-2 Language model in your work, please cite this paper:
```bash
@article{lin2023evolutionary,
  title={Evolutionary-scale prediction of atomic-level protein structure with a language model},
  author={Lin, Zeming and Akin, Halil and Rao, Roshan and Hie, Brian and Zhu, Zhongkai and Lu, Wenting and Smetanin, Nikita and Verkuil, Robert and Kabeli, Ori and Shmueli, Yaniv and others},
  journal={Science},
  volume={379},
  number={6637},
  pages={1123--1130},
  year={2023},
  publisher={American Association for the Advancement of Science}
}
```
