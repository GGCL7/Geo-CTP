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
## Requirements

- Python 3.10+
- PyTorch
- PyTorch Geometric
- `torch_scatter`
- `transformers`
- `biopython`
- `scikit-learn`
- `numpy`

## Stage1 Training

`Stage1/train.py` is a command-line script for binary classification.

Example:

```bash
python3 Stage1/train.py \
  --train-fa Dataset/1.0/train.txt \
  --test-fa Dataset/1.0/test.txt \
  --pdb-root data/pdb \
  --esm-model-path pretrained/esm_model \
  --epochs 200 \
  --batch-size 32 \
  --num-layers 2 \
  --num-heads 8 \
  --hidden-channels 128 \
  --out-channels 64 \
  --projection-dim 128 \
  --dropout 0.2 \
  --learning-rate 1e-4 \
  --temperature 0.2 \
  --weight-seqstr 0.6 \
  --weight-label 0.4 \
  --save-path checkpoints/stage1_best_model.pth
```

Main arguments:

- `--train-fa`, `--test-fa`: train/test sequence files
- `--pdb-root`: directory of residue-level PDB files
- `--esm-model-path`: ESM model directory or Hugging Face model id
- `--esm-device`: device for sequence embedding extraction
- `--train-device`: device for GeoCTP training
- `--distance-threshold`: graph edge cutoff distance
- `--num-layers`, `--num-heads`, `--hidden-channels`, `--out-channels`
- `--projection-dim`, `--dropout`
- `--learning-rate`, `--batch-size`, `--epochs`
- `--temperature`, `--weight-seqstr`, `--weight-label`
- `--save-path`

## Stage2 Training

`Stage2/train.py` is a command-line script for multi-label classification.

Example:

```bash
python3 Stage2/train.py \
  --train-dir Dataset/Stage2/train_test/train \
  --test-dir Dataset/Stage2/train_test/test \
  --pdb-root data/pdb \
  --label-file Dataset/Stage2/multi_label_sequences.txt \
  --esm-model-path pretrained/esm_model \
  --epochs 100 \
  --batch-size 64 \
  --num-layers 2 \
  --num-heads 4 \
  --hidden-channels 128 \
  --out-channels 64 \
  --projection-dim 128 \
  --dropout 0.3 \
  --learning-rate 1e-4 \
  --contrastive-temperature 0.1 \
  --contrastive-weight 1.0 \
  --focal-gamma 2.0 \
  --save-path checkpoints/stage2_best_model.pth
```

Main arguments:

- `--train-dir`, `--test-dir`: train/test multi-file fasta directories
- `--label-file`: sequence-to-label mapping file
- `--pdb-root`: directory of residue-level PDB files
- `--esm-model-path`: ESM model directory or Hugging Face model id
- `--esm-device`: device for sequence embedding extraction
- `--train-device`: device for GeoCTP training
- `--distance-threshold`: graph edge cutoff distance
- `--num-layers`, `--num-heads`, `--hidden-channels`, `--out-channels`
- `--projection-dim`, `--dropout`
- `--learning-rate`, `--batch-size`, `--epochs`
- `--contrastive-temperature`, `--contrastive-weight`, `--focal-gamma`
- `--save-path`



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
