# Welcome to : Predicting multi-functional cancer therapy peptides via geometric deep learning
Cancer therapy peptides (CTPs), as multifunctional peptides, possess the ability to target cancer cells or related biomarkers, exhibiting significant therapeutic potential. However, traditional experimental screening methods are time-consuming and labor-intensive. Computational models driven by artificial intelligence offer an effective solution to this challenge. In this study, we developed a geometric deep learning model that integrates both sequence and structural information for CTP prediction. We employed ESMfold to predict the 3D structure of the peptides, and a Graph Transformer was used to optimize the structural features. Additionally, ESM-2 language model  was utilized to extract semantic features from peptide sequences. The model further incorporates contrastive learning to enhance its ability to distinguish complex features. Our model demonstrated impressive performance in classification tasks and provided insights into potential functional sites by analyzing amino acids in high-attention regions. This study not only shows significant promise for CTP functional prediction but also introduces new perspectives for the application of bioinformatics in precision medicine.

This CTP prediction tool developed by teams from the University of Hong Kong and the Chinese University of Hong Kong (Shenzhen)

![The workflow of this study](https://github.com/GGCL7/Geo-CTP/blob/main/workflow.png)


# Dataset for this study
We provided our dataset and you can find them [Dataset](https://github.com/GGCL7/Geo-CTP/tree/main/Dataset)
# Source code
We provide the source code and you can find them [Code](https://github.com/GGCL7/Geo-CTP/tree/main/Code)
