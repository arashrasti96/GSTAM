## Official implementation of "GSTAM: Efficient Graph Distillation with Structural Attention-Matching", to be published as a conference paper at ECCV-DD 2024.

- Project Page: ??
- 
## Abstract
Graph distillation has emerged as a solution for reducing large graph datasets to smaller, more manageable, and informative ones. Existing methods primarily target node classification, involve computationally intensive processes, and fail to capture the true distribution of the full graph dataset. To address these issues, we introduce Graph Distillation with Structural Attention Matching (GSTAM), a novel method for condensing graph classification datasets. GSTAM leverages the attention maps of GNNs to distill structural information from the original dataset into synthetic graphs. The structural attention-matching mechanism exploits the areas of the input graph that GNNs prioritize for classification, effectively distilling such information into the synthetic graphs and improving overall distillation performance. Comprehensive experiments demonstrate GSTAM's superiority over existing methods, achieving 0.45% to 6.5% better performance in extreme condensation ratios, highlighting its potential use in advancing distillation for graph classification tasks.
<p align="center">
<img src="./img/GSTAM.png" width="600" height="400">
</p>
## File Tree
This folder contains all neccesary code files and supplemental material for the main paper.
.
├── main_attention.py         # Source Code for reproducing DataDAM results on behncmark datasets and IPCs
├── networks.py             # Defines all relevant network architectures, including cross-arch models
├── utils.py                # Defines all utility functions required for any task or ablation in main paper, inlcuding our attention module
├── requirements.txt        # Lists all related Python packages neccessary for reproducing our model results
├── Supplementary.pdf       # Supplementary pdf for our main paper -- DataDAM
└── README.md
