# PRSNet-2: End-to-end genotype-to-phenotype prediction via hierarchical graph neural networks

## Project Overview

PRSNet-2 proposes a novel hierarchical graph neural network architecture, which first employs a multi-kernel aggregator to map high-dimensional genotypic features to gene-level representations. Next, it models gene-gene interactions through message-passing operations and uses an attention-based readout module to generate interpretable phenotypic predictions. We further introduce significance-guided regularization strategies to boost model's generalizability based on prior genetic associations.

## Project Structure

```
PRSNet-2/
├── scripts/                # Training scripts
│   ├── train.py          # Main training script
│   ├── train.sh          # Training command script
│   ├── get_data.ipynb    # Data processing notebook
│   └── environment.yml   # Environment dependencies
├── src/                    # Source code
│   ├── model.py          # PRSNet2 model definition
│   ├── trainer.py        # Trainer class
│   ├── dataset.py        # Dataset class
│   └── utils.py          # Utility functions
├── example_data/          # Example data
│   └── AD/               # Alzheimer's Disease dataset
└── README.md             # This file
```

## Environment Setup

### Create Conda Environment

```bash
conda env create -f scripts/environment.yml
conda activate PRSNet-2
```

### Dependencies
- PyTorch
- DGL (Deep Graph Library)
- NumPy
- SciPy
- Pandas

## Usage

### Basic Training

```bash
cd scripts
python train.py \
  --data_path ../example_data \
  --dataset AD \
  --seed 42 \
  --lr 1e-4 \
  --batch_size 512
```

### Parameter Description

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_path` | Required | Path to data directory |
| `--dataset` | Required | Dataset name |
| `--seed` | Required | Random seed |
| `--lr` | 1e-4 | Learning rate |
| `--batch_size` | 512 | Batch size |
| `--n_snp_kernels` | 16 | Number of SNP convolutional kernels |
| `--n_gnn_layers` | 1 | Number of GNN layers |
| `--sg_l1` | 1000 | L1 regularization coefficient |
| `--sg_dropout_init` | 0.9 | Initial dropout rate |
| `--sg_dropout_min` | 0.15 | Minimum dropout rate |

## Data Format

Input data should be organized as follows:

```
data_path/dataset_name/
├── X.npy              # Feature matrix (num_samples × num_SNPs)
├── Y.npy              # Label vector (num_samples,)
├── pvalues.npy        # SNP p-values (num_SNPs,)
└── gene2snps.pkl      # Gene-to-SNP mapping dictionary
```


## License
MIT License
## Contact
lihan@nankai.edu.cn