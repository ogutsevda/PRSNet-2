import sys

sys.path.append("/mnt/lts4-pathofm/scratch/home/ogut/genomics-project/prsnet2/PRSNet-2")
import argparse
import warnings
import torch
from torch.utils.data import DataLoader
import dgl
import pandas as pd
import numpy as np
import scipy.sparse as sp
import json
import os

warnings.filterwarnings("ignore")
import pickle as pkl
from src.dataset import Dataset
from src.utils import generate_splits, seed_worker, collate_fn, set_random_seed
from src.model import PRSNet2
from src.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for training PRSNet-2")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/mnt/lts4-pathofm/scratch/home/ogut/genomics-project/prsnet2/PRSNet-2/example_data",
    )
    parser.add_argument("--dataset", type=str, default="AD")

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=512)

    parser.add_argument("--n_snp_kernels", type=int, default=16)
    parser.add_argument("--n_gnn_layers", type=int, default=1)

    parser.add_argument("--sg_l1", type=float, default=1000)
    parser.add_argument("--sg_dropout_init", type=float, default=0.9)
    parser.add_argument("--sg_dropout_min", type=float, default=0.15)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    set_random_seed(args.seed)

    # Data Loading
    ggi_graph = dgl.load_graphs(
        f"/mnt/lts4-pathofm/scratch/home/ogut/genomics-project/prsnet2/PRSNet-2/example_data/ggi_graph_800.bin"
    )[0][0]
    X = torch.tensor(np.load(f"{args.data_path}/{args.dataset}/X.npy"))
    Y = torch.tensor(np.load(f"{args.data_path}/{args.dataset}/Y.npy"))
    pvalues = torch.from_numpy(
        np.load(f"{args.data_path}/{args.dataset}/pvalues.npy").astype(np.float32)
    ).to(device)
    with open(f"{args.data_path}/{args.dataset}/gene2snps.pkl", "rb") as f:
        gene2snps = pkl.load(f)
        gene2snps = dict(sorted(gene2snps.items(), key=lambda x: x[0]))
    gene2snp_len = torch.LongTensor([len(gene2snps[gene]) for gene in gene2snps]).to(
        device
    )
    snp_ids = sum(gene2snps.values(), [])
    print(f"Number of snps: {len(snp_ids)}")
    gene_ids = list(gene2snps.keys())
    print(f"Number of genes: {len(gene_ids)}")
    ggi_graph = dgl.node_subgraph(ggi_graph, gene_ids)

    train_indices, val_indices, test_indices = generate_splits(Y, seed=args.seed)

    train_dataset = Dataset(X[train_indices], Y[train_indices], balanced_sampling=True)
    val_dataset = Dataset(X[val_indices], Y[val_indices])
    test_dataset = Dataset(X[test_indices], Y[test_indices])

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        worker_init_fn=seed_worker,
        drop_last=True,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        worker_init_fn=seed_worker,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        worker_init_fn=seed_worker,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    model = PRSNet2(
        n_snps=len(pvalues),
        n_genes=len(gene_ids),
        gene_ids=gene_ids,
        gene2snp_len=gene2snp_len,
        snp_ids=snp_ids,
        pvalues=pvalues,
        n_gnn_layers=args.n_gnn_layers,
        n_snp_kernels=args.n_snp_kernels,
        sg_dropout_init=args.sg_dropout_init,
        sg_dropout_min=args.sg_dropout_min,
    ).to(device)
    trainer = Trainer(
        args,
        model=model,
        g=ggi_graph,
        pvalues=pvalues,
        device=device,
        lr=args.lr,
        weight_decay=0,
        sg_l1=args.sg_l1,
        model_name=f"test",
    )
    best_val_score, test_auroc_score, test_ap_score = trainer.train(
        train_loader, val_loader, test_loader
    )
    print(
        f"Best validation AUPRC:{best_val_score}, test AUROC: {test_auroc_score}, test AUPRC: {test_ap_score}"
    )
