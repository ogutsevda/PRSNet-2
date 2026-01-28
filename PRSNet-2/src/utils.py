import numpy as np
import random
import torch
import dgl


def collate_fn(batch):
    x = [sample["x"] for sample in batch]
    y = [sample["y"] for sample in batch]
    x = torch.from_numpy(np.stack(x)).to(torch.float32)
    y = torch.FloatTensor(y).to(torch.float32).reshape(-1, 1)
    return x, y


def set_random_seed(seed=22, n_threads=8):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.seed(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def generate_splits(labels, seed):
    pos_indexs = np.where(labels == 1)[0]
    neg_indexs = np.where(labels == 0)[0]
    np.random.seed(seed)
    np.random.shuffle(pos_indexs)

    shuffled_indices = np.random.permutation(len(pos_indexs))

    total_length = len(pos_indexs)
    train_ratio = 0.8
    val_ratio = 0.1

    train_size = int(total_length * train_ratio)
    val_size = int(total_length * val_ratio)

    train_indices = shuffled_indices[:train_size]
    val_indices = shuffled_indices[train_size : train_size + val_size]
    test_indices = shuffled_indices[train_size + val_size :]

    train_pos_data = pos_indexs[train_indices]
    val_pos_data = pos_indexs[val_indices]
    test_pos_data = pos_indexs[test_indices]

    train_neg_indices = np.random.choice(
        neg_indexs, size=int(len(neg_indexs) * 0.8), replace=False
    )
    val_neg_indices = np.random.choice(
        np.setdiff1d(neg_indexs, train_neg_indices),
        size=int(len(neg_indexs) * 0.1),
        replace=False,
    )
    test_neg_indices = np.setdiff1d(
        np.setdiff1d(neg_indexs, train_neg_indices), val_neg_indices
    )

    train_indices = np.concatenate((train_pos_data, train_neg_indices))
    val_indices = np.concatenate((val_pos_data, val_neg_indices))
    test_indices = np.concatenate((test_pos_data, test_neg_indices))

    return train_indices, val_indices, test_indices
