import torch
from torch import nn
import dgl
import numpy as np
from torchmetrics import AUROC, AveragePrecision


class Trainer:
    def __init__(
        self,
        args,
        model,
        g,
        pvalues,
        device,
        sg_l1,
        lr=1e-4,
        weight_decay=0,
        eval_interval=100,
        n_steps=50000,
        n_early_stop=100,
        log_interval=20,
        model_name="model",
    ):
        self.args = args
        self.model = model
        self.pvalues = pvalues
        self.sg_l1 = sg_l1
        self.device = device
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        self.auroc = AUROC(task="binary")
        self.ap = AveragePrecision(task="binary")
        self.g = g.to(device)
        self.eval_interval = eval_interval
        self.n_steps = n_steps
        self.n_early_stop = n_early_stop
        self.log_interval = log_interval

    def train(self, train_loader, val_loader, test_loader):
        print("----------------Training----------------")
        best_val_score, test_auroc_score, test_ap_score = 0, 0, 0
        running_pred_loss = []
        running_wd_loss = []
        self.model.train()
        data_iter = iter(train_loader)
        next_batch = next(data_iter)
        next_batch = [_.cuda(non_blocking=True) for _ in next_batch]
        for i in range(len(train_loader)):
            batch = next_batch
            if i + 1 != len(train_loader):
                next_batch = next(data_iter)
                next_batch = [_.cuda(non_blocking=True) for _ in next_batch]
            inputs, labels = batch
            batched_graph = dgl.batch([self.g] * len(labels))

            self.optimizer.zero_grad()

            outputs, gene_weights = self.model(batched_graph, inputs)
            pred_loss = self.loss_fn(outputs, labels)

            l1_loss = (
                torch.sum(
                    torch.stack(
                        [
                            torch.sum(torch.abs(param) * self.pvalues)
                            for param in self.model.gene_encoder.filter_list
                        ]
                    )
                )
                / len(self.model.gene_encoder.filter_list)
                / torch.sum(self.pvalues)
            )
            loss = pred_loss + self.sg_l1 * l1_loss
            loss.backward()
            self.optimizer.step()

            # Update running loss
            running_pred_loss.append(pred_loss.detach().cpu().numpy())
            running_wd_loss.append(l1_loss.detach().cpu().numpy())
            if (i + 1) % self.log_interval == 0:
                print(
                    f"[{i+1}] pred loss: {np.mean(running_pred_loss):.3f}", flush=True
                )
                print(f"[{i+1}] l1 loss: {np.mean(running_wd_loss):.8f}", flush=True)
                running_pred_loss = []
                running_wd_loss = []
            if (i + 1) % self.eval_interval == 0:
                print("----------------Validating----------------")
                val_auroc_score, val_ap_score = self.evaluate(val_loader)
                if val_ap_score > best_val_score:
                    best_val_score = val_ap_score
                    test_auroc_score, test_ap_score = self.evaluate(test_loader)
                    cur_early_stop = 0
                else:
                    cur_early_stop += 1
                    if cur_early_stop == self.n_early_stop:
                        break
                print(
                    f"[{i+1}] val_auroc: {val_auroc_score:.5f}, val_ap: {val_ap_score:.5f}, test_auroc: {test_auroc_score: .5f}, test_ap: {test_ap_score: .5}"
                )
                print("----------------Training----------------")
                self.model.train()
            if i == self.n_steps:
                break
        return best_val_score, test_auroc_score, test_ap_score

    def evaluate(self, test_loader):
        with torch.no_grad():
            self.model.eval()
            preds_list, labels_list = [], []
            for inputs, labels in test_loader:
                batched_graph = dgl.batch([self.g] * len(labels))
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).reshape(-1).to(torch.long)
                preds, _ = self.model(batched_graph, inputs)
                preds_list.append(preds.detach())
                labels_list.append(labels.detach())
            preds = torch.cat(preds_list).reshape(-1)
            labels = torch.cat(labels_list).reshape(-1)
            return (
                self.auroc(preds, labels).item(),
                self.ap(preds, labels.long()).item(),
            )

    def predict(self, data_loader):
        with torch.no_grad():
            preds_list = []
            weights_list = []
            self.model.eval()
            for inputs, labels in data_loader:
                batched_graph = dgl.batch([self.g] * len(labels))
                inputs = inputs.to(self.device)
                preds, weights = self.model(batched_graph, inputs)
                print(weights.shape, flush=True)
                preds_list += preds.detach().cpu().numpy().reshape(-1).tolist()
                weights_list += (
                    weights.detach().cpu().numpy().reshape(len(labels), -1).tolist()
                )
            return preds_list, weights_list
