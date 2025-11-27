import torch
from torch import nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch.conv import GINConv
from dgl.readout import sum_nodes
def bert_init_params(module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.BatchNorm1d):
        if module.weight is not None:
            torch.nn.init.normal_(module.weight.data, mean=1, std=0.02)
        if module.weight is not None:
            torch.nn.init.constant_(module.bias.data, 0)

class AttentiveReadout(nn.Module):
    def __init__(self, in_feats):
        super(AttentiveReadout, self).__init__()
        self.in_feats = in_feats
        self.key_layer = nn.Linear(in_feats, in_feats)
        self.weight_layer = nn.Sequential(
            nn.Linear(in_feats, 1, bias=False),
            nn.Sigmoid()
        )
        self.value_layer = nn.Linear(in_feats, in_feats)
    def forward(self, g, feats):
        with g.local_scope():
            keys = self.key_layer(feats)
            g.ndata['w'] = self.weight_layer(keys)
            g.ndata['v'] = self.value_layer(feats)
            h = sum_nodes(g, 'v', 'w')
            return h, g.ndata['w']

class MLP(nn.Module):
    def __init__(self, d_input, d_hidden, d_output, n_layers, activation=nn.GELU(), bias=False, dropout=0, use_batchnorm=False, out_batchnorm=False):
        super().__init__()
        self.n_layers = n_layers
        self.use_batchnorm = use_batchnorm
        self.out_batchnorm = out_batchnorm
        self.linear_list = nn.ModuleList()
        if n_layers == 1:
            self.linear_list.append(nn.Linear(d_input, d_output, bias=bias))
        else:
            self.linear_list.append(nn.Linear(d_input, d_hidden, bias=bias))
            for _ in range(n_layers-2):
                self.linear_list.append(nn.Linear(d_hidden, d_hidden, bias=bias))
            self.linear_list.append(nn.Linear(d_hidden, d_output, bias=bias))
        if use_batchnorm:
            self.batch_norm_list = nn.ModuleList()
            for _ in range(n_layers):
                self.batch_norm_list.append(nn.BatchNorm1d((d_hidden),affine=False))
        self.activation=activation
        self.dropout = nn.Dropout(dropout)
    def forward(self, h):
        for i in range(self.n_layers-1):
            h = self.linear_list[i](h)
            if self.use_batchnorm:
                self.batch_norm_list[i](h)
            h = self.dropout(self.activation(h))
        h = self.linear_list[-1](h)
        if self.use_batchnorm and self.out_batchnorm:
            self.batch_norm_list[-1](h)
        if self.out_batchnorm:
            h = self.dropout(self.activation(h))
        return h

class SNP2Gene(nn.Module):
    def __init__(self, gene_ids, gene2snp_len, snp_ids, n_snps, n_genes, pvalues, d_gene, n_snp_kernels=32, sg_dropout_init=0.9, sg_dropout_min=0.15, use_adaptive_dropout=True):
        super().__init__()
        self.pvalues = pvalues
        self.gene_ids = gene_ids
        self.gene2snp_len = gene2snp_len
        self.snp_ids = snp_ids
        self.d_gene = d_gene
        self.n_snp_kernels = n_snp_kernels
        self.n_snps = n_snps
        self.n_genes = n_genes
        self.sg_dropout_init = sg_dropout_init
        self.sg_dropout_min = sg_dropout_min
        self.filter_list = nn.ParameterList()
        self.gene_padding = nn.Embedding(1, self.d_gene)
        self.gene_embedding = nn.Embedding(self.n_genes, self.d_gene)
        for _ in range(self.n_snp_kernels):
            self.filter_list.append(nn.Parameter(torch.randn(self.n_snps)*1e-5))
        # self.gene_proj = MLP(d_input=self.n_snp_kernels, d_hidden=self.d_gene, d_output=self.d_gene, n_layers=2, bias=True, use_batchnorm=True)
        self.gene_proj = nn.Embedding(self.n_genes, self.n_snp_kernels*self.d_gene)

    def forward(self, snp):
        device = snp.device
        batch_size = len(snp)
        if snp is not None:
            snp = snp.reshape(batch_size, -1, 1)
        snp_h_list = [torch.einsum('bnd,n->bnd', self.adaptive_dropout(snp.to(device), self.pvalues), self.filter_list[i].to(device)) for i in range(self.n_snp_kernels)]
        snp_h = torch.concatenate(snp_h_list, dim=-1)
        node_features = snp_h[:, self.snp_ids, :]
        data_reshaped = node_features.permute(1, 0, 2).reshape(len(self.snp_ids), -1)
        gene_features_reduced = torch.segment_reduce(data=data_reshaped, reduce="sum", lengths=self.gene2snp_len, axis=0)
        gene_h = gene_features_reduced.reshape(-1, batch_size, self.n_snp_kernels).permute(1, 0, 2)#.reshape(-1, self.n_snp_kernels) 
        gene_h = torch.einsum('bng,ngf->bnf', gene_h, self.gene_proj.weight.view(self.n_genes, self.n_snp_kernels, self.d_gene))
        gene_h = gene_h + self.gene_embedding.weight 
        return gene_h
    def adaptive_dropout(self, snp, p_value):
        if self.training:
            snp = snp.squeeze(-1)
            p_value = self.sg_dropout_init / (-torch.log10(p_value))
            p_value = torch.clamp(p_value, min=self.sg_dropout_min, max=0.99)
            random_mask = torch.rand(snp.shape[0], snp.shape[1]).to(snp.device) < p_value
            snp = snp.masked_fill(random_mask, 0)
            snp = snp / (1 - p_value)
            return snp.unsqueeze(-1)
        else:
            return snp
class PRSNet2(nn.Module):
    def __init__(self, n_snps, gene_ids, gene2snp_len, snp_ids, pvalues, n_genes, d_hidden=64, n_gnn_layers=1, n_predictor_layer=1, dropout=0, n_snp_kernels=32, sg_dropout_init=0.9, sg_dropout_min=0.15):
        super().__init__()
        self.activation = nn.GELU()
        ## Gene Encoder
        self.gene_encoder = SNP2Gene(gene_ids, gene2snp_len, snp_ids, n_snps, n_genes, pvalues, d_gene=d_hidden, n_snp_kernels=n_snp_kernels, sg_dropout_init=sg_dropout_init, sg_dropout_min=sg_dropout_min)
        ## GIN
        self.gnn_layer_list = nn.ModuleList()
        for _ in range(n_gnn_layers):
            mlp = MLP(d_hidden, d_hidden, d_hidden, n_layers=1, dropout=dropout, activation=self.activation, bias=False, use_batchnorm=True, out_batchnorm=False)
            self.gnn_layer_list.append(
                GINConv(mlp, learn_eps=False, aggregator_type='sum')
            )
        self.batch_norm_list = nn.ModuleList()
        for _ in range(n_gnn_layers):
            self.batch_norm_list.append(nn.BatchNorm1d(d_hidden, affine=False))
        ## Readout
        self.pool = AttentiveReadout(d_hidden)
        ## Predictor
        self.predictor = MLP(d_input=d_hidden, d_hidden=d_hidden, d_output=1, n_layers=n_predictor_layer, dropout=0, activation=self.activation, bias=True, use_batchnorm=True, out_batchnorm=False)
        self.apply(lambda module: bert_init_params(module))
        ## Hyperparameters
        self.n_gnn_layers = n_gnn_layers
        self.d_hidden = d_hidden
        self.n_genes = n_genes
    def forward(self, gene_gene_g, x):
        h = self.gene_encoder(x)
        h = h.reshape(-1, self.d_hidden)
        h_list = [h]
        for i in range(self.n_gnn_layers):
            h = self.gnn_layer_list[i](gene_gene_g, h)
            h = self.batch_norm_list[i](h)
            h = F.gelu(h)
            h_list.append(h)
        g_h, weights = self.pool(gene_gene_g, h)
        return self.predictor(g_h), weights
