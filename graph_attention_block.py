"""
图注意力网络模块 (纯 PyTorch 实现，无需 torch_geometric)
参考: Veličković et al. (2018) "Graph Attention Networks"

每个时间步接收动态邻接矩阵 + 节点特征 → 多头注意力聚合
适用于股票关系图: 节点=股票, 边=相关性, 每个时间步图结构变化
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """单层图注意力 (Multi-Head)"""

    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.1,
                 concat=True, negative_slope=0.2, residual=True):
        """
        in_dim:   输入节点特征维度
        out_dim:  每头输出维度
        num_heads: 注意力头数
        concat:    True=拼接多头输出, False=平均多头输出 (最后一层)
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.concat = concat
        self.dropout = dropout
        self.residual = residual

        # 线性变换 W (每头)
        self.W = nn.Parameter(torch.empty(num_heads, in_dim, out_dim))
        nn.init.xavier_uniform_(self.W)

        # 注意力权重向量 a (每头)
        self.a = nn.Parameter(torch.empty(num_heads, 2 * out_dim, 1))
        nn.init.xavier_uniform_(self.a)

        if residual and (in_dim != num_heads * out_dim if concat else out_dim):
            self.res_proj = nn.Linear(in_dim, num_heads * out_dim if concat else out_dim)
        else:
            self.res_proj = None

        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, adj, edge_weight=None):
        """
        x:    (batch_or_nodes, num_nodes, in_dim) — 节点特征
        adj:  (batch_or_nodes, num_nodes, num_nodes) — 邻接矩阵
        edge_weight: 可选的边权重 (batch_or_nodes, num_nodes, num_nodes)
        """
        # 确保有 batch 维度
        if x.dim() == 2:
            x = x.unsqueeze(0)
            adj = adj.unsqueeze(0)
            if edge_weight is not None:
                edge_weight = edge_weight.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        *batch_dims, N, _ = x.shape
        B = 1
        for d in batch_dims:
            B *= d

        # 线性变换: (B, N, heads, out_dim)
        x_transformed = torch.einsum('bni, hio -> bnho', x.reshape(B, N, -1), self.W)

        h_i = x_transformed.unsqueeze(2)   # (B, N, 1, heads, out_dim)
        h_j = x_transformed.unsqueeze(1)   # (B, 1, N, heads, out_dim)
        h_cat = torch.cat([h_i.expand(-1, -1, N, -1, -1),
                           h_j.expand(-1, N, -1, -1, -1)], dim=-1)

        e = self.leaky_relu(torch.einsum('bijhd, hdo -> bijh', h_cat, self.a))

        if edge_weight is not None:
            e = e + edge_weight.reshape(B, N, N).unsqueeze(-1).log().clamp(min=-10)

        adj_mask = (adj.reshape(B, N, N) > 0).float().unsqueeze(-1)
        e = e.masked_fill(adj_mask == 0, float('-inf'))

        alpha = F.softmax(e, dim=2)
        alpha = self.drop(alpha)

        h_out = torch.einsum('bijh, bjho -> biho', alpha, x_transformed)

        if self.concat:
            h_out = h_out.reshape(B, N, self.num_heads * self.out_dim)
        else:
            h_out = h_out.mean(dim=-2)

        # 残差连接
        x_flat = x.reshape(B, N, -1)
        if self.residual:
            if self.res_proj is not None:
                res = self.res_proj(x_flat)
            elif x_flat.shape[-1] == h_out.shape[-1]:
                res = x_flat
            else:
                res = torch.zeros(1, device=h_out.device)
            h_out = h_out + res

        h_out = F.elu(h_out)
        if squeeze_output:
            h_out = h_out.squeeze(0)
        return h_out


class DynamicStockGAT(nn.Module):
    """
    动态股票图注意力网络。
    每个时间步接收变化的邻接矩阵 (基于滚动相关系数),
    对股票节点特征进行消息传递，输出聚合后的节点表征。
    """

    def __init__(self, node_dim, hidden_dims=(64, 32), num_heads=(4, 4),
                 dropout=0.1, output_dim=1, use_edge_weight=True):
        """
        node_dim:     输入节点特征维度 (每只股票的技术指标数)
        hidden_dims:  各层隐藏维度 tuple
        num_heads:    各层注意力头数 tuple
        output_dim:   输出维度 (1=收益率预测)
        use_edge_weight: 是否将相关系数作为边权重
        """
        super().__init__()
        self.use_edge_weight = use_edge_weight

        self.layers = nn.ModuleList()
        cur_in = node_dim
        for i, hdim in enumerate(hidden_dims):
            is_last = (i == len(hidden_dims) - 1)
            nh = num_heads[i] if i < len(num_heads) else 1
            self.layers.append(GraphAttentionLayer(
                in_dim=cur_in,
                out_dim=hdim,
                num_heads=nh,
                dropout=dropout,
                concat=not is_last,
                residual=True,
            ))
            cur_in = hdim * nh if not is_last else hdim

        last_out = hidden_dims[-1]

        self.output_head = nn.Sequential(
            nn.Linear(last_out + 1, 64),  # +1 for global market signal
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim),
        )

    def forward(self, x, adj, edge_weight=None, market_signal=None):
        """
        x:             (num_nodes, node_dim) 或 (batch, num_nodes, node_dim)
        adj:           (num_nodes, num_nodes) 或 (batch, num_nodes, num_nodes)
        edge_weight:   可选边权重 (相关系数矩阵)
        market_signal: (num_nodes, 1) 或 None — 全局市场信号

        返回: (num_nodes, output_dim) 节点级预测
        """
        h = x
        for layer in self.layers:
            h = layer(h, adj, edge_weight if self.use_edge_weight else None)

        if market_signal is None:
            market_signal = torch.zeros(*h.shape[:-1], 1, device=h.device, dtype=h.dtype)
        h = torch.cat([h, market_signal], dim=-1)
        out = self.output_head(h)
        return out.squeeze(-1) if out.shape[-1] == 1 else out

    def build_adjacency_from_correlation(self, corr_matrix, top_k=None,
                                          keep_self_loop=True):
        """
        从相关系数矩阵构建邻接矩阵。
        corr_matrix: (num_nodes, num_nodes)
        top_k:       保留前k个最大的邻居 (None=全连接)
        """
        N = corr_matrix.shape[0]
        adj = corr_matrix.abs().clone().fill_diagonal_(0)

        if top_k is not None and top_k < N - 1:
            # 每行只保留 top_k 个最大值
            topk_vals, topk_idx = adj.topk(top_k, dim=-1)
            adj_new = torch.zeros_like(adj)
            adj_new.scatter_(-1, topk_idx, topk_vals)
            adj = adj_new

        if keep_self_loop:
            adj = adj + torch.eye(N, device=adj.device)

        return adj