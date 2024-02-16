import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
import torch.utils.checkpoint
import utils
import revtorch as rv
import torch.nn as nn

from encoder_lut import get_encoder_embedding

"""
RevMHAEncoder
"""


class MHABlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.mixing_layer_norm = nn.BatchNorm1d(hidden_size)
        self.mha = nn.MultiheadAttention(hidden_size, num_heads, bias=False)

    def forward(self, hidden_states: Tensor):
        assert hidden_states.dim() == 3
        hidden_states = self.mixing_layer_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        hidden_states_t = hidden_states.transpose(0, 1)
        mha_output = self.mha(hidden_states_t, hidden_states_t, hidden_states_t)[0].transpose(0, 1)

        return mha_output


class FFBlock(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.feed_forward = nn.Linear(hidden_size, intermediate_size)
        self.output_dense = nn.Linear(intermediate_size, hidden_size)
        self.output_layer_norm = nn.BatchNorm1d(hidden_size)
        self.activation = nn.GELU()

    def forward(self, hidden_states: Tensor):
        hidden_states = self.output_layer_norm(hidden_states.transpose(1, 2)).transpose(1, 2).contiguous()
        intermediate_output = self.feed_forward(hidden_states)
        intermediate_output = self.activation(intermediate_output)
        output = self.output_dense(intermediate_output)

        return output


class RevMHAEncoder(nn.Module):
    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        embedding_dim: int,
        input_dim: int,
        intermediate_dim: int,
        add_init_projection=True,
        use_position_encoding=False,
    ):
        super().__init__()
        # print("RevMHAEncoder init\n\n\n\n\n")
        if add_init_projection or input_dim != embedding_dim:
            self.init_projection_layer = torch.nn.Linear(input_dim, embedding_dim)
        else:
            self.lut = get_encoder_embedding
            self.w = nn.Linear(input_dim, embedding_dim)
            
        self.num_hidden_layers = n_layers
        blocks = []
        for _ in range(n_layers):
            f_func = MHABlock(embedding_dim, n_heads)
            g_func = FFBlock(embedding_dim, intermediate_dim)
            # we construct a reversible block with our F and G functions
            blocks.append(rv.ReversibleBlock(f_func, g_func, split_along_dim=-1))

        self.sequence = rv.ReversibleSequence(nn.ModuleList(blocks))

        self.use_position_encoding = use_position_encoding
        if self.use_position_encoding:
            # self.norm = nn.LayerNorm(embedding_dim)
            self.norm = nn.BatchNorm1d(embedding_dim)

    def forward(self, x: Tensor, pe: Tensor, mask=None):
        if hasattr(self, "init_projection_layer"):
            x = self.init_projection_layer(x)
        else:
            y = []
            for batch_idx, tsp_instance in enumerate(x):
                y.append(self.lut(tsp_instance))
            y = torch.stack(y, dim=0)
            x = self.w(y).relu()
        
        if self.use_position_encoding:
            # x = self.norm(x + pe)
            x = self.norm((x + pe).transpose(1, 2)).transpose(1, 2)
            
        x = torch.cat([x, x], dim=-1)
        out = self.sequence(x)
        return torch.stack(out.chunk(2, dim=-1))[-1]


class DecoderForLarge(torch.nn.Module):
    def __init__(
        self,
        embedding_dim,
        n_heads=8,
        tanh_clipping=10.0,
        multi_pointer=1,
        multi_pointer_level=1,
        add_more_query=True,
        use_position_encoding=False,
        graph_size=100
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.tanh_clipping = tanh_clipping

        self.Wq_graph = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wq_first = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.Wq_last = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.wq = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.W_visited = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.Wk = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
        # self.Wv = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
        # self.multi_head_combine = torch.nn.Linear(embedding_dim, embedding_dim)

        self.q_graph = None  # saved q1, for multi-head attention
        self.q_first = None  # saved q2, for multi-head attention
        self.glimpse_k = None  # saved key, for multi-head attention
        self.glimpse_v = None  # saved value, for multi-head_attention
        self.logit_k = None  # saved, for single-head attention
        self.group_ninf_mask = None  # reference to ninf_mask owned by state
        self.multi_pointer = multi_pointer  #
        self.multi_pointer_level = multi_pointer_level
        self.add_more_query = add_more_query
        
        
        self.use_position_encoding=use_position_encoding
        
        if self.use_position_encoding:
            # self.norm=torch.nn.LayerNorm(embedding_dim)
            self.norm=torch.nn.LayerNorm(embedding_dim)
            pe_init = get_circular_position_embeddings(max_len=graph_size,d_model = embedding_dim)
            self.pos_embedding = nn.Embedding.from_pretrained(pe_init, freeze=True)

        
    def get_circular_position_embeddings(max_len, d_model):
        """ Create a positional encoding matrix for max_len positions with d_model dimensions. """
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term+2*np.pi*position/max_len)
        pe[:, 1::2] = torch.cos(position * div_term+2*np.pi*position/max_len)
        
        return pe    
    
    def reset(self, coordinates, embeddings, G, trainging=True):
        # embeddings.shape = [B, N, H]
        # graph_embedding.shape = [B, 1, H]
        # q_graph.hape = [B, n_heads, 1, key_dim]
        # glimpse_k.shape = glimpse_v.shape =[B, n_heads, N, key_dim]
        # logit_k.shape = [B, H, N]
        # group_ninf_mask.shape = [B, G, N]

        B, N, H = embeddings.shape
        # G = group_ninf_mask.size(1)

        self.coordinates = coordinates  # [:,:2]
        self.embeddings = embeddings
        self.embeddings_group = self.embeddings.unsqueeze(1).expand(B, G, N, H)
        graph_embedding = self.embeddings.mean(dim=1, keepdim=True)

        self.q_graph = self.Wq_graph(graph_embedding)
        self.q_first = None
        self.logit_k = embeddings.transpose(1, 2)
        if self.multi_pointer > 1:
            self.logit_k = utils.make_heads(self.Wk(embeddings), self.multi_pointer).transpose(
                2, 3
            )  # [B, n_heads, key_dim, N]

    def forward(self, last_node, group_ninf_mask, S):
        B, N, H = self.embeddings.shape
        G = group_ninf_mask.size(1)

        if self.use_position_encoding:
            position_ids = (torch.arange(self.embeddings.size(1), dtype=torch.int, device=self.embeddings.device).unsqueeze(0).expand_as(self.embeddings))
            pos_encoding = pos_embedding(position_ids)
            pos_encoded = self.norm(pos_encoding + self.embeddings)

        # Get last node embedding
        last_node_index = last_node.view(B, G, 1).expand(-1, -1, H)
        last_node_embedding = self.embeddings.gather(1, last_node_index)
        q_last = self.Wq_last(last_node_embedding)

        # Get frist node embedding
        if self.q_first is None:
            self.q_first = self.Wq_first(last_node_embedding)
        group_ninf_mask = group_ninf_mask.detach()

        mask_visited = group_ninf_mask.clone()
        mask_visited[mask_visited == -np.inf] = 1.0
        q_visited = self.W_visited(torch.bmm(mask_visited, self.embeddings) / N)
        D = self.coordinates.size(-1)
        last_node_coordinate = self.coordinates.gather(dim=1, index=last_node.unsqueeze(-1).expand(B, G, D))
        distances = torch.cdist(last_node_coordinate, self.coordinates)

        if self.add_more_query:
            final_q = q_last + self.q_first + self.q_graph + q_visited
        else:
            final_q = q_last + self.q_first + self.q_graph

        if self.multi_pointer > 1:
            final_q = utils.make_heads(self.wq(final_q), self.n_heads)  # (B,n_head,G,H)  (B,n_head,H,N)
            score = (torch.matmul(final_q, self.logit_k) / math.sqrt(H)) - (distances / math.sqrt(2)).unsqueeze(
                1
            )  # (B,n_head,G,N)
            if self.multi_pointer_level == 1:
                score_clipped = self.tanh_clipping * torch.tanh(score.mean(1))
            elif self.multi_pointer_level == 2:
                score_clipped = (self.tanh_clipping * torch.tanh(score)).mean(1)
            else:
                # add mask
                score_clipped = self.tanh_clipping * torch.tanh(score)
                mask_prob = group_ninf_mask.detach().clone()
                mask_prob[mask_prob == -np.inf] = -1e8

                score_masked = score_clipped + mask_prob.unsqueeze(1)
                probs = F.softmax(score_masked, dim=-1).mean(1)
                return probs
        else:
            score = torch.matmul(final_q, self.logit_k) / math.sqrt(H) - distances / math.sqrt(2)
            score_clipped = self.tanh_clipping * torch.tanh(score)

        # add mask
        mask_prob = group_ninf_mask.detach().clone()
        mask_prob[mask_prob == -np.inf] = -1e8
        score_masked = score_clipped + mask_prob
        probs = F.softmax(score_masked, dim=2)

        return probs
