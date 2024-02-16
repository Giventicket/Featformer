class MHAEncoderLayer(torch.nn.Module):
    def __init__(self, embedding_dim, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.Wq = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wk = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wv = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.multi_head_combine = torch.nn.Linear(embedding_dim, embedding_dim)
        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, embedding_dim * 4),
            torch.nn.ReLU(),
            torch.nn.Linear(embedding_dim * 4, embedding_dim),
        )
        self.norm1 = torch.nn.BatchNorm1d(embedding_dim)
        self.norm2 = torch.nn.BatchNorm1d(embedding_dim)

    def forward(self, x, mask=None):
        q = utils.make_heads(self.Wq(x), self.n_heads)
        k = utils.make_heads(self.Wk(x), self.n_heads)
        v = utils.make_heads(self.Wv(x), self.n_heads)
        x = x + self.multi_head_combine(utils.multi_head_attention(q, k, v, mask))
        x = self.norm1(x.view(-1, x.size(-1))).view(*x.size())
        x = x + self.feed_forward(x)
        x = self.norm2(x.view(-1, x.size(-1))).view(*x.size())
        return x


class MHAEncoder(torch.nn.Module):
    def __init__(
        self,
        n_layers,
        n_heads,
        embedding_dim,
        input_dim,
        add_init_projection=True,
        use_position_encoding=False,
    ):
        super().__init__()
        # print("MHAEncoder init\n\n\n\n\n")
        if add_init_projection or input_dim != embedding_dim:
            self.init_projection_layer = torch.nn.Linear(input_dim, embedding_dim)
        self.attn_layers = torch.nn.ModuleList(
            [MHAEncoderLayer(embedding_dim=embedding_dim, n_heads=n_heads) for _ in range(n_layers)]
        )
        self.use_positionalencoding = positional_encoding
        self.positional_encoding_name = positional_encoding_name
        self.norm = nn.LayerNorm(embedding_dim)
        
        self.embedding_dim = embedding_dim
        if self.use_positionalencoding:
            self.grid_size = 16
            x_grid = torch.linspace(0, 1, self.grid_size)
            y_grid = torch.linspace(0, 1, self.grid_size)
            grid = torch.zeros(self.grid_size, self.grid_size, 128)
            for i, x_val in enumerate(x_grid):
                for j, y_val in enumerate(y_grid):
                    grid[i, j] = self.positional_encoding(x_val, y_val, 128, "xy_sum")
            self.pre_calc_pos = grid
            
        self.use_position_encoding = use_position_encoding
        if self.use_position_encoding:
            self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor, pe: Tensor, mask=None):
        if hasattr(self, "init_projection_layer"):
            x = self.init_projection_layer(x)
            
        if self.use_position_encoding:
            x = self.norm(x + pe)

        for idx, layer in enumerate(self.attn_layers):
            x = layer(x, mask)
        return x


class Decoder(torch.nn.Module):
    def __init__(
        self,
        embedding_dim,
        n_heads=8,
        tanh_clipping=10.0,
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

        self.Wk = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wv = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.multi_head_combine = torch.nn.Linear(embedding_dim, embedding_dim)
        # self.query_fc = torch.nn.Linear(embedding_dim*2, embedding_dim)

        self.q_graph = None  # saved q1, for multi-head attention
        self.q_first = None  # saved q2, for multi-head attention
        self.glimpse_k = None  # saved key, for multi-head attention
        self.glimpse_v = None  # saved value, for multi-head_attention
        self.logit_k = None  # saved, for single-head attention
        self.group_ninf_mask = None  # reference to ninf_mask owned by state
        
        self.use_position_encoding=use_position_encoding
        
        if self.use_position_encoding:
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
    

    def reset(self, coordinates, embeddings, G, trainging=False):
        # embeddings.shape = [B, N, H]
        # graph_embedding.shape = [B, 1, H]
        # q_graph.hape = [B, n_heads, 1, key_dim]
        # glimpse_k.shape = glimpse_v.shape =[B, n_heads, N, key_dim]
        # logit_k.shape = [B, H, N]
        # group_ninf_mask.shape = [B, G, N]
        self.coordinates = coordinates
        self.embeddings = embeddings
        graph_embedding = self.embeddings.mean(dim=1, keepdim=True)

        self.q_graph = self.Wq_graph(graph_embedding)
        self.q_first = None
        self.glimpse_k = utils.make_heads(self.Wk(embeddings), self.n_heads)
        self.glimpse_v = utils.make_heads(self.Wv(embeddings), self.n_heads)
        self.logit_k = embeddings.transpose(1, 2)
        # self.group_ninf_mask = group_ninf_mask

    def forward(self, last_node, group_ninf_mask, step):
        self.group_ninf_mask = group_ninf_mask
        B, N, H = self.embeddings.shape
        G = self.group_ninf_mask.size(1)

        last_node_index = last_node.view(B, G, 1).expand(-1, -1, H)
        last_node_embedding = self.embeddings.gather(1, last_node_index)

        #추정은 여기쯤?
        if self.use_position_encoding:
            position_ids = (torch.arange(self.embeddings.size(1), dtype=torch.int, device=self.embeddings.device).unsqueeze(0).expand_as(self.embeddings))
            pos_encoding = pos_embedding(position_ids)
            pos_encoded = self.norm(pos_encoding + self.embeddings)
        
        # q_graph.shape = [B, n_heads, 1, key_dim]
        # q_first.shape = q_last.shape = [B, n_heads, G, key_dim]
        if self.q_first is None:
            # self.q_first = utils.make_heads(self.Wq_first(last_node_embedding),
            #                                 self.n_heads)
            self.q_first = self.Wq_first(last_node_embedding)

        #근데 이 last도 별개로 있단 말이죠?
        
        q_last = self.Wq_last(last_node_embedding)
        # glimpse_q.shape = [B, n_heads, G, key_dim]
        glimpse_q = self.q_first + q_last + self.q_graph

        glimpse_q = utils.make_heads(glimpse_q, self.n_heads)
        if self.n_decoding_neighbors is not None:
            D = self.coordinates.size(-1)
            K = torch.count_nonzero(self.group_ninf_mask[0, 0] == 0.0).item()
            K = min(self.n_decoding_neighbors, K)
            last_node_coordinate = self.coordinates.gather(dim=1, index=last_node.unsqueeze(-1).expand(B, G, D))
            distances = torch.cdist(last_node_coordinate, self.coordinates)
            distances[self.group_ninf_mask == -np.inf] = np.inf
            indices = distances.topk(k=K, dim=-1, largest=False).indices
            glimpse_mask = torch.ones_like(self.group_ninf_mask) * (-np.inf)
            glimpse_mask.scatter_(dim=-1, index=indices, src=torch.zeros_like(glimpse_mask))
        else:
            glimpse_mask = self.group_ninf_mask

        # q_last_nhead=utils.make_heads(q_last,self.n_heads)
        attn_out = utils.multi_head_attention(
            q=glimpse_q,
            k=self.glimpse_k,
            v=self.glimpse_v,
            mask=glimpse_mask,
        )

        # mha_out.shape = [B, G, H]
        # score.shape = [B, G, N]
        final_q = self.multi_head_combine(attn_out)
        score = torch.matmul(final_q, self.logit_k)

        score_clipped = torch.tanh(score) * self.tanh_clipping
        score_masked = score_clipped + self.group_ninf_mask

        probs = F.softmax(score_masked, dim=2)

        assert (probs == probs).all(), "Probs should not contain any nans!"
        return probs


