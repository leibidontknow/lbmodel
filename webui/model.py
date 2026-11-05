import torch
from torch import nn
from torch.nn import functional as F
def calc_ang(per_d, max_len, base = 10000.0):
    dim = torch.arange(0, per_d, 2, dtype = torch.float32)
    freqs = base ** (-dim / per_d)
    pos = torch.arange(max_len, dtype = torch.float32)
    angs = torch.outer(pos, freqs)
    cos = angs.cos()
    sin = angs.sin()
    cos = torch.cat((cos, cos), dim = -1)
    sin = torch.cat((sin, sin), dim = -1)
    return cos, sin 
def seq_work(x):
    l, r = x.chunk(2, dim = -1)
    return torch.cat((-r, l), dim = -1)
def RoPE(q, k, cos, sin):
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    typ = q.dtype
    q = q * cos + seq_work(q) * sin
    k = k * cos + seq_work(k) * sin 
    return q.to(typ), k.to(typ)
class SelfAttention(nn.Module):
    def __init__(self, in_d, head_cnt, kv_cnt):
        super().__init__()
        self.in_d = in_d
        self.head_cnt = head_cnt
        self.per_d = in_d // head_cnt
        self.kv_cnt = kv_cnt
        self.q_per_kv = head_cnt // kv_cnt
        self.Q_lin = nn.Linear(in_d, in_d, bias = False)
        self.K_lin = nn.Linear(in_d, kv_cnt * self.per_d, bias = False)
        self.V_lin = nn.Linear(in_d, kv_cnt * self.per_d, bias = False)
        self.output = nn.Linear(in_d, in_d, bias = False)
        self.q_norm = nn.RMSNorm(self.per_d)
        self.k_norm = nn.RMSNorm(self.per_d)

    def forward(self, x, cos, sin, pre_kv = None, cache = False):
        batch_siz, in_len, in_d = x.shape
        q = self.Q_lin(x)
        k = self.K_lin(x)
        v = self.V_lin(x)
        q = q.reshape(batch_siz, in_len, self.head_cnt, self.per_d)
        k = k.reshape(batch_siz, in_len, self.kv_cnt, self.per_d)
        v = v.reshape(batch_siz, in_len, self.kv_cnt, self.per_d)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = RoPE(q, k, cos, sin)
        cached = pre_kv is not None
        if cached:
            pre_k, pre_v = pre_kv
            k = torch.cat((pre_k, k), dim = 2)
            v = torch.cat((pre_v, v), dim = 2)
        res_att = F.scaled_dot_product_attention(q, k, v, is_causal = not cached, enable_gqa=True)
        res = res_att.transpose(1, 2)
        res = res.reshape(batch_siz, in_len, in_d)
        cur_kv = None 
        if cache:
            cur_kv = (k, v)
        return self.output(res), cur_kv
class FeedForward(nn.Module):
    def __init__(self, in_d, hidden_d):
        super().__init__()
        self.up = nn.Linear(in_d, hidden_d, bias = False)
        self.gate = nn.Linear(in_d, hidden_d, bias = False)
        self.down = nn.Linear(hidden_d, in_d, bias = False)
    def forward(self, x):
        return self.down(self.up(x) * F.silu(self.gate(x)))
class TransformerBlock(nn.Module):
    def __init__(self, in_d, head_cnt, hidden_d, kv_cnt):
        super().__init__()
        self.norm_att = nn.RMSNorm(in_d);
        self.att = SelfAttention(in_d, head_cnt, kv_cnt)
        self.norm_ffn = nn.RMSNorm(in_d)
        self.ffn = FeedForward(in_d, hidden_d)
    def forward(self, x, cos, sin, pre_kv = None, cache = False):
        res, kv = self.att(self.norm_att(x), cos, sin, pre_kv, cache)
        x = x + res
        x = x + self.ffn(self.norm_ffn(x))
        return x, kv
class Model(nn.Module):
    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear):
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    def __init__(self, voc_siz, max_len, in_d, head_cnt, hidden_d, layer_cnt, kv_cnt=None, rope_base=10000.0):
        super().__init__()
        if kv_cnt is None:
            kv_cnt = head_cnt
        self.max_len = max_len 
        self.token_emb = nn.Embedding(voc_siz, in_d)
        self.blocks = nn.ModuleList()
        for i in range(layer_cnt):
            self.blocks.append(TransformerBlock(in_d, head_cnt, hidden_d, kv_cnt))
        self.norm = nn.RMSNorm(in_d)
        self.output = nn.Linear(in_d, voc_siz, bias=False)
        cos, sin = calc_ang(in_d // head_cnt, max_len, base=rope_base)
        self.register_buffer("rope_cos", cos, persistent = False)
        self.register_buffer("rope_sin", sin, persistent = False)
        self.apply(self.init_weights)
        self.output.weight = self.token_emb.weight
    def forward(self, ids, pre_kv = None, cache = False):
        batch_siz, in_len = ids.shape
        start_pos = 0
        if pre_kv is not None:
            start_pos = pre_kv[0][0].shape[2]
        end_pos = start_pos + in_len
        if(end_pos > self.max_len):
            raise ValueError("输入长度过长")
        cos = self.rope_cos[start_pos : end_pos]
        sin = self.rope_sin[start_pos : end_pos]
        x = self.token_emb(ids)
        kv = []
        for i, block in enumerate(self.blocks):
            cur_kv = None
            if pre_kv is not None:
                cur_kv = pre_kv[i]
            x, new_kv = block(x, cos, sin, cur_kv, cache)
            if cache:
                kv.append(new_kv)
        logits = self.output(self.norm(x))
        if cache:
            return logits, kv 
        return logits
