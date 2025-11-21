import torch
from torch import einsum
from einops import rearrange

class Attention(torch.nn.Module):
    def __init__(self, query_dim=768, context_dim=768, heads=8, dim_head=96, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = torch.nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = torch.nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = torch.nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = torch.nn.Sequential(
            torch.nn.Linear(inner_dim, query_dim),
            torch.nn.Dropout(dropout)
        )

    def forward(self, x, context=None):
        h = self.heads
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class Sharpener(torch.nn.Module):
    def __init__(self, input_dim, output_dim, nhead, inner_dim=512):
        super(Sharpener, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, inner_dim)
        self.attention = Attention(query_dim=inner_dim, context_dim=inner_dim, heads=nhead, dim_head=inner_dim//nhead)
        self.fc2 = torch.nn.Linear(inner_dim, output_dim)
        self.act = torch.nn.ReLU()

    def forward(self, src1, src2):
        output1, output2 = self.fc1(src1), self.fc1(src2)
        output1, output2 = self.act(output1), self.act(output2)
        output = self.attention(output1, output2)
        output = self.act(output)
        output = self.fc2(output)
        return output
