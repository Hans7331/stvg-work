import torch
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange




# part of Transformer  
class FeedForward(nn.Module): # MLP
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


## Main ViViT model (2)
class ViViT(nn.Module):
    # depth default 4
    def __init__(self, image_size, patch_size, num_classes, frames_per_clip = 32, dim = 192, depth = 4, heads = 3, 
        pool = 'cls', in_channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., scale_dim = 4, ):
        super().__init__()
        
        # mlp_dim = 3072
        # num_layers = 12 ( spatial) 4 ( temporal) [is depth]
        # num_heads = 12


        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'


        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, frames_per_clip, num_patches + 1, dim))
        
        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        self.space_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        
        x = self.to_patch_embedding(x)
        # torch.Size([32, 32, 196, 192])
        
        b, t, n, _ = x.shape
        # torch.Size([32, 32, 196, 192])
        
        cls_space_tokens = repeat(self.space_token, '() n d -> b t n d', b = b, t=t)
        # torch.Size([32, 32, 196, 192])
        
        x = torch.cat((cls_space_tokens, x), dim=2)
        # torch.Size([32, 32, 197, 192])
        
        x += self.pos_embedding[:, :, :(n + 1)]
        # torch.Size([32, 32, 197, 192])
        
        x = self.dropout(x)
        # torch.Size([32, 32, 197, 192])

        x = rearrange(x, 'b t n d -> (b t) n d')
        # torch.Size([1024, 197, 192])
        
        x = self.space_transformer(x)
        # torch.Size([1024, 197, 192])
        
        x = rearrange(x[:, 0], '(b t) ... -> b t ...', b=b)
        # torch.Size([32, 32, 192])
        
        cls_temporal_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)
        # torch.Size([32, 32, 192])
        
        x = torch.cat((cls_temporal_tokens, x), dim=1)
        # torch.Size([32, 32, 192])
        
        x = self.temporal_transformer(x)
        # torch.Size([32, 32, 192])
        
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        # torch.Size([32, 192])
        
        return self.mlp_head(x)
