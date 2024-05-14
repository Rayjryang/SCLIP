import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F
import torch.nn.init as init
import math

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim,eps=1e-6)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class OvercompleteISTABlock(nn.Module):
    """Transformer MLP / feed-forward block."""
    def __init__(self, d, overcomplete_ratio=4, eta=0.1, lmbda=0.1, decouple=True):
        super(OvercompleteISTABlock, self).__init__()
        self.eta = eta
        self.lmbda = lmbda
        self.overcomplete_ratio = overcomplete_ratio
        self.decouple = decouple
        self.d = d

        # Define the matrix D
        self.D = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d, overcomplete_ratio * d)))
        self.D1 = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d, overcomplete_ratio * d)))

    def forward(self, x):
        """Applies CRATE OvercompleteISTABlock module."""
        
        # First step of PGD: initialize at z0 = 0, compute lasso prox, get z1
        negative_lasso_grad = torch.einsum("pd,nlp->nld", self.D, x)
        z1 = F.relu(self.eta * negative_lasso_grad - self.eta * self.lmbda)

        # Second step of PGD: initialize at z1, compute lasso prox, get z2
        Dz1 = torch.einsum("dp,nlp->nld", self.D, z1)
        lasso_grad = torch.einsum("pd,nlp->nld", self.D, Dz1 - x)
        z2 = F.relu(z1 - self.eta * lasso_grad - self.eta * self.lmbda)

        xhat = torch.einsum("dp,nlp->nld", self.D1, z2)
      
        return xhat
    

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
      

        self.qkv = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False),
        ) if project_out else nn.Identity()

    def forward(self, x):
        w = rearrange(self.qkv(x), 'b n (h d) -> b h n d', h = self.heads)

        dots = torch.matmul(w, w.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
       

        out = torch.matmul(attn, w)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.heads = heads
        self.depth = depth
        self.dim = dim
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head)),
                PreNorm(dim, OvercompleteISTABlock(d = dim))
            ]))

    def forward(self, x):
      
        for attn, ff in self.layers:
           
            x = attn(x) + x
            x = ff(x) + x
        return x 

class CRATE(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, pool = 'cls', dim_head = 64):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        self.patch_size = patch_size
        self.heads = heads

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=dim, kernel_size=patch_size, stride=patch_size, bias=False, padding='valid')

        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
     

        self.transformer = Transformer(dim, depth, heads, dim_head)

        self.pool = pool

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes,bias=None)
        )
    def forward(self, img, return_all=False, csa=True):
        x = self.conv1(img)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)

        x += self.pos_embedding.to(x.dtype)
        x = self.transformer(x)
        
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x = self.mlp_head(x)

        return x
    

    def forward_(self, img, return_all=False, csa=True):

        B, nc, w, h = img.shape

        x = self.conv1(img)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]


        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)

        if x.shape[1] != self.pos_embedding.shape[0]:
            x = x + self.interpolate_pos_encoding(x, w, h).to(x.dtype)
        else:
            # x = x + self.positional_embedding.to(x.dtype)
            x += self.pos_embedding.to(x.dtype)
            # x += self.pos_embedding[:, :(n + 1)]
    
    
        for blk in self.transformer.layers[:-1]:
            # print(f'blk arch: {blk}')
            x = nn.Sequential(*blk)(x)
        for blk in self.transformer.layers[-1:]:
            x = x + self.custom_attn(blk[0].fn, blk[0].norm(x), csa=csa) # blk[0].fn is attn. norm is ln.
            x = x + blk[1](x) # blk[1] = mlp
        
        # x = self.transformer(x)

        if return_all:
            return  self.mlp_head(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x = self.mlp_head(x)

        return x
    
    def custom_attn(self, attn_layer, x, return_attn=False, with_attn=False, csa=False):
        
        # num_heads = attn_layer.num_heads
        num_heads = self.heads # ray

        _, bsz, embed_dim = x.size()
        head_dim = embed_dim // num_heads
        scale = head_dim ** -0.5

        q = F.linear(x, attn_layer.qkv.weight)
        k = v = q
        q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        if csa:
            q_attn = torch.bmm(q, q.transpose(1, 2)) * scale
            k_attn = torch.bmm(k, k.transpose(1, 2)) * scale
            attn_weights = F.softmax(q_attn, dim=-1) + F.softmax(k_attn, dim=-1)
        else:
            attn_weights = torch.bmm(q * scale, k.transpose(1, 2))
            attn_weights = F.softmax(attn_weights, dim=-1)

        if return_attn:
            return attn_weights

        attn_output = torch.bmm(attn_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(-1, bsz, embed_dim)
        # attn_output = attn_layer.out_proj(attn_output)
        attn_output = attn_layer.to_out(attn_output)

        if with_attn:
            return attn_output, attn_weights

        return attn_output
    
    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embedding.shape[0] - 1
        if npatch == N and w == h:
            return self.pos_embedding
        class_pos_embed = self.pos_embedding[[0]]
        patch_pos_embed = self.pos_embedding[1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    
    def get_attn(self, x, layer='all', csa=False):

        B, nc, w, h = x.shape

        x = self.conv1(x.type(self.conv1.weight.dtype))  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        
        if x.shape[1] != self.pos_embedding.shape[0]:
            x = x + self.interpolate_pos_encoding(x, w, h).to(x.dtype)
        else:
            x = x + self.pos_embedding.to(x.dtype)

        # x = self.ln_pre(x)
        # x = x.permute(1, 0, 2)  # NLD -> LND

        if layer == 'final':
            for blk in self.transformer.layers[:-1]:
                x = blk(x)
            attn_map = self.custom_attn(self.transformer.layers[-1][0].fn,
                                        self.transformer.layers[-1][0].norm(x),
                                        csa=csa, return_attn=True)
            return attn_map
        elif layer == 'all':
            attn_map = []
            for blk in self.transformer.resblocks[:-1]:
                x_i, attn_i = self.custom_attn(blk[0].fn, blk[0].norm(x), with_attn=True)
                x = x + x_i
                x = x + blk[1](x)
                attn_map.append(attn_i)
            for blk in self.transformer.resblocks[-1:]:
                x_i, attn_i = self.custom_attn(blk[0].fn, blk[0].norm(x), with_attn=True, csa=True)
                x = x + x_i
                x = x + blk[1](x)
                attn_map.append(attn_i)
            return attn_map
        else:
            raise ValueError('layer should be final or all')