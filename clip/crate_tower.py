### CLIP source code from OpenAI:
# https://github.com/openai/CLIP/blob/main/clip/clip.py

from collections import OrderedDict
from typing import Tuple, Union
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.normalization import _shape_t

import torchvision.transforms.functional as VF
from crate_clip import CRATE
from open_clip import get_tokenizer



class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def __init__(self, normalized_shape: _shape_t, eps: float = 0.00001, elementwise_affine: bool = True, device=None, dtype=None) -> None:
        super().__init__(normalized_shape, eps, elementwise_affine, device, dtype)

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
    

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", nn.GELU(approximate='tanh')),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        # self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False)[0]
        # return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):

        print("============single block start============")
        print(f'ln_1 input: {np.mean(x.detach().numpy())}')
        y = self.ln_1(x)
        print(f'attn input: {np.mean(y.detach().numpy())}')
        y = self.attention(y)
        print(f'attn output: {np.mean(y.detach().numpy())}')
        x = x + y
        print(f'attn residual output: {np.mean(x.detach().numpy())}')


        y = self.ln_2(x)
        print(f'mlp input: {np.mean(y.detach().numpy())}')
        y = self.mlp(y)
        print(f'mlp output: {np.mean(y.detach().numpy())}')
        x = x + y
        print(f'mlp residual output: {np.mean(x.detach().numpy())}')
        # x = x + self.attention(self.ln_1(x))
        # x = x + self.mlp(self.ln_2(x))
        print("============single block end============\n")
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([])
        for _ in range(layers):
            self.resblocks.append(nn.ModuleList([
                ResidualAttentionBlock(width, heads, attn_mask) 
            ]))


        # self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        for block in  self.resblocks:
            x = nn.Sequential(*block)(x)
        return x
        # return self.resblocks(x)



class CLIP_crate(nn.Module):
    def __init__(self,
                 embed_dim: int, # 512
                  # vision
                 image_resolution: int, # 224
                 vision_layers: Union[Tuple[int, int, int, int], int], # 12
                 vision_width: int, # 768
                 vision_patch_size: int, # 16
                 num_heads: int, #64
                 # text
                 context_length: int, # 77
                 vocab_size: int, # 49408
                 transformer_width: int, # 512
                 transformer_heads: int, # 8
                 transformer_layers: int # 12
                 ):
        super().__init__()
        self.context_length = context_length

        dim_head = vision_width // num_heads

        self.visual = CRATE(image_size=image_resolution,
                patch_size=14,
                num_classes=embed_dim,
                dim=vision_width,
                depth=vision_layers,
                heads=num_heads,
                dim_head=dim_head)
    
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))


    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, return_all=False, csa=False):
        return self.visual(image.type(self.dtype), return_all=return_all, csa=csa)

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        # x = text
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        x = x[:, -1, :]
        x = x @ self.text_projection.T
        return x
        # return x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text

def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):



    image_resolution = 224
    vision_patch_size = 14
    vision_width, vision_layers = get_vision_config('H').values()
    embed_dim = 1024 # vision and text dim 

    
    context_length,vocab_size,transformer_width,transformer_layers = get_text_config['H'].values()
    transformer_heads = transformer_width // 64

    model = CLIP_crate(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)

    model.load_state_dict(state_dict)

    return model.eval()


def get_text_config(model):
   return {
        'vocab_size': 32000,
        'context_length': 32,
        "width": {"Ti": 192, "S": 384, "M": 512, "B": 512, "L": 768, "H": 1024, "g": 1024, "G": 1664, "e": 1792}[model],
        "depth": {"Ti": 12, "S": 12, "M": 12, "B": 12, "L": 12, "H": 24, "g": 24, "G": 48, "e": 56}[model],
        # "mlp_dim": {"Ti": 768, "S": 1536, "M": 2048, "B": 2048, "L": 3072, "H": 4096, "g": 4096, "G": 8192, "e": 15360}[model],
        # "num_heads": {"Ti": 3, "S": 6, "M": 8, "B": 8, "L": 12, "H": 16, "g": 16, "G": 16, "e": 16}[model],
    }

def get_vision_config(model):
 return {
    #   'patch_size': 14,
      "width": {"Ti": 192, "S": 384, "M": 512, "B": 768, "L": 1024, "H": 1280, "g": 1408, "G": 1664, "e": 1792}[model],
      "depth": {"Ti": 12, "S": 12, "M": 12, "B": 12, "L": 24, "H": 32, "g": 40, "G": 48, "e": 56}[model],
    #   "mlp_dim": {"Ti": 768, "S": 1536, "M": 2048, "B": 3072, "L": 4096, "H": 5120, "g": 6144, "G": 8192, "e": 15360}[model],
      "num_heads": {"Ti": 3, "S": 6, "M": 8, "B": 12, "L": 16, "H": 16, "g": 16, "G": 16, "e": 16}[model],
  }

embed_dim = 1024 # vision and text dim 

image_resolution = 224
vision_patch_size = 14
vision_width, vision_layers, num_heads = get_vision_config('H').values()


vocab_size,context_length,transformer_width,transformer_layers = get_text_config('H').values()
transformer_heads = transformer_width // 64

model = CLIP_crate(
    embed_dim,
    image_resolution, vision_layers, vision_width, vision_patch_size, num_heads,
    context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
)

workdir = '/HDD_data_storage_2u_1/jinruiyang/shared_space/code/SCLIP/clip/'
pth_path = workdir + 'converted_checkpoints/pretrain_H14_112_32_32k_2000e_res_mlp_4x_decouple_ft_512m_224_checkpoint.pth'
model_weights = torch.load(pth_path)

def rename_params(model_weights):
    new_params = dict()

    for k,v in model_weights.items():
        if k.startswith('transformer.resblocks'):
            parts = k.split('.')  # Split the string into parts
            parts.insert(3, '0')  # Insert '0' at the desired position (index 3)
            k = '.'.join(parts)  # Rejoin the list into a string with dots
        
        new_params[k] = v
    return new_params

model_weights = rename_params(model_weights)
# for k,v in model_weights.items():
#     print(k,v.shape,np.mean(v.numpy()))

# for k,v in model.state_dict().items():
#     print(k,v.shape,np.mean(v.numpy()))

model.load_state_dict(model_weights)


# for k,v in model.state_dict().items():
#     print(k,v.shape)

#863009793
#863,008,769 jax
total_params = sum(p.numel() for p in model.parameters())
print(f"Total Parameters: {total_params}")



tokenizer = get_tokenizer('hf-hub:UCSC-VLAA/ViT-H-14-CLIPA-datacomp1B')

# import tokenize
bs = 4
img_size = 224
image = torch.ones((bs, 3, img_size, img_size))  # (batch_size, channels, height, width)
text = torch.ones((bs, 32,),dtype=torch.int32)
# text = tokenizer(["a diagram", "a dog", "a cat", "a beignet"], context_length=context_length)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

norms = torch.linalg.norm(image_features, dim=1, keepdim=True)
image_features = image_features / (norms + 1e-8)

norms = torch.linalg.norm(text_features, dim=1, keepdim=True)
text_features = text_features / (norms + 1e-8)

print(f'image_features: {image_features}. shape: {image_features.shape}. mean: {np.mean(image_features.detach().numpy())}')
print(f'text_features: {text_features}. shape: {text_features.shape}. mean: {np.mean(text_features.detach().numpy())}')