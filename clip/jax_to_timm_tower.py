import io

import numpy as np
#import flax
import torch
from tensorflow.python.platform import gfile
import os

def _n2p(w, t=True):
    if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
        w = w.flatten()
    if t:
        if w.ndim == 4:
            w = w.transpose([3, 2, 0, 1])
        elif w.ndim == 3:
            w = w.transpose([2, 0, 1])
        elif w.ndim == 2:
            w = w.transpose([1, 0])
    return torch.from_numpy(w)

init_file = '/HDD_data_storage_2u_1/jinruiyang/shared_space/files/pretrain_H14_112_32_32k_2000e_res_mlp_4x_decouple_ft_512m_224_checkpoint.npz'


save_name = 'pretrain_H14_112_32_32k_2000e_res_mlp_4x_decouple_ft_512m_224_checkpoint.pth'


def npload(fname):

  loaded = np.load(fname, allow_pickle=False, mmap_mode='r')

  return dict(loaded)

w = npload(init_file)

# for k,v in w.items():
#     print(k,' ',v.shape)

# 保存新的npz文件
# np.savez('/HDD_data_storage_2u_1/jinruiyang/shared_space/files/filtered_pretrain_H14_112_32_32k_2000e_res_mlp_4x_decouple_ft_512m_224_checkpoint.npz', **filtered_data)


torch_weights = {}

prefix = 'params/img/'

embed_conv_w =  _n2p(w[f'{prefix}embedding/kernel'])
torch_weights['visual.conv1.weight'] = embed_conv_w
# torch_weights['visual.conv1.bias'] = (_n2p(w[f'{prefix}embedding/bias']))
torch_weights['visual.cls_token'] = _n2p(w[f'{prefix}cls'], t=False)
pos_embed_w = _n2p(w[f'{prefix}pos_embedding'], t=False)
pos_embed_w = np.squeeze(pos_embed_w, axis=0) # [1, n+1, dim] -> [n+1,dim]
torch_weights['visual.pos_embedding'] = pos_embed_w
torch_weights['visual.mlp_head.0.weight']= _n2p(w[f'{prefix}encoder_norm/scale'])
torch_weights['visual.mlp_head.0.bias']=_n2p(w[f'{prefix}encoder_norm/bias'])
torch_weights['visual.mlp_head.1.weight']=_n2p(w[f'{prefix}head/kernel'])
# torch_weights['visual.mlp_head.1.bias']=_n2p(w[f'{prefix}head/bias'])



jax_num_items = 0
for k, v in w.items():
    if 'params/img' in k:
        print(k)
        jax_num_items += 1
    if 'encoderblock' in k and 'params' in k:
        # block_num = k.split('/')[2].split('_')[-1] # only for img model
        block_num = k.split('/')[3].split('_')[-1] # for clip
        
        block_prefix = f'{prefix}Transformer/encoderblock_{block_num}/'
        # torch_prefix = f'blocks.{block_num}'
        torch_prefix = f'visual.transformer.layers.{block_num}'

        mha_prefix =f'{block_prefix}MultiHeadDotProductSubspaceAttention_0/'
        ista_prefix = f'{block_prefix}OvercompleteISTABlock_0/'
        # block_num: 0. block_prefix: params/Transformer/encoderblock_0/. torch_prefix: blocks.0. mha_prefix: params/Transformer/encoderblock_0/MultiHeadDotProductAttention_0/
       
        # load layernorm
        for i in range(2):
          torch_weights[f'{torch_prefix}.{i}.norm.weight'] = _n2p(w[f'{block_prefix}LayerNorm_{i}/scale'])
          torch_weights[f'{torch_prefix}.{i}.norm.bias'] = _n2p(w[f'{block_prefix}LayerNorm_{i}/bias'])

        # load attention
        torch_weights[f'{torch_prefix}.0.fn.qkv.weight'] = torch.from_numpy(w[f'{mha_prefix}proj/kernel'].reshape(1280,1280).transpose([1,0])) # (768, 12, 64)
        # torch_weights[f'{torch_prefix}.0.fn.qkv.weight'] = torch.from_numpy(w[f'{mha_prefix}proj/kernel'].reshape(768,768)) # (768, 12, 64)

        # load attention projection
        torch_weights[f'{torch_prefix}.0.fn.to_out.0.weight'] = torch.from_numpy(w[f'{mha_prefix}out/kernel'].reshape(1280,1280).transpose([1,0])) # (12, 64, 768) 
        # torch_weights[f'{torch_prefix}.0.fn.to_out.0.weight'] = torch.from_numpy(w[f'{mha_prefix}out/kernel'].reshape(768,768)) # (12, 64, 768) 


        # load ista
        torch_weights[f'{torch_prefix}.1.fn.D'] = torch.from_numpy(w[f'{ista_prefix}D'])
        torch_weights[f'{torch_prefix}.1.fn.D1'] = torch.from_numpy(w[f'{ista_prefix}D1'])

    
print(f'vision parameter transfer successfully at {save_name}')
# torch.save(torch_weights, save_name)


################## logit_scale ##################

torch_weights['logit_scale'] = _n2p(w['params/t'])[0]

################## load text encoder ##################

prefix = 'params/txt/'

torch_weights['positional_embedding'] = _n2p(w[f'{prefix}pos_embedding'], t=False).squeeze(0)
torch_weights['text_projection'] = _n2p(w[f'{prefix}head/kernel'])
torch_weights['token_embedding.weight'] = _n2p(w[f'{prefix}Embed_0/embedding'], t=False)
torch_weights['ln_final.weight'] =  _n2p(w[f'{prefix}encoder_norm/scale'])
torch_weights['ln_final.bias'] =  _n2p(w[f'{prefix}encoder_norm/bias'])


jax_num_items = 0
for k, v in w.items():
    if 'params/txt' in k:
        print(k)
        jax_num_items += 1
    if 'encoderblock' in k and 'params/txt' in k:
        #jax
        block_num = k.split('/')[3].split('_')[-1] # for clip
        block_prefix = f'{prefix}Transformer/encoderblock_{block_num}/' #  'params/txt/'
        mha_prefix = block_prefix + f'MultiHeadDotProductAttention_0/'
        # torch
        torch_prefix = f'transformer.resblocks.{block_num}'
        # converting
        # transformer.resblocks.0.ln_1.weight torch.Size([1024]) 0.4317572
        # transformer.resblocks.0.ln_1.bias torch.Size([1024]) -0.007854426
        # layer norm
        torch_weights[f'{torch_prefix}.ln_1.weight'] = _n2p(w[f'{block_prefix}LayerNorm_0/scale'])
        torch_weights[f'{torch_prefix}.ln_1.bias'] =  _n2p(w[f'{block_prefix}LayerNorm_0/bias'])              
        # mha
        torch_weights[f'{torch_prefix}.attn.in_proj_weight'] = torch.cat([
                _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')])
        torch_weights[f'{torch_prefix}.attn.in_proj_bias'] = torch.cat([
                _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')])

        torch_weights[f'{torch_prefix}.attn.out_proj.weight'] = _n2p(w[f'{mha_prefix}out/kernel']).flatten(1)
        torch_weights[f'{torch_prefix}.attn.out_proj.bias'] = _n2p(w[f'{mha_prefix}out/bias'])
        # layer norm
        torch_weights[f'{torch_prefix}.ln_2.weight'] = _n2p(w[f'{block_prefix}LayerNorm_1/scale'])
        torch_weights[f'{torch_prefix}.ln_2.bias'] =  _n2p(w[f'{block_prefix}LayerNorm_1/bias'])              
        #mlp
        torch_weights[f'{torch_prefix}.mlp.c_fc.weight'] = _n2p(w[f'{block_prefix}MlpBlock_0/Dense_0/kernel'])
        torch_weights[f'{torch_prefix}.mlp.c_fc.bias'] = _n2p(w[f'{block_prefix}MlpBlock_0/Dense_0/bias'])
        torch_weights[f'{torch_prefix}.mlp.c_proj.weight'] = _n2p(w[f'{block_prefix}MlpBlock_0/Dense_1/kernel'])
        torch_weights[f'{torch_prefix}.mlp.c_proj.bias'] = _n2p(w[f'{block_prefix}MlpBlock_0/Dense_1/bias'])


for k,v in torch_weights.items():
    print(k,v.shape)

print(f'text parameter transfer successfully at {save_name}')
torch.save(torch_weights, os.path.join('./converted_checkpoints',save_name))