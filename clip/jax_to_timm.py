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


#init_file = 'gs://lxh_jaxtpu_eu_ckpt/scaling_adv/configs/adv_attack_mae_sup/large/ft_clip/336_pgd3_3_30e_drop0_mix0_ft_from_laion400m_7e.py/checkpoint.npz'

#init_file = 'gs://lxh_jaxtpu_eu_ckpt/scaling_adv/configs/adv_attack_mae_sup/large/ft_clip/224_pgd3_3_30e_drop0_mix0_ft_from_laion400m_7e.py/checkpoint.npz'

#init_file = 'gs://lxh_jaxtpu_eu_ckpt/scaling_adv/configs/adv_attack_mae_sup/large/ft/224_pgd3_3_30e_drop0_ft_from_in21k_60e_mix05.py/checkpoint.npz'

#init_file = 'gs://lxh_jaxtpu_eu_ckpt/scaling_adv/configs/adv_attack_mae_sup/large/ft_clip/336_pgd3_3_30e_drop0_mix0_ft_from_data1b_2e.py/checkpoint.npz'

#init_file = 'gs://lxh_jaxtpu_eu_ckpt/scaling_adv/configs/adv_attack_mae_sup/large/ft_clip/224_pgd3_3_30e_drop0_mix0_ft_from_data1b_2e.py/checkpoint.npz'

# h14 224 datacomp 1b ft
#init_file = 'gs://lxh_jaxtpu_eu_ckpt/scaling_adv//configs/adv_attack_mae_sup/huge/ft/clip/data1b/data_84_32_pre_training_post_norm_x2_pgd2_2/checkpoint.npz'


# h14 336 datacomp 1b ft
#init_file = 'gs://lxh_jaxtpu_eu_ckpt/scaling_adv//configs/adv_attack_mae_sup/huge/ft/clip/data1b/336_ft_from_84_pgd2_2_post_norm_x2/checkpoint.npz'

#init_file = 'gs://lxh_jaxtpu_eu_ckpt/adv_clip//configs/adv_scale/vit_gaint/ablation/ft/clip/g14_ft_data1b_224_3_4_from_112_pgd2_2_1b_512b/checkpoint.npz'

#init_file = 'gs://lxh_jaxtpu_eu_ckpt/adv_clip//configs/adv_scale/vit_large/ablation/ft/clip/g14_ft_data1b_336_4_4_from_112_pgd2_2_1b_512b_rerpo/checkpoint.npz'

# crate
#init_file = 'gs://lxh-jax-us-east1/jinruiyang_ckpt/crate/datacomp1b/crate_L14/finetune/clipa_pretrain_L14_84_32_32k_2000e_res_mlp_4x_decouple_use_huge_text_ft_512m_224_med_v3_256/checkpoint.npz'

#init_file = '/HDD_data_storage_2u_1/jinruiyang/shared_space/files/checkpoint.npz'
# init_file = '/HDD_data_storage_2u_1/jinruiyang/shared_space/files/g14_ft_data1b_336_4_4_from_112_pgd2_2_1b_512b_rerpo_checkpoint.npz'
init_file = '/HDD_data_storage_2u_1/jinruiyang/shared_space/files/pretrain_H14_112_32_32k_2000e_res_mlp_4x_decouple_ft_512m_224_checkpoint.npz'
prefix = 'params/'

#save_name = 'l16_224_pgd3_3_drop0_mix0_ft_from_data1b_2e.pth'

save_name = 'g14_ft_data1b_336_4_4_from_112_pgd2_2_1b_512b.pth'

#save_name = 'g14_ft_data1b_224_3_4_from_112_pgd2_2_1b_512b.pth'

#save_name = 'h14_ft_data1b_224_3_4_from_112_pgd2_2_1b_512b.pth'

#save_name = 'h14_ft_data1b_336_3_4_from_112_pgd2_2_1b_512b.pth'


def npload(fname):
  """Loads `fname` and returns an np.ndarray or dict thereof."""
  # Load the data; use local paths directly if possible:
  if os.path.exists(fname):
    loaded = np.load(fname, allow_pickle=False)
  else:
    # For other (remote) paths go via gfile+BytesIO as np.load requires seeks.
    with gfile.GFile(fname, "rb") as f:
      data = f.read()
    loaded = np.load(io.BytesIO(data), allow_pickle=False)

  # Support loading both single-array files (np.save) and zips (np.savez).
  if isinstance(loaded, np.ndarray):
    return loaded
  else:
    return dict(loaded)

w = npload(init_file)

for k,v in w.items():
    print(k,' ',v.shape)

# exit()

torch_weights = {}



embed_conv_w =  _n2p(w[f'{prefix}embedding/kernel'])
torch_weights['patch_embed.proj.weight'] = embed_conv_w
torch_weights['patch_embed.proj.bias'] = (_n2p(w[f'{prefix}embedding/bias']))
torch_weights['cls_token'] = _n2p(w[f'{prefix}cls'], t=False)
pos_embed_w = _n2p(w[f'{prefix}pos_embedding'], t=False)
torch_weights['pos_embed'] = pos_embed_w
torch_weights['fc_norm.weight']= _n2p(w[f'{prefix}encoder_norm/scale'])
torch_weights['fc_norm.bias']=_n2p(w[f'{prefix}encoder_norm/bias'])
torch_weights['head.weight']=_n2p(w[f'{prefix}head/kernel'])
torch_weights['head.bias']=_n2p(w[f'{prefix}head/bias'])




jax_num_items = 0
for k, v in w.items():
    if 'params' in k:
        print(k)
        jax_num_items += 1
    if 'encoderblock' in k and 'params' in k:
        block_num = k.split('/')[2].split('_')[-1]
        block_prefix = f'{prefix}Transformer/encoderblock_{block_num}/'
        torch_prefix = f'blocks.{block_num}'
        mha_prefix =f'{block_prefix}MultiHeadDotProductAttention_0/'
        torch_weights[f'{torch_prefix}.norm1.weight'] = _n2p(w[f'{block_prefix}LayerNorm_0/scale'])
        torch_weights[f'{torch_prefix}.norm1.bias'] = _n2p(w[f'{block_prefix}LayerNorm_0/bias'])
        torch_weights[f'{torch_prefix}.attn.qkv.weight'] = torch.cat([
                _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')])
        torch_weights[f'{torch_prefix}.attn.qkv.bias'] = torch.cat([
                _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')])
        torch_weights[f'{torch_prefix}.attn.proj.weight'] = _n2p(w[f'{mha_prefix}out/kernel']).flatten(1)
        torch_weights[f'{torch_prefix}.attn.proj.bias'] = _n2p(w[f'{mha_prefix}out/bias'])
        for r in range(2):
            torch_weights[f'{torch_prefix}.mlp.fc{r + 1}.weight']= _n2p(w[f'{block_prefix}MlpBlock_0/Dense_{r}/kernel'])
            torch_weights[f'{torch_prefix}.mlp.fc{r + 1}.bias'] = _n2p(w[f'{block_prefix}MlpBlock_0/Dense_{r}/bias'])
            ##layer_scale
            #torch_weights[f'{torch_prefix}.ls{r + 1}.gamma'] = _n2p(w[f'{block_prefix}ls{r+1}/ls{r+1}'])
        torch_weights[f'{torch_prefix}.norm2.weight']=_n2p(w[f'{block_prefix}LayerNorm_1/scale'])
        torch_weights[f'{torch_prefix}.norm2.bias']=_n2p(w[f'{block_prefix}LayerNorm_1/bias'])

print(f'transfer successfully at {save_name}')
torch.save(torch_weights, save_name)
