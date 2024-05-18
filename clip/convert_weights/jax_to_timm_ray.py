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



# crate
#init_file = 'gs://lxh-jax-us-east1/jinruiyang_ckpt/crate/datacomp1b/crate_L14/finetune/clipa_pretrain_L14_84_32_32k_2000e_res_mlp_4x_decouple_use_huge_text_ft_512m_224_med_v3_256/checkpoint.npz'

#init_file = '/HDD_data_storage_2u_1/jinruiyang/shared_space/files/checkpoint.npz'
# init_file = '/HDD_data_storage_2u_1/jinruiyang/shared_space/files/g14_ft_data1b_336_4_4_from_112_pgd2_2_1b_512b_rerpo_checkpoint.npz'
# init_file = '/HDD_data_storage_2u_1/jinruiyang/shared_space/files/ablation_ftin1k_base_in21k_res_mlp_decouple_x4_mixup_open_warm10_4096_lr1e5_91e_v3_128_checkpoint.npz' # B/32 on 1k
# init_file = '/HDD_data_storage_2u_1/jinruiyang/shared_space/files/B32_ablation_in21k_mlp_nodecouple_x1_mixup_open_warm10_4096_lr5e5_91e_norangaug_no_label_sm_v3_128_checkpoint.npz' # vanilla crate B32 on 21k
# init_file = '/HDD_data_storage_2u_1/jinruiyang/shared_space/files/B32_ablation_in21k_res_mlp_decouple_x4_mixup_open_warm10_4096_lr5e5_91e_norangaug_no_label_sm_v3_128_checkpoint.npz' #  crate alpha B32 on 21k
# init_file = '/HDD_data_storage_2u_1/jinruiyang/shared_space/files/B32_ablation_in21k_mlp_nodecouple_x4_mixup_open_warm10_4096_lr5e5_91e_norangaug_no_label_sm_v3_128_checkpoint.npz' #  4x, nodecouple, no residual,  B32 on 21k
# init_file = '/HDD_data_storage_2u_1/jinruiyang/shared_space/files/B32_ablation_in21k_mlp_decouple_x4_mixup_open_warm10_4096_lr5e5_91e_norangaug_no_label_sm_v3_128_checkpoint.npz' #  4x, decouple, no residual,  B32 on 21k
#init_file = '/HDD_data_storage_2u_1/jinruiyang/shared_space/files/B8_in21k_res_mlp_fixed_decouple_x4_no_mixup_open_warm10_4096_lr5e5_wd01_91e_no_randaug_no_label_sm_v3_256_spot_checkpoint.npz' #  crate alpha B8 on 21k
# init_file = '/HDD_data_storage_2u_1/jinruiyang/shared_space/files/in21k_res_mlp_fixed_decouple_x4_mixup_open_warm10_4096_lr5e5_wd01_dp01_91e_no_randaug_no_labelsm_L32_v3_256_checkpoint.npz' #  crate alpha L32 on 21k

# init_file='gs://lxh_jaxtpu_eu_ckpt/jinruiyang_ckpt/crate_ckpt/ablation/crate_B/32/ablation_ftin1k_base_in21k_res_mlp_decouple_x4_mixup_open_warm10_4096_lr1e5_91e_v3_128/checkpoint.npz' # vanilla crate  B32, 1k gs bucket
# init_file = '/HDD_data_storage_2u_1/jinruiyang/shared_space/files/B32_ablation_ftin1k_base_in21k_res_mlp_decouple_x4_mixup_open_warm10_4096_lr1e5_91e_v3_128_checkpoint.npz' #  vanilla crate  B32, 1k

init_file = '/HDD_data_storage_2u_1/jinruiyang/shared_space/files/L14_in21k_res_mlp_fixed_decouple_x4_mixup_open_warm10_4096_lr5e5_wd01_dp01_91e_no_randaug_no_labelsm_L14_med_v3_256_checkpoint.npz' #  crate alpha L14 on 21k


# init_file = '/HDD_data_storage_2u_1/jinruiyang/shared_space/files/finetune_in1k_res_mlp_fixed_decouple_x4_yesra_mixup_open_warm10_4096_lr1e5_wd01_dp01_91e_L8_load_L8_v3_256_checkpoint.npz' # L/8


prefix = 'params/'
#save_name = 'l16_224_pgd3_3_drop0_mix0_ft_from_data1b_2e.pth'

save_name = 'L14_in21k_res_mlp_fixed_decouple_x4_mixup_open_warm10_4096_lr5e5_wd01_dp01_91e_no_randaug_no_labelsm_L14_med_v3_256_checkpoint.pth'


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
torch_weights['conv1.weight'] = embed_conv_w
torch_weights['conv1.bias'] = (_n2p(w[f'{prefix}embedding/bias']))
# torch_weights['conv1.bias'] = torch_weights['conv1.bias'].flip(dims=[0])
torch_weights['cls_token'] = _n2p(w[f'{prefix}cls'], t=False)
pos_embed_w = _n2p(w[f'{prefix}pos_embedding'], t=False)
torch_weights['pos_embedding'] = pos_embed_w
torch_weights['mlp_head.0.weight']= _n2p(w[f'{prefix}encoder_norm/scale'])
torch_weights['mlp_head.0.bias']=_n2p(w[f'{prefix}encoder_norm/bias'])
torch_weights['mlp_head.1.weight']=_n2p(w[f'{prefix}head/kernel'])
torch_weights['mlp_head.1.bias']=_n2p(w[f'{prefix}head/bias'])


dim = 1024


jax_num_items = 0
for k, v in w.items():
    if 'params' in k:
        print(k)
        jax_num_items += 1
    if 'encoderblock' in k and 'params' in k:
        block_num = k.split('/')[2].split('_')[-1]
        block_prefix = f'{prefix}Transformer/encoderblock_{block_num}/'
        # torch_prefix = f'blocks.{block_num}'
        torch_prefix = f'transformer.layers.{block_num}'

        mha_prefix =f'{block_prefix}MultiHeadDotProductSubspaceAttention_0/'
        ista_prefix = f'{block_prefix}OvercompleteISTABlock_0/'
        # ista_prefix = f'{block_prefix}ISTABlock_0/' # for vanilla crate
        # block_num: 0. block_prefix: params/Transformer/encoderblock_0/. torch_prefix: blocks.0. mha_prefix: params/Transformer/encoderblock_0/MultiHeadDotProductAttention_0/

        # load layernorm
        for i in range(2):
          torch_weights[f'{torch_prefix}.{i}.norm.weight'] = _n2p(w[f'{block_prefix}LayerNorm_{i}/scale'])
          torch_weights[f'{torch_prefix}.{i}.norm.bias'] = _n2p(w[f'{block_prefix}LayerNorm_{i}/bias'])

        # load attention
        torch_weights[f'{torch_prefix}.0.fn.qkv.weight'] = torch.from_numpy(w[f'{mha_prefix}proj/kernel'].reshape(dim,dim).transpose([1,0])) # (768, 12, 64)
        # torch_weights[f'{torch_prefix}.0.fn.qkv.weight'] = torch.from_numpy(w[f'{mha_prefix}proj/kernel'].reshape(768,768)) # (768, 12, 64)

        # load attention projection
        torch_weights[f'{torch_prefix}.0.fn.to_out.0.weight'] = torch.from_numpy(w[f'{mha_prefix}out/kernel'].reshape(dim,dim).transpose([1,0])) # (12, 64, 768) 
        # torch_weights[f'{torch_prefix}.0.fn.to_out.0.weight'] = torch.from_numpy(w[f'{mha_prefix}out/kernel'].reshape(768,768)) # (12, 64, 768) 


        # load ista
        torch_weights[f'{torch_prefix}.1.fn.D'] = torch.from_numpy(w[f'{ista_prefix}D'])
        torch_weights[f'{torch_prefix}.1.fn.D1'] = torch.from_numpy(w[f'{ista_prefix}D1'])
        # for vanilla crate 
        # torch_weights[f'{torch_prefix}.1.fn.D'] = torch.from_numpy(w[f'{ista_prefix}D'])


    
print(f'transfer successfully at {save_name}')
save_dir = '/HDD_data_storage_2u_1/jinruiyang/shared_space/code/SCLIP/clip/converted_checkpoints/'

torch.save(torch_weights, os.path.join(save_dir,save_name))
