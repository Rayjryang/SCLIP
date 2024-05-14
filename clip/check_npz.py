import numpy as np
import os
import matplotlib.pyplot as plt


class Tools:
    def __init__(self, ):
        pass

    def check_npz(self):

        init_file = '/HDD_data_storage_2u_1/jinruiyang/shared_space/files/filtered_pretrain_H14_112_32_32k_2000e_res_mlp_4x_decouple_ft_512m_224_checkpoint.npz'
        w = np.load(init_file, allow_pickle=False, mmap_mode='r')
        for k,v in w.items():
            if 'params/txt' in k:
                print(k,v.shape,np.mean(v))
            # print(k,v.shape,np.mean(v.numpy()))

    
    def check_weight(self):
        torch_res = open('./torch.log','r').readlines()
        jax_res = open('./jax_output','r').readlines()

       
        
        def read_text(path):
            torch_res = open(path,'r').readlines()
            torch_dict = {'ln_1_input': [], 'attn_input': [], 'attn_output': [], 'attn_residual_output': [], 'mlp_input': [], 'mlp_output': [], 'mp_residual_output': [] }
            keys = list(torch_dict.keys())
            index = 0
            for line in torch_res:
                if 'block' in line:
                    continue
                if len(line) == 1:
                    continue
                # line = line.replace('\n','')
                value = line.strip().split(':')[1].strip()
                index = index % 7
                torch_dict[keys[index]].append(float(value))
                index += 1
            return torch_dict
        
        torch_dict = read_text('./torch.log')
        jax_dict = read_text('./jax_output')

        for k,v in torch_dict.items():
            print(len(v))
        
        # 计算误差范围
        error_dict = {}
        for key in torch_dict.keys():
            torch_values = torch_dict[key]
            another_values = jax_dict[key]
            errors = [abs(a - b) for a, b in zip(torch_values, another_values)]
            error_dict[key] = errors

        print(error_dict)

      # 创建绘图
        plt.figure(figsize=(10, 6))

        # 绘制每个键的误差范围
        for key, errors in error_dict.items():
            plt.plot(range(len(errors)), errors, marker='o', label=key)

        # 添加标题和标签
        plt.title('Error Range for Each Key')
        plt.xlabel('layer')
        plt.ylabel('Error Range')
        plt.legend()

        # 保存图像到本地
        plt.savefig('error_ranges.png')



        return


tools = Tools()

tools.check_weight()



