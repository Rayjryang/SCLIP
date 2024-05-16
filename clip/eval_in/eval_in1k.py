import numpy as np
import torch
import sys

sys.path.append('../../')

# from tqdm.notebook import tqdm
import torch
from torchvision import transforms
import torchvision
from imagenet_templates import  * 

from clip.vanilla_model import get_crate_clip
import torch.nn.functional as  F
from torchvision.transforms import functional as TF


mean = [0.485 , 0.456 , 0.406]
std = [0.229, 0.224, 0.225 ]
resize_size = (224, 224)  # 例如，将图像调整为 224x224 大小
preprocess = transforms.Compose([
    transforms.Resize(resize_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# 自定义 Resize 操作
class ResizeSmall:
    def __init__(self, smaller_size, method='bilinear', antialias=True):
        self.smaller_size = smaller_size
        self.method = method
        self.antialias = antialias

    def __call__(self, image):
        h, w = image.shape[-2], image.shape[-1]
        ratio = self.smaller_size / min(h, w)
        new_h = int(round(h * ratio))
        new_w = int(round(w * ratio))
        image = image.unsqueeze(0)  # 添加 batch 维度
        image = F.interpolate(image, size=(new_h, new_w), mode=self.method, align_corners=False, antialias=self.antialias)
        image = image.squeeze(0)  # 去掉 batch 维度
        return image

# 自定义 Crop 操作
class CentralCrop:
    def __init__(self, crop_size):
        self.crop_size = self.maybe_repeat(crop_size, 2)

    def maybe_repeat(self, value, n):
        if isinstance(value, (list, tuple)):
            assert len(value) == n, f"Expected {n} values, got {len(value)}"
            return value
        return (value,) * n

    def __call__(self, image):
        h, w = self.crop_size
        _, img_h, img_w = image.shape
        dy = (img_h - h) // 2
        dx = (img_w - w) // 2
        return TF.crop(image, dy, dx, h, w)



# 创建 transforms.Compose
transform = transforms.Compose([
    transforms.ToTensor(),
    ResizeSmall(smaller_size=224),  # 您想要的较小尺寸
    CentralCrop(crop_size=(224,224)),     # 您想要的裁剪尺寸
    transforms.Normalize(mean=mean, std=std)
])


# 设置可用的 CUDA 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()

print(f"Number of available GPUs: {num_gpus}")


model, tokenize = get_crate_clip('cpu')
model = torch.nn.DataParallel(model).cuda()  # 将模型包装在 DataParallel 中

images = torchvision.datasets.ImageNet("/HDD_data_storage_2u_1/jinruiyang/datasets/in1k", split='val', transform=transform)





def zeroshot_classifier(classnames, templates):
    with torch.no_grad():
        zeroshot_weights = []
        # for classname in tqdm(classnames):
        for classname in classnames:
            texts = [template.format(classname) for template in templates] #format with class
            texts = tokenize(texts).cuda() #tokenize
            class_embeddings = model.module.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]
     
zeroshot_weights = zeroshot_classifier(imagenet_classes, imagenet_templates)

loader = torch.utils.data.DataLoader(images, batch_size=128, num_workers=6)


with torch.no_grad():
    top1, top5, n = 0., 0., 0.
    for i, (images, target) in enumerate(loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        
        # predict
        image_features = model.module.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits = 100. * image_features @ zeroshot_weights

        # measure accuracy
        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        top1 += acc1
        top5 += acc5
        n += images.size(0)
        print(f'finished {i} batches')
        # if i >= 20:
        #     break

top1 = (top1 / n) * 100
top5 = (top5 / n) * 100 

print(f"Top-1 accuracy: {top1:.5f}")
print(f"Top-5 accuracy: {top5:.5f}")
     
