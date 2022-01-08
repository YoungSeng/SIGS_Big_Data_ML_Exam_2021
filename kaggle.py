# 20210927
# Official entry code

# import torch
#
# print(torch.__version__)
# print(torch.cuda.is_available())
# print(torch.cuda.get_device_name(0))

# 20210928 迭代300次模型：Best Prec@1: 26.865%
# 跑通 baseline 300 epoch

import os  # 打开文件夹用
from torchvision import transforms
from PIL import Image
import torch
# from model import resnet34
# from model import ResNet50
# from model import ResNet101
# from model import ResNet152
# from EfficientNet import efficientnet_b0
# from DenseNet import densenet121
from DenseNet import densenet161

import config as cfg

# model = resnet34()
# model = ResNet50()
# model = ResNet101()
# model = ResNet152()
# model = efficientnet_b0()
model = densenet161()

checkpoint = torch.load(
    "/ceph/home/yangsc21/kaggle/food_cls/ckpt/Densenet161/model_best_densenet161.pth.tar")  # 最好模型的位置
model.load_state_dict(checkpoint['state_dict_model'])
model.eval()
model = model.cuda(cfg.gpu)

dir = "/ceph/home/yangsc21/kaggle/data/food/test/test/"

list_test_image_path = []
for file in os.listdir(dir):  # 遍历dir文件夹
    list_test_image_path.append(file)  # len(list_test_image_path) = 10100

print('Test image loaded! length of test set is {0}'.format(len(list_test_image_path)))

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

output_result = '/ceph/home/yangsc21/kaggle/result.txt'
cnt = 0

with open(output_result, 'w', encoding='utf-8') as result:
    with torch.no_grad():
        for i in list_test_image_path:
            path = dir + i
            if cfg.gpu is not None:
                with open(path, 'rb') as f:
                    sample = Image.open(f).convert('RGB')
                    sample = transform_test(sample)
                images = sample.cuda(cfg.gpu, non_blocking=True)
                output = model(images.unsqueeze(0))
                _, predicted = output.max(1)
                # print(predicted.cpu().tolist()[0])
                result.write(i + ',' + str(predicted.cpu().tolist()[0]) + '\n')
                cnt += 1
                if cnt % 200 == 0:
                    print(str(cnt / 200) + ' of 50')
