import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class Reader(Dataset):
    def __init__(self, data_path, img_size=224):
        """
        数据读取器
        :param data_path: 数据集所在路径
        """
        super().__init__()
        self.data_path = data_path
        self.img_paths = []
        self.labels = []

        self.img_size = img_size

        with open(self.data_path, "r", encoding="utf-8") as f:
            self.info = f.readlines()
        for img_info in self.info:
            img_path, label = img_info.strip().split('\t')
            self.img_paths.append(img_path)
            self.labels.append(int(label))


    def __getitem__(self, index):
        """
        获取一组数据
        :param index: 文件索引号
        :return:
        """
        # 通过index从self.img_paths与self.labels中获得对应图片路径和label值，并读取图片
        # 如果图像非RGB模式，转为RGB模式，可以用img.mode和img.convert来查看和更改RGB模式，其中img为Image库中的Image对象。
        # 图像大小改为(224,224)，并改变图片通道，由HWC转换为CHW

        img_path = self.img_paths[index]
        img = Image.open(img_path)

        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        img = np.array(img).astype('float32')
        img = img.transpose((2, 0, 1)) / 255

        # if using nn.CrossEntropyLoss, the label should be a scalar, e.g. 4 instaed of [4] or [0,0,0,0,1,0,...]
        label = self.labels[index]
        # label = np.array([label], dtype="int64")

        return img, label

    def print_sample(self, index: int = 0):
        print("文件名", self.img_paths[index], "\t标签值", self.labels[index])

    def __len__(self):
        return len(self.img_paths)
