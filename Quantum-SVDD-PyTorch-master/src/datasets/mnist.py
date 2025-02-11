from torch.utils.data import random_split, Subset
from PIL import Image
from torchvision.datasets import MNIST
from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import get_target_label_idx, global_contrast_normalization
#from sklearn import datasets
#digits = datasets.load_digits()
#images, labels = digits.data, digits.target
#import numpy as np
import torchvision.transforms as transforms


class MNIST_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class=0):
        super().__init__(root)

        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 10))
        self.outlier_classes.remove(normal_class)

        # 预先计算的最小值和最大值（应用GCN后）来自每类训练数据Pre-computed min and max values (after applying GCN) from train data per class
        min_max = [(-0.8826567065619495, 9.001545489292527),
                   (-0.6661464580883915, 20.108062262467364),
                   (-0.7820454743183202, 11.665100841080346),
                   (-0.7645772083211267, 12.895051191467457),
                   (-0.7253923114302238, 12.683235701611533),
                   (-0.7698501867861425, 13.103278415430502),
                   (-0.778418217980696, 10.457837397569108),
                   (-0.7129780970522351, 12.057777597673047),
                   (-0.8280402650205075, 10.581538445782988),
                   (-0.7369959242164307, 10.697039838804978)]

        # MNIST预处理：GCN（具有L1范数）和最小-最大特征缩放到[0,1]MNIST preprocessing: GCN (with L1 norm) and min-max feature scaling to [0,1]
        transform = transforms.Compose([#transforms.Resize((32, 32)),
                                        transforms.ToTensor(),
                                        transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),
                                        transforms.Normalize([min_max[normal_class][0]],
                                                             [min_max[normal_class][1] - min_max[normal_class][0]])])

        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

        train_set = MyMNIST(root=self.root, train=True, download=True,
                            transform=transform, target_transform=target_transform)
        #print(self.train_set.shape)
        # 将其设置到正常类Subset train_set to normal class
        train_idx_normal = get_target_label_idx(train_set.train_labels.clone().data.cpu().numpy(), self.normal_classes)


        self.train_set = Subset(train_set, train_idx_normal)
        #self.train_set = Subset(self.train_set, range(0, 300))
        subset_size = 300

        # 随机拆分训练集
        self.train_set, _ = random_split(self.train_set, [subset_size, len(self.train_set) - subset_size])

        self.test_set = MyMNIST(root=self.root, train=False, download=True,
                                transform=transform, target_transform=target_transform)
        # 定义要随机采样的样本数量
        subset_size = 300

        # 随机拆分训练集
        self.test_set, _ = random_split(self.test_set, [subset_size, len(self.test_set) - subset_size])

        #self.test_set = Subset(self.test_set, range(0, 300))

        #self.test_set = self.train_set

class MyMNIST(MNIST):#MNIST
    """Torchvision MNIST ."""

    def __init__(self, *args, **kwargs):
        super(MyMNIST, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        """重写MNIST类的原始方法Override the original method of the MNIST class.
        Args:
            index (int): Index
        Returns:
            triple: (image, target, index) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # 这样做是为了与所有其他数据集保持一致doing this so that it is consistent with all other datasets
        # 使用_getitem__;方法的补丁返回PIL Imageclass，同时返回数据样本的索引to return a PIL Imageclass with patch of __getitem__ method to also return the index of a data sample
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index  # 只换了一行only line changed
