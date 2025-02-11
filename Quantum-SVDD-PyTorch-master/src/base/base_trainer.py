from abc import ABC, abstractmethod
from .base_dataset import BaseADDataset
from .base_net import BaseNet


class BaseTrainer(ABC):
    """Trainer base class."""

    def __init__(self, optimizer_name: str, lr: float, n_epochs: int, lr_milestones: tuple, batch_size: int,
                 weight_decay: float, device: str, n_jobs_dataloader: int):
        super().__init__()
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.n_epochs = n_epochs
        self.lr_milestones = lr_milestones
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device
        self.n_jobs_dataloader = n_jobs_dataloader

    @abstractmethod
    def train(self, dataset: BaseADDataset, net: BaseNet) -> BaseNet:
        """
        实现使用数据集的训练集训练给定网络的训练方法。Implement train method that trains the given network using the train_set of dataset.
        :return: Trained net
        """
        pass

    @abstractmethod
    def test(self, dataset: BaseADDataset, net: BaseNet):
        """
        实施测试方法，评估给定网络上数据集的测试集Implement test method that evaluates the test_set of dataset on the given network.
        """
        pass
