import json
import torch

from base.base_dataset import BaseADDataset
from networks.main import build_network, build_autoencoder
from optim.deepSVDD_trainer import DeepSVDDTrainer
from optim.ae_trainer import AETrainer


class DeepSVDD(object):
    """Deep SVDD方法的一个类。

    属性:
        objective: 指定深度SVDD目标的字符串（“一个类”或“软边界”）.
        nu: 深度SVDD超参数 nu (must be 0 < nu <= 1).
        R: 超球体半径 R.
        c: 超球体中心 c.
        net_name: 一个字符串，指示要使用的神经网络的名称.
        net: 神经网络 \phi.
        ae_net: 对应于网络权重预训练的\phi的自动编码器网络。
        trainer: DeepSVDDTrainer 训练一个 Deep SVDD 模型.
        optimizer_name: 一个字符串，指示用于训练深层SVDD网络的优化器.
        ae_trainer: AETrainer 在训练前训练自动编码器.
        ae_optimizer_name: 一个字符串，指示用于预训练自动编码器的优化器.
        results: 保存结果.
    """

    def __init__(self, objective: str = 'one-class', nu: float = 0.1):
        """Inits DeepSvd具有两个目标之一和超参数nu."""

        assert objective in ('one-class', 'soft-boundary'), "目标必须是“一类”或“软边界”'."
        self.objective = objective
        assert (0 < nu) & (nu <= 1), "For hyperparameter nu, it must hold: 0 < nu <= 1."
        self.nu = nu
        self.R = 0.0  # hypersphere radius R
        self.c = None  # hypersphere center c

        self.net_name = None
        self.net = None  # neural network \phi

        self.trainer = None
        self.optimizer_name = None

        self.ae_net = None  # autoencoder network for pretraining
        self.ae_trainer = None
        self.ae_optimizer_name = None

        self.results = {
            'train_time': None,
            'test_auc': None,
            'test_time': None,
            'test_scores': None,
        }

    def set_network(self, net_name):
        """建造神经网络 \phi."""
        self.net_name = net_name
        self.net = build_network(net_name)

    def train(self, dataset: BaseADDataset, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 50,
              lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
              n_jobs_dataloader: int = 0):
        """在训练数据上训练深度SVDD模型 Trains the Deep SVDD model on the training data."""

        self.optimizer_name = optimizer_name
        self.trainer = DeepSVDDTrainer(self.objective, self.R, self.c, self.nu, optimizer_name, lr=lr,
                                       n_epochs=n_epochs, lr_milestones=lr_milestones, batch_size=batch_size,
                                       weight_decay=weight_decay, device=device, n_jobs_dataloader=n_jobs_dataloader)
        # Get the model
        self.net = self.trainer.train(dataset, self.net)
        self.R = float(self.trainer.R.cpu().data.numpy())  # get float
        self.c = self.trainer.c.cpu().data.numpy().tolist()  # get list
        self.results['train_time'] = self.trainer.train_time

    def test(self, dataset: BaseADDataset, device: str = 'cuda', n_jobs_dataloader: int = 0):
        """在测试集上实验数据."""

        if self.trainer is None:
            self.trainer = DeepSVDDTrainer(self.objective, self.R, self.c, self.nu,
                                           device=device, n_jobs_dataloader=n_jobs_dataloader)

        self.trainer.test(dataset, self.net)
        # Get results
        self.results['test_auc'] = self.trainer.test_auc
        self.results['test_time'] = self.trainer.test_time
        self.results['test_scores'] = self.trainer.test_scores

    def pretrain(self, dataset: BaseADDataset, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 100,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0):
        """通过自动编码器预训练深度SVDD网络的权重\phi."""

        self.ae_net = build_autoencoder(self.net_name)
        self.ae_optimizer_name = optimizer_name
        self.ae_trainer = AETrainer(optimizer_name, lr=lr, n_epochs=n_epochs, lr_milestones=lr_milestones,
                                    batch_size=batch_size, weight_decay=weight_decay, device=device,
                                    n_jobs_dataloader=n_jobs_dataloader)
        self.ae_net = self.ae_trainer.train(dataset, self.ae_net)
        self.ae_trainer.test(dataset, self.ae_net)
        self.init_network_weights_from_pretraining()

    def init_network_weights_from_pretraining(self):
        """根据预训练自动编码器的编码器权重初始化深度SVDD网络权重。"""

        net_dict = self.net.state_dict()
        ae_net_dict = self.ae_net.state_dict()

        # Filter out decoder network keys
        ae_net_dict = {k: v for k, v in ae_net_dict.items() if k in net_dict}
        # Overwrite values in the existing state_dict
        net_dict.update(ae_net_dict)
        # Load the new state_dict
        self.net.load_state_dict(net_dict)

    def save_model(self, export_model, save_ae=True):
        """保存深度SVDD模型以导出_模型Save Deep SVDD model to export_model."""

        net_dict = self.net.state_dict()
        ae_net_dict = self.ae_net.state_dict() if save_ae else None

        torch.save({'R': self.R,
                    'c': self.c,
                    'net_dict': net_dict,
                    'ae_net_dict': ae_net_dict}, export_model)
        print('R', self.R)
        print('c', self.c)

    def load_model(self, model_path, load_ae=False):
        """从模型路径加载深度SVDD模型Load Deep SVDD model from model_path."""

        model_dict = torch.load(model_path)

        self.R = model_dict['R']
        self.c = model_dict['c']
        self.net.load_state_dict(model_dict['net_dict'])
        if load_ae:
            if self.ae_net is None:
                self.ae_net = build_autoencoder(self.net_name)
            self.ae_net.load_state_dict(model_dict['ae_net_dict'])

    def save_results(self, export_json):
        """将结果dict保存到JSON文件Save results dict to a JSON-file."""
        with open(export_json, 'w') as fp:
            json.dump(self.results, fp)
