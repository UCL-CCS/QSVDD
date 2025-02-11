import logging
import torch.nn as nn
import numpy as np


class BaseNet(nn.Module):
    """所有神经网络的基类Base class for all neural networks."""

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.rep_dim = None  # representation dimensionality, i.e. dim of the last layer

    def forward(self, *input):
        """
        前向传递逻辑Forward pass logic
        :返回：网络输出return: Network output
        """
        raise NotImplementedError

    def summary(self):
        """网络摘要Network summary."""
        net_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in net_parameters])
        self.logger.info('Trainable parameters: {}'.format(params))
        self.logger.info(self)
