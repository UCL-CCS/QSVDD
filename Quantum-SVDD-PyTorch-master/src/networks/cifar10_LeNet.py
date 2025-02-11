import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
from base.base_net import BaseNet
import matplotlib.pyplot as plt
# Pennylane
import pennylane as qml
from pennylane import numpy as np

# Other tools
import time
import copy

n_qubits = 12
quantum = True
q_depth = 20
max_layers = 21
q_delta = 0.01
dev = qml.device('default.qubit', wires=n_qubits)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
def H_layer(nqubits):
    """Layer of single-qubit Hadamard gates.
    """
    for idx in range(nqubits):
        qml.Hadamard(wires=idx)


def RY_layer(w):
    """Layer of parametrized qubit rotations around the y axis.
    """
    #k=w[:16]
    for idx, element in enumerate(w):

        qml.RY(element, wires=idx)

#def RX_layer(w):
    """Layer of parametrized qubit rotations around the y axis.
    """
    #m = w[16:]
    #for idx, element in enumerate(m):
     #   qml.RX(element, wires=idx)


def entangling_layer(nqubits):
    """Layer of CNOTs followed by another shifted layer of CNOT.
    """
    # In other words it should apply something like :
    # CNOT  CNOT  CNOT  CNOT...  CNOT
    #   CNOT  CNOT  CNOT...  CNOT
    for i in range(0, nqubits - 1, 2):  # Loop over even indices: i=0,2,...N-2
        qml.CNOT(wires=[i, i + 1])
    for i in range(1, nqubits - 1, 2):  # Loop over odd indices:  i=1,3,...N-3
        qml.CNOT(wires=[i, i + 1])


@qml.qnode(dev, interface='torch')
def q_net(q_in, q_weights_flat):
    # Reshape weights
    q_weights = q_weights_flat.reshape(max_layers, n_qubits)

    # Start from state |+> , unbiased w.r.t. |0> and |1>
    H_layer(n_qubits)
    wires = list(range(n_qubits))
    qml.AmplitudeEmbedding(features=q_in, wires=wires, pad_with=0.5)
    # Embed features in the quantum node在量子节点中嵌入功能
    #RY_layer(q_in)
    #RX_layer(q_in)
    # Sequence of trainable variational layers
    for k in range(q_depth):
        entangling_layer(n_qubits)
        RY_layer(q_weights[k + 1])

    # Expectation values in the Z basis
    return [qml.expval(qml.PauliZ(j)) for j in range(n_qubits)]
class Quantumnet(nn.Module):
    def __init__(self):
        super().__init__()
        #self.pre_net = nn.Linear(64, 16)
        self.q_params = nn.Parameter(q_delta * torch.randn(max_layers * n_qubits))

        #self.post_net = nn.Linear(n_qubits, 16)

    def forward(self, input_features):
        #pre_out = self.pre_net(input_features)
        #q_in = input_features
        pre_out = input_features
        q_in = torch.tanh(pre_out) * numpy.pi / 2.0

        # Apply the quantum circuit to each element of the batch, and append to q_out
        q_out = torch.Tensor(0, n_qubits) #q_out = torch.Tensor(0, n_qubits)
        q_out = q_out.to(device)
        for elem in q_in:
            q_out_elem = q_net(elem, self.q_params).float().unsqueeze(0)
            q_out = torch.cat((q_out, q_out_elem))
        #return self.post_net(q_out)
        return q_out#, q_in

class CIFAR10_LeNet(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = 12
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 32, 5, bias=False, padding=2)
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=False, padding=2)
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(64, 128, 5, bias=False, padding=2)
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(128 * 4 * 4, self.rep_dim, bias=False)
        self.qnn = Quantumnet()

    def forward(self, x):
        #x = self.conv1(x)
        #x = self.pool(F.leaky_relu(self.bn2d1(x)))
        #x = self.conv2(x)
        #x = self.pool(F.leaky_relu(self.bn2d2(x)))
        #x = self.conv3(x)
        #x = self.pool(F.leaky_relu(self.bn2d3(x)))
        x = x.view(x.size(0), -1)
        #x = self.fc1(x)
        x = self.qnn(x)
        x = torch.sigmoid(x)
        return x


class CIFAR10_LeNet_Autoencoder(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = 12
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder (must match the Deep SVDD network above)
        self.conv1 = nn.Conv2d(3, 32, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.conv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.conv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(64, 128, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.conv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(128 * 4 * 4, self.rep_dim, bias=False)
        self.bn1d = nn.BatchNorm1d(self.rep_dim, eps=1e-04, affine=False)
        self.qnn = Quantumnet()
        # Decoder
        self.deconv1 = nn.ConvTranspose2d(1, 128, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d4 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d5 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d6 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.deconv4 = nn.ConvTranspose2d(32, 3, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv4.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, x):
        #x = self.conv1(x)
        #x = self.pool(F.leaky_relu(self.bn2d1(x)))
        #x = self.conv2(x)
        #x = self.pool(F.leaky_relu(self.bn2d2(x)))
        #x = self.conv3(x)
        #x = self.pool(F.leaky_relu(self.bn2d3(x)))
        x = x.view(x.size(0), -1)
        #x = self.bn1d(self.fc1(x))
        x = self.qnn(x)
        x = torch.sigmoid(x)
        x = x.view(x.size(0), 1, 3, 4)
        x = F.interpolate(F.leaky_relu(x), scale_factor=(4/3, 1))
        x = F.leaky_relu(x)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn2d4(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn2d5(x)), scale_factor=2)
        x = self.deconv3(x)
        x = F.interpolate(F.leaky_relu(self.bn2d6(x)), scale_factor=2)
        x = self.deconv4(x)
        x = torch.sigmoid(x)
        return x
