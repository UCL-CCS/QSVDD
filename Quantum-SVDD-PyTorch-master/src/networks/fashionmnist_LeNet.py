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
#Define QNN
#from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
#from qiskit_machine_learning.neural_networks import CircuitQNN, TwoLayerQNN
#from qiskit_machine_learning.connectors import TorchConnector
#qi = QuantumInstance(Aer.get_backend("aer_simulator_statevector"))
#feature_map = ZZFeatureMap(2)
#ansatz = RealAmplitudes(2, reps=1)
##REMEMBER TO SET input_gradients=True FOR ENABLING HYBRID GRADIENT BACKPROP
#qnn4 = TwoLayerQNN(2, feature_map, ansatz, input_gradients=True, exp_val=AerPauliExpectation(), quantum_instance=qi)
n_qubits = 10
quantum = True
q_depth = 20
max_layers = 21
q_delta = 0.01
dev = qml.device('default.qubit', wires=n_qubits)
#dev = qml.device("forest.qvm", device="4q", noisy=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
#device = torch.device("cuda:0")
#量子卷积
#import tensorflow as tf
#from tensorflow import keras
#from pennylane.templates import RandomLayers
#n_layers = 1    # Number of random layers
#np.random.seed(0)           # Seed for NumPy random number generator
#tf.random.set_seed(0)       # Seed for TensorFlow random number generator

#dev2 = qml.device("default.qubit", wires=4)
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Random circuit parameters
#rand_params = np.random.uniform(high=2 * np.pi, size=(n_layers, 4))

#@qml.qnode(dev)
#def circuit(phi):
    # Encoding of 4 classical input values
    #for j in range(4):
     #   qml.RY(np.pi * phi[j], wires=j)

    # Random quantum circuit
    #RandomLayers(rand_params, wires=list(range(4)))

    # Measurement producing 4 classical output values
    #return [qml.expval(qml.PauliZ(j)) for j in range(4)]

#def quanv(image):
 #   """Convolves the input image with many applications of the same quantum circuit."""
  #  out = np.zeros((4,14, 14))

    # Loop over the coordinates of the top-left pixel of 2X2 squares
   # for j in range(0, 28, 2):
    #    for k in range(0, 28, 2):
            # Process a squared 2x2 region of the image with a quantum circuit
     #       q_results = circuit(
                #[
                 #   image[0,j, k],
                  #  image[0,j, k + 1],
                   # image[0,j + 1, k],
                    #image[0,j + 1, k + 1]
                #]
            #)
            # Assign expectation values to different channels of the output pixel (j/2, k/2)
            #for c in range(4):
             #   out[c,j // 2, k // 2] = q_results[c]
    #return out
#class quanvnet(nn.Module):
 #   def __init__(self):
  #      super().__init__()

   # def forward(self, input_features):
    #    q_train_images = []
        #pre_out = [self.input_features]
     #   for idx, img in enumerate(input_features):
      #    q_train_images.append(quanv(img))
       # x = np.asarray(q_train_images)
        #return x

#量子替代全连接
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
    #for j in range(0, nqubits - 1, 1):
    for i in range(0, nqubits - 1, 2):  # Loop over even indices: i=0,2,...N-2
        qml.CNOT(wires=[i, i+1])
    for i in range(1, nqubits - 1, 2):  # Loop over odd indices:  i=1,3,...N-3
        qml.CNOT(wires=[i, i+1])

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

class FashionMNIST_LeNet(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = n_qubits #16
        self.pool = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(7, 7)
        self.conv1 = nn.Conv2d(1, 8, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(8, 2, 5, bias=False, padding=2)#self.conv2 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(2, eps=1e-04, affine=False)#self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        #self.quanv=quanvnet()
        self.fc1 = nn.Linear(2*7*7 , 32, bias=False)#self.fc1 = nn.Linear(2 * 7 * 7, self.rep_dim, bias=False)
        self.fc2 = nn.Linear(32, 32, bias=False)#16
        #self.fc3 = nn.Linear(16, 16, bias=False)
        #self.qnn = TorchConnector(qnn4)
        self.qnn = Quantumnet()

    def forward(self, x):
        #x = self.quanv(x)
        #x = self.conv1(x)
        #x = self.pool(F.leaky_relu(self.bn1(x)))
        #x = self.conv2(x)
        #x = self.pool(F.leaky_relu(self.bn2(x)))
        #x = self.pool2(x)
        x = x.view(x.size(0), -1)
        #x=torch.tensor(x)
        #x = x.view(-1,2*7*7)
        #x = x.cuda()
        #x = x.to(torch.float32)
        #x = self.fc1(x)
        #x = self.fc2(x)
        x = self.qnn(x)
        #print('features for qnn are:', y)
        #x = self.fc3(x)
        return x


class FashionMNIST_LeNet_Autoencoder(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = n_qubits#16
        self.pool = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(7, 7)
        # 编码器（必须与上面的深度SVDD网络匹配）Encoder (must match the Deep SVDD network above)
        self.conv1 = nn.Conv2d(1, 8, 5, bias=False, padding=2)#第一个卷积层
        self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)#第一个bn层：BN层能够让网络更快收敛、而且对不同的学习率鲁棒
        self.conv2 = nn.Conv2d(8, 2, 5, bias=False, padding=2)#第二个卷积层self.conv2 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(2, eps=1e-04, affine=False)#self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        #self.qnn = TorchConnector(qnn4)
        self.fc1 = nn.Linear(2*7*7 , 32, bias=False)#全连接层（输入，输出，bias）self.fc1 = nn.Linear(4 * 7 * 7, self.rep_dim, bias=False)
        #self.qnn = TorchConnector(qnn4)
        self.fc2 = nn.Linear(32, 32, bias=False)#16
        self.qnn = Quantumnet()

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(1, 4, 5, bias=False, padding=2)
        self.bn3 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(4, 8, 5, bias=False, padding=3)
        self.bn4 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(8, 1, 5, bias=False, padding=2)

    def forward(self, x):
        #x = self.conv1(x)
        #x = self.pool(F.leaky_relu(self.bn1(x)))
        #x = self.conv2(x)
        #x = self.pool(F.leaky_relu(self.bn2(x)))
        #x = self.pool2(x)
        x = x.view(x.size(0), -1)
        #x = self.fc1(x)
        x = self.qnn(x)
        #x = self.fc2(x)
        #x = self.fc3(x)
        x = x.view(x.size(0), 1, 2, 5)
        #x = x.view(x.size(0), int(self.rep_dim / 16), 4, 4) 4:0122
        x = F.interpolate(F.leaky_relu(x), scale_factor=(4, 8/5))
        #x = F.interpolate(F.leaky_relu(x), scale_factor=2)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn3(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn4(x)), scale_factor=2)
        x = self.deconv3(x)
        x = torch.sigmoid(x)

        return x
