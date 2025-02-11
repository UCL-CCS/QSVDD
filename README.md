# PyTorch Implementation of Quantum SVDD
This repository provides a [PyTorch](https://pytorch.org/) implementation of the *Quantum SVDD* method presented in our 2024 paper ”A Parameter-Efficient Quantum Anomaly Detection Method on a Superconducting Quantum Processor”.


## Citation and Contact
You can find a PDF of the A Parameter-Efficient Quantum Anomaly Detection Method on a Superconducting Quantum Processor paper at 
[[http://](https://arxiv.org/abs/2412.16867)].

If you use our work, please also cite the paper:
```
https://arxiv.org/abs/2412.16867
```

If you would like to get in touch, please contact [maida.wang.24@ucl.ac.uk](mailto:maida.wang.24@ucl.ac.uk).


## Abstract
Quantum machine learning has gained attention for its potential to address computational challenges. However, whether those algorithms can effectively solve practical problems and outperform their classical counterparts, especially on current quantum hardware, remains a critical question. In this work, we propose a novel quantum machine learning method, called Quantum Support Vector Data Description (QSVDD), for practical image anomaly detection, which aims to achieve both parameter efficiency and superior accuracy compared to classical models.  Emulation results indicate that QSVDD demonstrates favourable recognition capabilities compared to classical baselines, achieving an average accuracy of over 90% on benchmarks with significantly fewer trainable parameters. Theoretical analysis confirms that QSVDD has a comparable expressivity to classical counterparts while requiring only a fraction of the parameters.
Furthermore, we demonstrate the first implementation of a quantum anomaly detection method for general image datasets on a superconducting quantum processor. Specifically, we achieve an accuracy of over 80% with only 16 parameters on the device, providing initial evidence of QSVDD's practical viability in the noisy intermediate-scale quantum era and highlighting its significant reduction in parameter requirements.

![QSVDD on processor(only)](https://github.com/user-attachments/assets/ba85a024-9de5-4c66-a667-13b902d57039)




## Installation
This code is written in `Python 3.8` and requires the packages listed in `requirements.txt`.

Clone the repository to your local machine and directory of choice:
```
[git clone https://github.com/UCL-CCS/QSVDD.git]
```

To run the code, we recommend setting up a virtual environment, e.g. using `virtualenv` or `conda`:

### `virtualenv`
```
# pip install virtualenv
cd <path-to-Quantum-SVDD-directory>
virtualenv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

### `conda`
```
cd <path-to-Quantum-SVDD-directory>
conda create --name myenv
source activate myenv
while read requirement; do conda install -n myenv --yes $requirement; done < requirements.txt
```


## Running simulations

We currently have implemented the MNIST ([http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)) and 
CIFAR-10 ([https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)) datasets and 
FashionMNIST datasets ([https://github.com/zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist)).

Have a look into `main.py` for all possible arguments and options.

### MNIST example
```
cd <path-to-Quantum-SVDD-directory>

# activate virtual environment
source myenv/bin/activate  # or 'source activate myenv' for conda

# create folder for experimental output
mkdir log/mnist_test

# change to source directory
cd src

# run simulation
python main.py mnist mnist_LeNet ../log/mnist_test ../data --objective one-class --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001 --ae_n_epochs 150 --ae_lr_milestone 50 --ae_batch_size 200 --ae_weight_decay 0.5e-3 --normal_class 3;
```
This example trains a Quantum SVDD model where digit 3 (`--normal_class 3`) is considered to be the normal class. Autoencoder
pretraining is used for parameter initialization.

### FashionMNIST example
```
cd <path-to-Quantum-SVDD-directory>

# activate virtual environment
source myenv/bin/activate  # or 'source activate myenv' for conda

# create folder for experimental output
mkdir log/FashionMNIST_test

# change to source directory
cd src

# run simulation
python main.py FashionMNIST FashionMNIST_LeNet ../log/FashionMNIST_test ../data --objective one-class --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001 --ae_n_epochs 350 --ae_lr_milestone 250 --ae_batch_size 200 --ae_weight_decay 0.5e-6 --normal_class 3;
```
This example trains a Quantum SVDD model where cloth (`--normal_class 3`) is considered to be the normal class. 
Autoencoder pretraining is used for parameter initialization.

### CIFAR-10 example
```
cd <path-to-Quantum-SVDD-directory>

# activate virtual environment
source myenv/bin/activate  # or 'source activate myenv' for conda

# create folder for experimental output
mkdir log/cifar10_test

# change to source directory
cd src

# run simulation
python main.py cifar10 cifar10_LeNet ../log/cifar10_test ../data --objective one-class --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001 --ae_n_epochs 350 --ae_lr_milestone 250 --ae_batch_size 200 --ae_weight_decay 0.5e-6 --normal_class 3;
```
This example trains a Quantum SVDD model where cats (`--normal_class 3`) are considered to be the normal class. 
Autoencoder pretraining is used for parameter initialization.

## Experiments
We can run QSVDD on several different quantum devices, including superconducting processors and photonic processors (IBM IQM Quafu). We also plan to implement this method on more and larger scale chips in the future.
If any academic organization has hardware and would like to run the algorithm on their hardware, please contact the authors.


## License
MIT
