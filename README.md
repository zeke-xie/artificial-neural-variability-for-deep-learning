# artificial-neural-variability-for-deep-learning

The Pytorch Implementation of Variable Optimizers/ Neural Variable Risk Minimization. 

The algortihms are proposed in our paper: 
[Artificial Neural Variability for Deep Learning: On Overfitting, Noise Memorization, and Catastrophic Forgetting](https://arxiv.org/abs/2011.06220).


# Why Artificial Neural Variability?

We introduce a neuroscience concept, called neural variability, into deep learning. 

It helps DNNs learn from neuroscience.

At negligible computational and coding costs, our neuroscience-inspired optimization method can 

(1) enhance the robustness to weight perturbation;

(2) improve generalizability;

(3) relieve the memorization of noisy labels;

(4) mitigate catastrophic forgetting.


# How good is Artificial Neural Variability?

![The learning curves of ResNet-34 on CIFAR-10 with 40% asymmetric label noise. NVRM prevents overitting noisy labels effectively, while SGD almost memorizes all noisy labels.](/figure/CIFAR10_acc_resnet34_LabelNoise40.png?raw=true "Title")
*Figure 1. The learning curves of ResNet-34 on CIFAR-10 with 40% asymmetric label noise. NVRM prevents overitting noisy labels effectively, while SGD almost memorizes all noisy labels.*

# The environment is as bellow:

Ubuntu 18.04.4 LTS

Python >= 3.7.3 

PyTorch >= 1.4.0



# Code Example: 

You may use it as a standard Pytorch optimizer.

```python
import variable_optim

optimizer = variable_optim.VSGD(net.parameters(), lr=lr, variability=variability, num_iters=num_iters)
```

# Citing

If you use artificieal neural variabiliy in your work, please cite

```
@article{xie2020artificial,
  title={Artificial Neural Variability for Deep Learning: On Overfitting, Noise Memorization, and Catastrophic Forgetting},
  author={Xie, Zeke and He, Fengxiang and Fu, Shaopeng and Sato, Issei and Tao, Dacheng and Sugiyama, Masashi},
  journal={arXiv preprint arXiv:2011.06220},
  year={2020}
}
```
