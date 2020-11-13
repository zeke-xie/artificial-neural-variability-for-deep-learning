# artificial-neural-variability-for-deep-learning

The Pytorch Implementation of Variable Optimizers/ Neural Variable Risk Minimization. 

The algortihms are based on our paper: 
[Artificial Neural Variability for Deep Learning: On Overfitting, Noise Memorization, and Catastrophic Forgetting.](https://arxiv.org/abs/2011.06220)


# The environment is as bellow:

Ubuntu 18.04.4 LTS

Python >= 3.7.3 

PyTorch >= 1.4.0



# Code Example: 

You may use it as a standard Pytorch optimizer.

```python
import variable_optim

optimizer = variable_optim.VSGD(net.parameters(), lr=lr, variability=vb, num_iters=num_iters)
```


