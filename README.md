# artificial-neural-variability-variable-optimizer
The Pytorch Implementation of Variable Optimizers/ Neural Variable Risk Minimization. 
The algortihms are based on our original paper: Artificial Neural Variability.


# The environment is as bellow:
Ubuntu 18.04.4 LTS
Python 3.7.3 
PyTorch 1.4.0



# Code Example: 

You may use it as a standard Pytorch optimizer.

optimizer = variable_optim.VSGD(net.parameters(), lr=lr, variability=vb, num_iters=num_iters)




