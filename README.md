# artificial-neural-variability-variable-optimizer
The Pytorch Implementation of Variable Optimizers/ Neural Variable Risk Minimization. 
The algortihms are based on our original paper: Artificial Neural Variability.


# The environment is as bellow:
Ubuntu 18.04.4 LTS
Python 3.7.3 
PyTorch 1.4.0



# Code Example: 

import variable_optim

lr=0.1

#Set the neural variability scale.
vb = 0.01

#Let num_iters = the length of trainloader.
num_iters = len(trainloader) 

#Define a variable optimizer.
optimizer = variable_optim.VSGD(net.parameters(), lr=lr, variability=vb, num_iters=num_iters)

#Then, you may use it as a standard Pytorch optimizer.
optimizer.zero_grad()
loss_fn(net(input), target).backward()
optimizer.step()
