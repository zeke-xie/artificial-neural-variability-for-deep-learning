
import torch
from torch.optim.optimizer import Optimizer, required

class VSGD(Optimizer):
    r"""Implements Neural Variable Stochastic Gradient Descent (VSGD/ NVRM-SGD).
    It has be proposed in 
    `Artificial Neural Variability for Deep Learning: On Overfitting, 
    Noise Memorization, and Catastrophic Forgetting`__.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        variability (float, optional): the neural variability scale (default: 0.01)
        num_iters (int, optional): the number of iterations per epoch (default: 1e3)
        momentum (float, optional): momentum factor (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        noise_type (string, optional): the neural noise type (default: 'Gaussian')
        nesterov (bool, optional): enables Nesterov momentum (default: False)
    """

    def __init__(self, params, lr=required, variability=1e-2, num_iters=1e3, momentum=0, dampening=0,
                 weight_decay=0, noise_type='Gaussian', nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if variability <= 0.0:
            raise ValueError("Invalid variability value: {}".format(variability))
        if num_iters < 0.0:
            raise ValueError("Invalid num_iters: {}".format(num_iters))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if noise_type not in {'Gaussian','Laplace','Uniform'}:
            raise ValueError("Invalid noise_type. Only Gaussian, Laplace, and Uniform are available.")

        defaults = dict(lr=lr, variability=variability, num_iters=num_iters, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, noise_type=noise_type, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(VSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(VSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            momentum = group['momentum']
            dampening = group['dampening']
            weight_decay = group['weight_decay']
            nesterov = group['nesterov']
            

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # The parameter noise in last iteration
                    state['noise1'] = torch.zeros_like(d_p, memory_format=torch.preserve_format)
                    # The parameter noise in current iteration
                    state['noise2'] = torch.zeros_like(d_p, memory_format=torch.preserve_format)
                    #state['noise2'].add_(torch.normal(torch.zeros_like(d_p), variability * torch.ones_like(d_p)))
                    
                state['step'] += 1

                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-group['lr'])
                
                variability = group['variability']
                num_iters = group['num_iters']
                noise_type = group['noise_type']

                noise1 = state['noise1']
                noise2 = state['noise2'] 
                
                if state['step'] % num_iters != 0:
                    if noise_type == 'Gaussian':
                        noise2.mul_(0.).add_(torch.normal(torch.zeros_like(d_p),variability))
                    elif noise_type == 'Laplace':
                        noise2.mul_(0.).add_(torch.distributions.laplace.Laplace(torch.zeros_like(d_p),variability).sample())
                    elif noise_type == 'Uniform':
                        noise2.mul_(0.).add_(torch.distributions.uniform.Uniform(torch.zeros_like(d_p) - variability,torch.zeros_like(d_p) + variability).sample())
                    p.add_(noise2 - noise1)
                    noise1.mul_(0.).add_(noise2)
                elif state['step'] % num_iters == 0:
                    p.add_(-noise1)
                    noise1.mul_(0.)

        return loss