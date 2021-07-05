import math
import torch
from torch.optim.optimizer import Optimizer, required


class VAdam(Optimizer):
    r"""Implements Neural Variable Adaptive Momentum Estimation (VAdam/NVRM-Adam).
    It has be proposed in 
    `Artificial Neural Variability for Deep Learning: On Overfitting, 
    Noise Memorization, and Catastrophic Forgetting`__.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        variability (float, optional): neural variability (default: 0.01)
        num_iters (int, optional): the number of iterations or the number 
            of minibatches (default: 1e3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        noise_type (string, optional): the neural noise type (default: 'Gaussian')
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
        decoupled (bool, optional): decoupled weight decay or L2 regularization 
            (default: False)
    """

    def __init__(self, params, lr=1e-3, variability=1e-2, num_iters=1e3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, noise_type='Gaussian', amsgrad=False, decoupled=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if variability <= 0.0:
            raise ValueError("Invalid variability value: {}".format(variability))
        if num_iters < 0.0:
            raise ValueError("Invalid num_iters: {}".format(num_iters))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if noise_type not in {'Gaussian','Laplace','Uniform'}:
            raise ValueError("Invalid noise_type. Only Gaussian, Laplace, and Uniform are available.")
        defaults = dict(lr=lr, variability=variability, num_iters=num_iters, betas=betas, eps=eps,
                        weight_decay=weight_decay, noise_type=noise_type, amsgrad=amsgrad, decoupled=decoupled)
        super(VAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(VAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('decoupled', False)

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
            
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('VAdam does not support sparse gradients.')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        
                    # The parameter noise in last iteration
                    state['noise1'] = torch.zeros_like(grad, memory_format=torch.preserve_format)
                    # The parameter noise in current iteration
                    state['noise2'] = torch.zeros_like(grad, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Perform decoupled weight decay or L2 Regularization
                if group['decoupled']:
                    p.mul_(1 - group['lr'] * group['weight_decay'])
                else:
                    grad.add_(p.data, alpha=group['weight_decay'])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                p.addcdiv_(exp_avg, denom, value=-step_size)
                
                
                variability = group['variability']
                num_iters = group['num_iters']
                noise_type = group['noise_type']
                noise1 = state['noise1']
                noise2 = state['noise2'] 
                
                if state['step'] % num_iters != 0:
                    if noise_type == 'Gaussian':
                        noise2.mul_(0.).add_(torch.normal(torch.zeros_like(grad),variability))
                    elif noise_type == 'Laplace':
                        noise2.mul_(0.).add_(torch.distributions.laplace.Laplace(torch.zeros_like(grad),variability).sample())
                    elif noise_type == 'Uniform':
                        noise2.mul_(0.).add_(torch.distributions.uniform.Uniform(torch.zeros_like(grad) - variability,torch.zeros_like(grad) + variability).sample())
                    p.add_(noise2 - noise1)
                    noise1.mul_(0.).add_(noise2)
                elif state['step'] % num_iters == 0:
                    p.add_(-noise1)
                    noise1.mul_(0.)

        return loss
