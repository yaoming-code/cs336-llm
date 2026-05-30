from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # 获取学习率。
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # 获取与 p 关联的状态。
                t = state.get("t", 0)  # 从状态中获取迭代次数，或初始值。
                grad = p.grad.data  # 获取损失对 p 的梯度。
                p.data -= lr / math.sqrt(t + 1) * grad  # 就地更新权重张量。
                state["t"] = t + 1  # 增加迭代次数。
        return loss

weights = torch.nn.Parameter(5 * torch.randn((10, 10))) 
opt = SGD([weights], lr=1e2) 
for t in range(10): 
    opt.zero_grad() # Reset the gradients for all learnable parameters.
    loss = (weights**2).mean() # Compute a scalar loss value. 
    print(loss.cpu().item()) 
    loss.backward() # Run backward pass, which computes gradients.
    opt.step() # Run optimizer step.