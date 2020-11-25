import torch
import torch.optim as optim


class Optimizer(object):
	def __init__(self, params, lr=0.001, betas=(0.9, 0.999), 
				eps=1e-08, warmup_steps=4000):
		self.optimizer = optim.Adam(params, lr, betas, eps)
		self.init_lr = lr
		self.current_lr = lr
		self.warmup_steps=warmup_steps
		self._step = 0
	def step(self):
		self._step+=1
		lr = self.init_lr* min(self._step ** (-0.5), self._step * self.warmup_steps**(-1.5))
		self.current_lr = lr
		self.optimizer.param_groups[0]['lr'] = lr
		self.optimizer.step()
	def zero_grad(self):
		self.optimizer.zero_grad()




