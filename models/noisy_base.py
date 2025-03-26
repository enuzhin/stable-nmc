import torch
import torch.nn as nn
from torch.func import functional_call

class NoisyBase(nn.Module):
    def __init__(self, w_max=1, noise_spread=1):
        super(NoisyBase, self).__init__()
        self.w_max = w_max
        self.noise_spread = noise_spread
        self.mean_parameters = []

    def inject_noise(self,param):
        """
        Apply noise regularization:
          a = max(0, param - noise_spread/2)
          b = min(w_max, param + noise_spread/2)
          theta = a + u * (b - a), where u ~ U(0,1)
        """
        a = torch.clamp(param - self.noise_spread / 2, min=0)
        b = torch.clamp(param + self.noise_spread / 2, max=self.w_max)
        u = torch.rand_like(param)
        return a + u * (b - a)

    def noisy_forward(self,input):
        noisy_params = {name: self.inject_noise(param) if param.dim() != 1 else param
                        for name, param in self.named_parameters()}

        return functional_call(self, noisy_params,input)


    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            m.weight.data = torch.abs(m.weight.data)